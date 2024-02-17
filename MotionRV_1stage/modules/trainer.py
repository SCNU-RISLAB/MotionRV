#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import datetime
import os
import time
import imp
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import __init__ as booger


import torch.optim as optim
from tensorboardX import SummaryWriter as Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import warmupLR
from common.cosine_schedule import CosineAnnealingWarmUpRestarts

from modules.MotionRV import MotionRV
from modules.loss.Lovasz_Softmax import Lovasz_softmax, Lovasz_softmax_PointCloud
from modules.loss.boundary_loss import BoundaryLoss
from modules.tools import AverageMeter, iouEval, save_checkpoint, show_scans_in_training, save_to_txtlog, make_log_img

# import torch.backends.cudnn as cudnn
# from icecream import ic
# from modules.ioueval import *
# from modules.SalsaNext import *
# import modules.adf as adf
# from modules.SalsaNextAdf import *
# from torch.autograd import Variable
# from modules.loss.DiceLoss import DiceLoss


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, local_rank, path=None, point_refine=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.path = path  # 这才是预训练模型路径
        self.epoch = 0
        self.point_refine = point_refine
        self.rank = local_rank


        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()

        # put logger where it belongs
        if self.rank == 0:
            self.tb_logger = Logger(self.logdir + "/tb")
            self.info = {"train_update": 0,
                        "train_loss": 0, "train_acc": 0, "train_iou": 0,
                        "valid_loss": 0, "valid_acc": 0, "valid_iou": 0,
                        "best_train_iou": 0, "best_val_iou": 0}

        # 初始化DDP
        # os.environ['MASTER_ADDR'] = '172.26.110.3'
        # os.environ['MASTER_PORT'] = '5500'
        # dist.init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=2, timeout=datetime.timedelta(seconds=5))
        # dist.init_process_group(backend='nccl', init_method=dist_url, rank=local_rank, world_size=2)
        # torch.cuda.set_device(local_rank)

        # get the data
        parserModule = imp.load_source("parserModule",
                                       f"{booger.TRAIN_PATH}/common/dataset/{self.DATA['name']}/parser.py")
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          split='train',
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          sub_kitti=self.ARCH["train"]["sub_kitti"],
                                          gt=True,
                                          knn=False,
                                          shuffle_train=False)  # 使用DDP，shuffle_train=False

        with torch.no_grad():
            # 创建模型                     
            self.model = MotionRV(self.parser.get_n_classes(), self.ARCH)

            # add SyncBN
            if self.ARCH["train"]["syncbn"]:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)  

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)  # cuda
            print("Training in device: ", self.device)

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)  # 将其移动到local_rank对应的GPU上   
            # self.gpu = True
            # if local_rank == 0:
            #     print(model)
            #     save_to_txtlog(self.logdir, 'log.txt', str(model))

        self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)  # output_device=, find_unused_parameters=True
            
        self.set_loss_weight()        
        self.set_loss_function(point_refine)  #####################################
        self.set_optim_scheduler()

        # if need load the pre-trained model from checkpoint
        if self.path is not None:
            self.load_pretrained_model()
    ##############################################init_end************************

    def set_loss_weight(self):
        """
            Used to calculate the weights for each class
            weights for loss (and bias)
        """
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)   # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):   # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:    # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

    def set_loss_function(self, point_refine):
        """
            Used to define the loss function, multiple gpus need to be parallel
            # self.dice = DiceLoss().to(self.device)
            # self.dice = nn.DataParallel(self.dice).cuda()
        """
        self.criterion = nn.NLLLoss(weight=self.loss_w.double()).cuda()
        self.bd = BoundaryLoss(ignore=0).cuda()

        if not point_refine:
            self.ls = Lovasz_softmax(ignore=0).cuda()
        else:
            self.ls = Lovasz_softmax_PointCloud(ignore=0).cuda()

    def set_gpu_cuda(self):
        """
            Used to set gpus and cuda information
        """
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # cudnn.benchmark = True
            # cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)      # spread in gpus
            self.model = convert_model(self.model).cuda() # sync batchnorm
            self.model_single = self.model.module         # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()


    def set_optim_scheduler(self):
        """
            Used to set the optimizer and scheduler
        """
        if self.ARCH["train"]["optimizer"] == 'sgd':
            self.optimizer = optim.SGD([{'params': self.model.parameters()}],  #############################################################adamW
                                    lr=self.ARCH["train"]["lr"],
                                    momentum=self.ARCH["train"]["momentum"],
                                    weight_decay=self.ARCH["train"]["w_decay"])        
            train_loader = self.parser.get_train_set()
        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
            steps_per_epoch = self.parser.get_train_size()
            up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)  
            final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
            self.scheduler = warmupLR(optimizer=self.optimizer,  #####################################cosLR
                                    lr=self.ARCH["train"]["lr"],
                                    warmup_steps=up_steps,
                                    momentum=self.ARCH["train"]["momentum"],
                                    decay=final_decay)       
        elif self.ARCH["train"]["optimizer"] == 'adam':
             self.optimizer = optim.Adam(self.model.parameters(), lr=self.ARCH["train"]["lr"])
             lr_step = []
             for step in self.ARCH["train"]["lr_step"]:
                 lr_step.append(step * len(train_loader))
             self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                milestones=lr_step,
                                                                gamma=self.ARCH["train"]["lr_factor"])            
        else:
            raise ValueError('unsopported optimizer type!')
        
        # self.optimizer = optim.SGD([{'params': self.model.parameters()}],  #############################################################adamW
        #                            lr=self.ARCH["train"]["lr"],
        #                            momentum=self.ARCH["train"]["momentum"],
        #                            weight_decay=self.ARCH["train"]["w_decay"])

        # # Use warmup learning rate
        # # post decay and step sizes come in epochs and we want it in steps
        # steps_per_epoch = self.parser.get_train_size()
        # up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)  
        # final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        # self.scheduler = warmupLR(optimizer=self.optimizer,  #####################################cosLR
        #                           lr=self.ARCH["train"]["lr"],
        #                           warmup_steps=up_steps,
        #                           momentum=self.ARCH["train"]["momentum"],
        #                           decay=final_decay)

    def load_pretrained_model(self):
        """
            If you want to resume training, reload the model
        """
        torch.nn.Module.dump_patches = True
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}  # Map tensors from GPU 0 to GPU rank
        # map_location = torch.device(f'cuda:{self.local_rank}')
        if not self.point_refine:
            checkpoint = "MotionRV"
            w_dict = torch.load(f"{self.path}/{checkpoint}", map_location=map_location)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.optimizer.load_state_dict(w_dict['optimizer'])
            self.epoch = w_dict['epoch'] + 1
            self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
            self.info = w_dict['info']
            print("info", w_dict['info'])
            print("load the pretrained model of MotionRV")
        else:
            checkpoint = "MotionRV_valid_best"
            w_dict = torch.load(f"{self.path}/{checkpoint}", map_location=map_location)
            # self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.model.load_state_dict({k.replace('module.',''):v for k,v in w_dict['state_dict'].items()})
            self.optimizer.load_state_dict(w_dict['optimizer'])
            print("load the coarse model of MotionRV_valid_best")


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) *
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                           iter + 1 + epoch * self.parser.get_train_size()))) + \
            int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def save_to_tensorboard(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            if 'valid_classes' in tag:
                continue # solve the bug of saving tensor type of value
            logger.add_scalar(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def init_evaluator(self):
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.rank, self.ignore_class)  ###################### self.device

    def train(self):

        self.init_evaluator()

        if self.rank == 0:
            str_line = str(self.model)
            save_to_txtlog(self.logdir, 'log.txt', str_line)
            str_line = ''
            str_line = 'The Training strat time: ' + time.asctime() + '\n'
            save_to_txtlog(self.logdir, 'log.txt', str_line)

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):

            # train for 1 epoch
            train_sampler = self.parser.get_train_sampler()
            train_sampler.set_epoch(epoch)

            acc, iou, loss, rec, pcs, update_mean, hetero_l = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                                     model=self.model,
                                                                     criterion=self.criterion,
                                                                     optimizer=self.optimizer,
                                                                     epoch=epoch,
                                                                     evaluator=self.evaluator,  # init_evaluator
                                                                     scheduler=self.scheduler,
                                                                     color_fn=self.parser.to_color,
                                                                     report=self.ARCH["train"]["report_batch"],
                                                                     show_scans=self.ARCH["train"]["show_scans"])

            # update the info dict and save the training checkpoint
            if self.rank == 0:
                self.update_training_info(epoch, acc, iou, loss, rec, pcs, update_mean, hetero_l)
            rand_img = []

            # evaluate on validation set
            if epoch >= self.ARCH["train"]["valid_epoch"]:
                if epoch % self.ARCH["train"]["report_epoch"] == 0 or epoch+1 == self.ARCH["train"]["max_epochs"]:
                    acc, iou, loss, rec, pcs, rand_img, hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                                    model=self.model,
                                                                    criterion=self.criterion,
                                                                    evaluator=self.evaluator,
                                                                    class_func=self.parser.get_xentropy_class_string,
                                                                    color_fn=self.parser.to_color,
                                                                    save_scans=self.ARCH["train"]["save_scans"])
                # if self.rank == 0:
                #     str_line = ("save per validation epoch model!\n" + "*" * 80)
                #     print(str_line)
                #     save_to_txtlog(self.logdir, 'log.txt', str_line)

                #     # save the weights!
                #     state = {'epoch': epoch,
                #             'state_dict': self.model.state_dict(),
                #             'optimizer': self.optimizer.state_dict(),
                #             'info': self.info,
                #             'scheduler': self.scheduler.state_dict()}
                #     save_checkpoint(state, self.logdir, suffix=f"_valid_best_{epoch}")                

                    if self.rank == 0:
                        self.update_validation_info(epoch, acc, iou, loss, rec, pcs, hetero_l)

            # save to tensorboard log
            if self.rank == 0:
                Trainer.save_to_tensorboard(logdir=self.logdir,
                                            logger=self.tb_logger,
                                            info=self.info,
                                            epoch=epoch,
                                            w_summary=self.ARCH["train"]["save_summary"],
                                            model=self.model,
                                            img_summary=self.ARCH["train"]["save_scans"],
                                            imgs=rand_img)

        if self.rank == 0:
            print('Finished Training')
            str_line = 'The Training end time: ' + time.asctime() + '\n'
            save_to_txtlog(self.logdir, 'log.txt', str_line)

        dist.destroy_process_group()  #######################

        return

    def train_epoch(self, train_loader, model, criterion, optimizer,
                    epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):

        losses = AverageMeter()  # 管理一些变量的记录和更新,初始化就reset,调用该类对象的update就变量更新，当读取某个变量的时候，可以通过对象.属性的方式来读取
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()
        rec = AverageMeter()
        pcs = AverageMeter()


        # empty the cache to train now
        # if self.gpu:
        #     torch.cuda.empty_cache()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, 
                _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
            # measure data loading time
            self.data_time_t.update(time.time() - end)

            in_vol = in_vol.cuda(non_blocking=True)
            proj_labels = proj_labels.long().cuda(non_blocking=True)

            # compute output
            output, _ = model(in_vol)
            bdlosss = self.bd(output, proj_labels.long())
            loss_m0 = criterion(torch.log(output.clamp(min=1e-8)).double(), proj_labels.long()).float() + 1.5 * self.ls(output, proj_labels.long())
            loss_m = loss_m0 + bdlosss
            
            optimizer.zero_grad()
            loss_m.backward()
            optimizer.step()

            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
                recall, class_recall = evaluator.getRecall()
                precision, class_precision = evaluator.getPrecision()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            rec.update(class_recall[2].item(), in_vol.size(0))
            pcs.update(class_precision[2].item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) * value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                show_scans_in_training(
                    proj_mask, in_vol, argmax, proj_labels, color_fn)

            if self.rank == 0:  ##########################################
                if i % self.ARCH["train"]["report_batch"] == 0:
                    str_line = ('Lr: {lr:.3e} | '
                                'Update: {umean:.3e} mean,{ustd:.3e} std | '
                                'Epoch: [{0}][{1}/{2}] | '
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                                'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                                'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                                'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                                'IoU {iou.val:.3f} ({iou.avg:.3f}) | '
                                'Recall {rec.val:.3f} ({rec.avg:.3f}) | '
                                'Precision {pcs.val:.3f} ({pcs.avg:.3f}) | '
                                '[{estim}]').format(
                        epoch, i, len(train_loader), batch_time=self.batch_time_t,
                        data_time=self.data_time_t, loss=losses, acc=acc, iou=iou, lr=lr, rec=rec, pcs=pcs,
                        umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i))
                    print(str_line)
                    save_to_txtlog(self.logdir, 'log.txt', str_line)

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, rec.avg, pcs.avg, update_ratio_meter.avg, hetero_l.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans=False):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rec = AverageMeter()
        pcs = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        # if self.gpu:
        #     torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                    _, _, _, _, _, _, _, _, _)\
                    in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):
                
                in_vol = in_vol.cuda(non_blocking=True)
                proj_mask = proj_mask.cuda(non_blocking=True)
                proj_labels = proj_labels.long().cuda(non_blocking=True)

                # compute output
                output, _ = model(in_vol)
                log_out = torch.log(output.clamp(min=1e-8))

                # wce = criterion(log_out, proj_labels)
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out.double(), proj_labels).float()
                loss = wce + jacc

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)

                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(),in_vol.size(0))
                wces.update(wce.mean().item(),in_vol.size(0))                             

                if save_scans:
                    if self.rank == 0:
                    # get the first scan in batch and project points
                        mask_np = proj_mask[0].cpu().numpy()
                        depth_np = in_vol[0][0].cpu().numpy()
                        pred_np = argmax[0].cpu().numpy()
                        gt_np = proj_labels[0].cpu().numpy()
                        out = make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)
                        rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            recall, class_recall = evaluator.getRecall()
            precision, class_precision = evaluator.getPrecision()

            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            rec.update(class_recall[2].item(), in_vol.size(0))
            pcs.update(class_precision[2].item(), in_vol.size(0))

            if self.rank == 0:
                str_line = ("*" * 80 + '\n'
                            'Validation set:\n'
                            'Time avg per batch {batch_time.avg:.3f}\n'
                            'Loss avg {loss.avg:.4f}\n'
                            'Jaccard avg {jac.avg:.4f}\n'
                            'WCE avg {wces.avg:.4f}\n'
                            'Acc avg {acc.avg:.6f}\n'
                            'IoU avg {iou.avg:.6f}\n'
                            'Recall avg {rec.avg:.6f}\n'
                            'Precison avg {pcs.avg:.6f}'
                            ).format(
                                batch_time=self.batch_time_e, loss=losses,
                                jac=jaccs, wces=wces, acc=acc, iou=iou, rec=rec, pcs=pcs)
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

                # print also classwise
                for i, jacc in enumerate(class_jaccard):
                    self.info["valid_classes/" + class_func(i)] = jacc
                    str_line = 'IoU class {i:} [{class_str:}] = {jacc:.6f}'.format(
                        i=i, class_str=class_func(i), jacc=jacc)
                    print(str_line)
                    save_to_txtlog(self.logdir, 'log.txt', str_line)
                str_line = '*' * 80
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

        return acc.avg, iou.avg, losses.avg, rec.avg, pcs.avg, rand_imgs, hetero_l.avg

    def update_training_info(self, epoch, acc, iou, loss, rec, pcs, update_mean, hetero_l):
        # update info
        self.info["train_update"] = update_mean
        self.info["train_loss"] = loss
        self.info["train_acc"] = acc
        self.info["train_iou"] = iou
        self.info["train_recall"] = rec
        self.info["train_precison"] = pcs
        self.info["train_hetero"] = hetero_l

        # remember best iou and save checkpoint
        state = {'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'info': self.info,
                'scheduler': self.scheduler.state_dict()}        
        save_checkpoint(state, self.logdir, suffix="")

        if self.info['train_iou'] > self.info['best_train_iou']:
            print("Best mean iou in training set so far, save model!")
            self.info['best_train_iou'] = self.info['train_iou']
            state = {'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'info': self.info,
                    'scheduler': self.scheduler.state_dict()}
            save_checkpoint(state, self.logdir, suffix="_train_best")

    def update_validation_info(self, epoch, acc, iou, loss,rec, pcs, hetero_l):
        # update info
        self.info["valid_loss"] = loss
        self.info["valid_acc"] = acc
        self.info["valid_iou"] = iou
        self.info["valid_recall"] = rec
        self.info["valid_precison"] = pcs        
        self.info['valid_heteros'] = hetero_l

        # remember best iou and save checkpoint
        if self.info['valid_iou'] > self.info['best_val_iou']:
            str_line = ("Best mean iou in validation so far, save model!\n" + "*" * 80)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)
            self.info['best_val_iou'] = self.info['valid_iou']

            # save the weights!
            state = {'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'info': self.info,
                    'scheduler': self.scheduler.state_dict()}
            save_checkpoint(state, self.logdir, suffix="_valid_best")
            save_checkpoint(state, self.logdir, suffix=f"_valid_best_{epoch}")


#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch.distributed as dist
import os
import random
import numpy as np
import torch
import __init__ as booger

from datetime import datetime

from utils.utils import *
from modules.trainer import Trainer
from modules.tools import save_to_txtlog
from modules.MotionRV import MotionRV


def set_seed(seed=1024, cuda_deterministic=False):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       torch.backends.cudnn.deterministic = True  # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
       torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
       torch.backends.cudnn.deterministic = False
       torch.backends.cudnn.benchmark = True    
    # torch.backends.cudnn.enabled = False
    # If we need to reproduce the results, 
    #    set benchmark = False
    # If we don’t need to reproduce the results, increase the training speed, improve the network performance as much as possible
    #    set benchmark = True

if __name__ == '__main__':
    parser = get_args(flags="train")
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.log = FLAGS.log + '/logs/' + datetime.now().strftime("%Y-%-m-%d-%H:%M") + FLAGS.name  # 都会起一个新的log

    # open arch / data config file
    ARCH = load_yaml(FLAGS.arch_cfg)
    DATA = load_yaml(FLAGS.data_cfg)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    gpus = [int(i) for i in FLAGS.gpus.split(',')]
    world_size = len(gpus)  

   #  port = find_free_port()
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = os.environ["MASTER_PORT"]
    FLAGS.dist_url = '{}:{}'.format(MASTER_ADDR, MASTER_PORT)

   #  dist.init_process_group(backend=FLAGS.dist_backend,
   #                          init_method='env://',
   #                          world_size=world_size,
   #                          rank=FLAGS.local_rank)
    dist.init_process_group(backend=FLAGS.dist_backend, init_method='env://')
    torch.cuda.set_device(FLAGS.local_rank)    

    params = MotionRV(nclasses=3, params=ARCH)
    pytorch_total_params = sum(p.numel() for p in params.parameters() if p.requires_grad)
    del params  # 删除params

    if FLAGS.local_rank == 0:    
       str_line = (
                  "--------------------------------\n"
                  'Training setting:\n'
                  'pwd: {pwd}\n'
                  'dataset: {dataset}\n'
                  'arch_cfg: {arch_cfg}\n'
                  'data_cfg: {data_cfg}\n'
                  'dist_url: {dist_url}\n'
                  'gpus: {gpus}\n'
                  'use SyncBatchNorm: {syncbn}\n'
                  'use MSDSCM: {MSDSCM}\n'
                  'use MSDSRV kitti: {MSDSRV}\n'
                  'use MGAM kitti: {MGAM}\n'
                  'Total of Trainable Parameters: {Parameters}\n'
                  'log: {log}\n'
                  'pretrained: {pretrained}\n'
                  "--------------------------------\n"
                  ).format(pwd=os.getcwd(), dataset=FLAGS.dataset, arch_cfg=FLAGS.arch_cfg, data_cfg=FLAGS.data_cfg,
                           dist_url=FLAGS.dist_url, gpus=FLAGS.gpus, 
                           syncbn=ARCH["train"]["syncbn"], MSDSCM = ARCH["train"]["MSDSCM"],
                           MSDSRV = ARCH["train"]["MSDSRV"], MGAM = ARCH["train"]["MGAM"],
                           Parameters=millify(pytorch_total_params,2),
                           log=FLAGS.log, pretrained=FLAGS.pretrained)
       
       print(str_line)  

       make_logdir(FLAGS=FLAGS, resume_train=False) # create log folder
       check_pretrained_dir(FLAGS.pretrained)       # does model folder exist?，只是打印
       backup_to_logdir(FLAGS=FLAGS)                # backup code and config files to logdir

       save_to_txtlog(FLAGS.log, 'log.txt', str_line)
       str_line = '' 

    set_seed(1024+FLAGS.local_rank)  # 两张卡的local_rank不一样，所以种子不一样
    # create trainer and start the training
    trainer = Trainer(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.local_rank, FLAGS.pretrained)
    trainer.train()

# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
# from icecream import ic

from modules.BaseBlocks import MetaKernel, ResContextBlock, ResBlock, UpBlock, MSDSCM , MSDSRV_ENCODER, MSDSRV_DECODER, MGAM


class MotionRV(nn.Module):
    def __init__(self, nclasses, params, num_batch=None, point_refine=None):
        super(MotionRV, self).__init__()

        self.nclasses = nclasses
        self.use_attention = "MGA"
        self.point_refine = point_refine
        self.use_MGAM = params['train']['MGAM']
        self.MSDSCM = params['train']['MSDSCM']
        self.MSDSRV = params['train']['MSDSRV']

        self.range_channel = 5
        print("Channel of range image input = ", self.range_channel)
        print("Number of residual images input = ", params['train']['n_input_scans'])
        

        if self.MSDSCM:
            self.downCntx = MSDSCM(self.range_channel, 32)  
            self.downCntx2 = MSDSCM(32, 32)
            self.downCntx3 = MSDSCM(32, 32)
        else:                 
            self.downCntx = ResContextBlock(self.range_channel, 32)  
            self.downCntx2 = ResContextBlock(32, 32)
            self.downCntx3 = ResContextBlock(32, 32)

        # / torch.cuda.device_count()
        self.metaConv = MetaKernel(num_batch=int(params['train']['batch_size']) if num_batch is None else num_batch,
                                   feat_height=params['dataset']['sensor']['img_prop']['height'],
                                   feat_width=params['dataset']['sensor']['img_prop']['width'],
                                   coord_channels=self.range_channel)

        if self.MSDSRV:
            self.resBlock1 = MSDSRV_ENCODER(32, 2 * 32, 0.2, pooling=True, drop_out=False)
            self.resBlock2 = MSDSRV_ENCODER(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
            self.resBlock3 = MSDSRV_ENCODER(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
            self.resBlock4 = MSDSRV_ENCODER(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
            self.resBlock5 = MSDSRV_ENCODER(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)
        else:
            self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
            self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
            self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
            self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
            self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)            

        if self.MSDSRV:
            self.upBlock1 = MSDSRV_DECODER(2 * 4 * 32, 4 * 32, 0.2)
            self.upBlock2 = MSDSRV_DECODER(4 * 32, 4 * 32, 0.2)
            self.upBlock3 = MSDSRV_DECODER(4 * 32, 2 * 32, 0.2)
            self.upBlock4 = MSDSRV_DECODER(2 * 32, 32, 0.2, drop_out=False)        
        else:
            self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
            self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
            self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
            self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)            

        if self.use_MGAM:
            self.feature_aggregation = MGAM(32, 32, 32, 3)
        else:
            self.feature_aggregation = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

        # Context Block for residual image 
        if self.MSDSCM: 
            self.RI_downCntx = MSDSCM(params['train']['n_input_scans'], 32)
        else:
            self.RI_downCntx = ResContextBlock(params['train']['n_input_scans'], 32)
        # self.RI_downCntx2 = ResContextBlock(32, 32)
        # self.RI_downCntx3 = ResContextBlock(32, 32)

        if self.MSDSRV:
            self.RI_resBlock1 = MSDSRV_ENCODER(32, 2 * 32, 0.2, pooling=True, drop_out=False)
            self.RI_resBlock2 = MSDSRV_ENCODER(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
            self.RI_resBlock3 = MSDSRV_ENCODER(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
            self.RI_resBlock4 = MSDSRV_ENCODER(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
            # self.RI_resBlock5 = MS_ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)
        else:
            self.RI_resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
            self.RI_resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
            self.RI_resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
            self.RI_resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
            # self.RI_resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        if self.use_attention == "MGA":
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.conv1x1_conv1_channel_wise = nn.Conv2d(32, 32, 1, bias=True)
            self.conv1x1_conv1_spatial = nn.Conv2d(32, 1, 1, bias=True)

            self.conv1x1_layer0_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
            self.conv1x1_layer0_spatial = nn.Conv2d(64, 1, 1, bias=True)

            self.conv1x1_layer1_channel_wise  = nn.Conv2d(128, 128, 1, bias=True)
            self.conv1x1_layer1_spatial = nn.Conv2d(128, 1, 1, bias=True)

            self.conv1x1_layer2_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer2_spatial = nn.Conv2d(256, 1, 1, bias=True)

            self.conv1x1_layer3_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer3_spatial = nn.Conv2d(256, 1, 1, bias=True)

            self.conv1x1_layer4_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer4_spatial = nn.Conv2d(256, 1, 1, bias=True)
        else:
            pass # raise NotImplementedError

    def encoder_attention_module_MGA_tmc(self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial):
        """
            flow_feat_map:  [bsize, 1, h, w]
            feat_vec:       [bsize, channel, 1, 1]
            channel_attentioned_img_feat:  [bsize, channel, h, w]
        """
        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)  
        flow_feat_map = nn.Sigmoid()(flow_feat_map)
        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

    def forward(self, x):
        """
            x: shape [bs, c, h, w],  c = range image channel + num of residual images
            *_downCntx:[bs, .., h, w]
            RI_down0c: [bs, c', h/2, w/2]       RI_down0b:  [bs, c', h, w] 
            RI_down1c: [bs, c'', h/4, w/4]      RI_down1b:  [bs, c'', h/2, w/2] 
            RI_down2c: [bs, c'', h/8, w/8]      RI_down2b:  [bs, c'', h/4, w/4] 
            RI_down3c: [bs, c'', h/16, w/16]    RI_down3b:  [bs, c'', h/8, w/8] 
            up4e: [bs, .., h/8, w/8] 
            up3e: [bs, .., h/4, w/4]
            up2e: [bs, .., h/2, w/2]
            up1e: [bs, .., h, w]
            logits: [bs, num_class, h, w]
        """

        # split the input data to range image (5 channel) and residual images       
        current_range_image = x[:, :self.range_channel, : ,:]  # 5
        residual_images = x[:, self.range_channel:, : ,:]  # 8   

        ###### the Encoder for residual image ######
        RI_downCntx = self.RI_downCntx(residual_images)
        residual_feature =  RI_downCntx

        RI_down0c, RI_down0b = self.RI_resBlock1(RI_downCntx)
        RI_down1c, RI_down1b = self.RI_resBlock2(RI_down0c)
        RI_down2c, RI_down2b = self.RI_resBlock3(RI_down1c)
        RI_down3c, RI_down3b = self.RI_resBlock4(RI_down2c)
        # RI_down5c = self.RI_resBlock5(RI_down3c)

        ###### the Encoder for range image ######
        downCntx = self.downCntx(current_range_image)
        # Use MetaKernel to capture more spatial information
        downCntx = self.metaConv(data=downCntx,
                                coord_data=current_range_image,
                                data_channels=downCntx.size()[1],
                                coord_channels=current_range_image.size()[1],
                                kernel_size=3)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        ###### Bridging two specific branches using MotionGuidedAttention ######
        if self.use_attention == "MGA":
            downCntx = self.encoder_attention_module_MGA_tmc(downCntx, RI_downCntx, self.conv1x1_conv1_channel_wise, self.conv1x1_conv1_spatial)
        elif self.use_attention == "Add":
            downCntx += RI_downCntx
        down0c, down0b = self.resBlock1(downCntx)

        if self.use_attention == "MGA":
            down0c = self.encoder_attention_module_MGA_tmc(down0c, RI_down0c, self.conv1x1_layer0_channel_wise, self.conv1x1_layer0_spatial)
        elif self.use_attention == "Add":
            down0c += RI_down0c
        down1c, down1b = self.resBlock2(down0c)

        if self.use_attention == "MGA":
            down1c = self.encoder_attention_module_MGA_tmc(down1c, RI_down1c, self.conv1x1_layer1_channel_wise, self.conv1x1_layer1_spatial)
        elif self.use_attention == "Add":
            down1c += RI_down1c
        down2c, down2b = self.resBlock3(down1c)

        if self.use_attention == "MGA":
            down2c = self.encoder_attention_module_MGA_tmc(down2c, RI_down2c, self.conv1x1_layer2_channel_wise, self.conv1x1_layer2_spatial)
        elif self.use_attention == "Add":
            down2c += RI_down2c
        down3c, down3b = self.resBlock4(down2c)

        if self.use_attention == "MGA":
            down3c = self.encoder_attention_module_MGA_tmc(down3c, RI_down3c, self.conv1x1_layer3_channel_wise, self.conv1x1_layer3_spatial)
        elif self.use_attention == "Add":
            down3c += RI_down3c

        down5c = self.resBlock5(down3c)

        ###### the Decoder, same as SalsaNext ######
        up4e = self.upBlock1(down5c, down3b, RI_down3b)  # 256->128
        up3e = self.upBlock2(up4e, down2b, RI_down2b)  # 128->128
        up2e = self.upBlock3(up3e, down1b, RI_down1b)  # 128->64
        up1e = self.upBlock4(up2e, down0b, RI_down0b)  # 64->32

        # aggregate above features and get logits under range view
        if self.use_MGAM:
            range_view_logits, up1e = self.feature_aggregation(residual_feature, up1e)
        else:
            range_view_logits, up1e = self.feature_aggregation(up1e)
        range_view_logits = F.softmax(range_view_logits, dim=1)

        return range_view_logits, up1e

# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

from hashlib import sha1
import math
from pickle import FALSE
from numpy.core.numeric import False_
from numpy.lib.twodim_base import diag

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torchvision.transforms.transforms import ToPILImage
from .networks import Backbone
# from .arch_resnet import resnet
# from .arch_resnest import resnest
# from .abc_modules import ABC_Model
# from .deeplab_utils import ASPP, Decoder
# from .aff_utils import PathIndex
from tools.ai.torch_utils import resize_for_tensors
from tools.general.Q_util import *
from core.models.model_util import conv

#######################################################################
# Normalization
#######################################################################
# from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
def conv_bn(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

def conv_dilation(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,dilation=16):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, bias=False,dilation=dilation,padding_mode='circular'),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True), 
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True,dilation=dilation,padding_mode='circular'),
            nn.ReLU(inplace=True), 
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

def get_noliner(features):
            b, c, h, w = features.shape
            if(c==9):
                feat_pd = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
            elif(c==25):
                feat_pd = F.pad(features, (2, 2, 2, 2), mode='constant', value=0)

            diff_map_list=[]
            nn=int(math.sqrt(c))
            for i in range(nn):
                for j in range(nn):
                        diff_map_list.append(feat_pd[:,i*nn+j,i:i+h,j:j+w])
            ret = torch.stack(diff_map_list,dim=1)
            return ret


class SANET_Model_new0(Backbone):

    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       
        logits = self.classifier(x5)
        return logits


class SANET_Model_new1(Backbone):

    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       
        logits = self.classifier(x5)
        logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        logits=poolfeat(logits,probs)
        return logits


 
class SANET_Model_Qbranch(Backbone):

    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.dia1=conv_dilation(True,9, 64,  3, stride=1,dilation=16)
        # self.dia2=conv_dilation(True,ch_q, ch_q,  3, stride=1,dilation=8)
        self.get_NSAM=nn.Sequential(
                        conv_dilation(True,9, 64,  3, stride=1,dilation=16),
                        conv(True,64, ch_q,  3, stride=2), ##包含了卷积/正则化/Relu/maxpooling
                        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        conv(True,ch_q,ch_q*2, 3, stride=2),
                        conv_dilation(True,ch_q*2, ch_q*2,  3, stride=1,dilation=8),
                        conv(True,ch_q*2,ch_q*4, 3, stride=2),
                        conv(True,ch_q*4,25, 3, stride=2),
                        nn.Softmax(dim=1)     
                        )

    
    def forward(self, inputs,probs):
        # b,c,w,h=probs.shape
        # x1 = self.stage1(inputs)
        # x2 = self.stage2(x1)
        # x3 = self.stage3(x2)
        # x4 = self.stage4(x3)
        # x5 = self.stage5(x4)       
        # logits = self.classifier(x5)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        # logits=poolfeat(logits,probs)
        # dia_prob= self.dia1(probs)
        nsam=self.get_NSAM(probs)
                      
        return nsam       

class SANET_Model_Qbranch3(Backbone):

    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        # self.dia1=conv_dilation(True,9, 64,  3, stride=1,dilation=1)
        # self.dia2=conv_dilation(True,ch_q, ch_q,  3, stride=1,dilation=8)
        self.get_NSAM=nn.Sequential(
                    conv(True,9, ch_q,  3, stride=2), ##包含了卷积/正则化/Relu/maxpooling
                    # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    # conv_dilation(True,ch_q, ch_q,  3, stride=1,dilation=8),
                    conv(True,ch_q,ch_q*2, 3, stride=2),
                    conv(True,ch_q*2,ch_q*4, 3, stride=2),
                    conv(True,ch_q*4,25, 3, stride=2),
                    nn.Softmax(dim=1)     
                    )

    
    def forward(self, inputs,probs):
        # b,c,w,h=probs.shape
        # x1 = self.stage1(inputs)
        # x2 = self.stage2(x1)
        # x3 = self.stage3(x2)
        # x4 = self.stage4(x3)
        # x5 = self.stage5(x4)       
        # logits = self.classifier(x5)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        # logits=poolfeat(logits,probs)
        # dia_prob= self.dia1(probs)
        nsam=self.get_NSAM(probs)
                      
        return nsam       


class SANET_Model_new16(Backbone):

    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.prefusion=nn.Sequential(
                    # conv(True, 9,   64, kernel_size=3, stride=2),
                    # # conv(True, 64,  64, kernel_size=3),
                    # conv(True, 64, 128, kernel_size=3, stride=2),
                    # # conv(True, 64, 128, kernel_size=3, stride=2)
                        conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                        conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
        self.seconv=nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(ch_q*4,+128, (ch_q*4,+128)//16, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d((ch_q*4,+128)//16, ch_q*4,+128, 1, 1, 0),
                nn.Sigmoid()
                )                 
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
      
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

        q=self.prefusion(probs) 
        x55 = self.qcov_for_x5(x5.detach())
        q=torch.cat([x55,q],dim=1)
        q_se=self.seconv(q)
        q=q*q_se
        aff22 = self.qcov2(q)

        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits



    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.prefusion=nn.Sequential(
                        conv_bn(True,9, ch_q,  3, stride=2), ##包含了卷积/正则化/Relu/maxpooling

                        conv_bn(True,ch_q,ch_q*2, 3, stride=2),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=2),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=2),

                        )
                
        self.qcov2=nn.Sequential(
                conv(False,ch_q*4+2048, int(1024),3),
                conv(False,1024,512,3),
                conv(False,512,256,3),
                conv(False,256,18,1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       

        q=self.prefusion(probs) 
        logits = self.classifier(x5)

        x55 = x5.detach()
        q=torch.cat([x55,q],dim=1)
        # q_se=self.seconv(q)
        # q=q*q_se
        aff22 = self.qcov2(q)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)
        logits =torch.cat([bg,fg],dim=1)
        return logits


class SANET_Model_new_base(Backbone):

    def __init__(self, model_name, num_classes=21,process=32,fgORall=True):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=process
        self.outc=9*2

        if not (fgORall):

              self.outc=9*num_classes

        self.get_qfeats=nn.Sequential(
                        conv_dilation(True,9,ch_q,  3, stride=1,dilation=16),
                        conv(True,ch_q, ch_q,  3, stride=2), ##包含了卷积/正则化/Relu/maxpooling
                        conv(True,ch_q,ch_q*2, 3, stride=2),
                        conv(True,ch_q*2,ch_q*4, 3, stride=2),
                        conv(True,ch_q*4,ch_q*4, 3, stride=2),
                        )
                
        self.get_tran_conv=nn.Sequential(
                conv(False,ch_q*4+2048, int(1024),3),
                conv(False,1024,256,3),
                conv(False,256,128,3),
                conv(False,128,  self.outc,1),
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

    def get_x5_features(self,inputs):
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return    x5
    

    def get_tconv_cam(self,logits,tconv):
        if(self.outc==18):
            bg_aff=get_noliner(F.softmax(tconv[:,:9],dim=1))#torch.sum(fg_aff).max()# fg_aff[0,:,10:20,10:20].detach().cpu().numpy()
            fg_aff=get_noliner(F.softmax(tconv[:,9:],dim=1))#torch.sum(aff22).min()
            bg= upfeat(logits[:,0:1],bg_aff,1,1)
            fg= upfeat(logits[:,1:],fg_aff,1,1)
            logits =torch.cat([bg,fg],dim=1)
            return logits
        else:
            logits_list=[]
            for i in range (logits.shape[1]):
                    cur_aff=get_noliner(F.softmax(tconv[:,i*9:(i+1)*9],dim=1))#torch.sum(cur_aff,dim=1).min()
                    cur_c= upfeat(logits[:,i:i+1],cur_aff,1,1)
                    logits_list.append(cur_c)
            logits =torch.cat(logits_list,dim=1)
            return logits
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x5 =self.get_x5_features(inputs)

        logits = self.classifier(x5)

        q=self.get_qfeats(probs) 
        q=torch.cat([x5.detach(),q],dim=1)
        tconv = self.get_tran_conv(q)

        logits = self.get_tconv_cam(logits,tconv)
        return logits
   
    def get_parameter_groups1(self, print_fn=print):
        groups = ([], [], [], [],[],[],[],[])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name:
                if 'weight' in name:
                    # print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    # print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)
                    
            # scracthed weights
            else:
                if('tran_conv' in name ):
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[4].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[5].append(value)
                elif('qfeats' in name):
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[6].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[7].append(value)
                else:
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[2].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[3].append(value)
        return groups




class SANET_Model_new_base1(SANET_Model_new_base):
        def __init__(self, model_name, num_classes=21,process=64,fgORall=False):
            super().__init__(model_name, num_classes, process,fgORall)
            ch_q=process
  
            # self.load_state_dict(torch.load('experiments/models/cam_batch8/2021-11-05 01:42:11_eps.pth'),strict=False) 
        def get_conv_cam(self,logits,tconv):
            if(self.outc==18):
                bg_aff=F.softmax(tconv[:,:9],dim=1)#torch.sum(bg_aff,dim=1).max()
                fg_aff=F.softmax(tconv[:,9:],dim=1)#torch.sum(aff22).min()
                bg= upfeat(logits[:,0:1],bg_aff,1,1)
                fg= upfeat(logits[:,1:],fg_aff,1,1)
                logits =torch.cat([bg,fg],dim=1)
                return logits
            else:
                logits_list=[]
                for i in range (logits.shape[1]):
                        cur_aff=get_noliner(F.softmax(tconv[:,i*9:(i+1)*9],dim=1))#torch.sum(cur_aff,dim=1).min()
                        cur_c= upfeat(logits[:,i:i+1],cur_aff,1,1)
                        logits_list.append(cur_c)
                logits =torch.cat(logits_list,dim=1)
                return logits
        def forward(self, inputs,probs):
            b,c,w,h=probs.shape
            # with torch.no_grad():
            x5 =self.get_x5_features(inputs)

            logits = self.classifier(x5)

            q=self.get_qfeats(probs) 
            q=torch.cat([x5.detach(),q],dim=1)
            tconv = self.get_tran_conv(q)

            logits = self.get_conv_cam(logits,tconv)
            return logits




import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo
from tools.ai.demo_utils import crf_inference
from .deeplab_utils import ASPP, Decoder

from tools.ai.torch_utils import make_cam

from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model
from tools.general.Q_util import *
from core.models.model_util import conv

#######################################################################
# Normalization
#######################################################################
class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

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

from collections import OrderedDict 

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:

            if('moco' in model_name ): 
                state_dict = torch.load("models_ckpt/moco_v2_800ep_pretrain.pth.tar")['state_dict']
                model_name=model_name[:-5]
            elif('mm' in model_name ): 
                state_dict = torch.load("models_ckpt/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth")['state_dict']
                model_name=model_name[:-3]
            elif('detco' in model_name ):
                state_dict = torch.load("models_ckpt/detco_200ep.pth")
                model_name=model_name[:-6]
            elif('dino' in model_name ):
                state_dict = torch.load("models_ckpt/dino_resnet50_pretrain.pth")
                model_name=model_name[:-5]
            else:
                state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
                state_dict.pop('fc.weight')
                state_dict.pop('fc.bias')
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)
            # self.initialize(self.model.modules())

            
            # for k, v in state_dict.items():
            #     name = k[15:]   # remove `vgg.`，即只取vgg.0.weights的后面几位
            #     if(name[:2]=="fc") or (name[:2]=="r."):
            #         continue
            #     new_state_dict[name] = v 
            # state_dict=  new_state_dict
            #state_dict = torch.load("models_ckpt/dino_resnet50_pretrain.pth")
            
            self.model.load_state_dict(state_dict)
            print("load pretrained model from resnet")
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

class CAM_Model(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.ala2 = nn.Conv2d(2048, 2048, 1, bias=False)
        self.ala1 = nn.Conv2d(1024, 1024, 1, bias=False)

    def forward(self, inputs,pcm=0):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x).detach()
        x4 = self.stage4(x)
        # ala1=self.ala1(F.adaptive_avg_pool2d( x4.detach(), 1))
        # ala1=torch.sigmoid(ala2)
        # x4=x4*ala1
        
        x5 = self.stage5(x4)
        # ala2=self.ala2(F.adaptive_avg_pool2d( x5.detach(), 1))
        # ala2=torch.sigmoid(ala2)
        # x5=x5*ala2
        logits = self.classifier(x5)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        logits_min =(F.adaptive_avg_pool2d( self.classifier(x5), 1))
        
        return logits,logits_min
    
class SP_CAM_Model(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=32
        self.outc=9*2

        self.get_qfeats=nn.Sequential(
                        # conv_dilation(True,9,ch_q,  3, stride=1,dilation=16),
                        conv(True,9, ch_q,  4, stride=4), 
                        conv(True,ch_q, ch_q*2,  4, stride=4), 
                        conv(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv(True,ch_q*4,ch_q*4, 3, stride=1),
                        )
        # self.get_qfeats=nn.Sequential(
        # # conv_dilation(True,9,ch_q,  3, stride=1,dilation=16),
        # conv(True,9, ch_q,  4, stride=2), 
        # conv(True,ch_q, ch_q,  3, stride=2), 
        
        # conv(True,ch_q,ch_q*2, 3, stride=2),
        # conv(True,ch_q*2,ch_q*4, 3, stride=2),
        # conv(True,ch_q*4,ch_q*4, 3, stride=2),
        # )
        self.get_tran_conv=nn.Sequential(
                conv(False,ch_q*4+2048, int(1024),3),
                conv(False,1024,256,1),
                conv(False,256,  self.outc,1),
            )   
        # self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.classifier = conv(True,2048,num_classes,1)

    def get_x5_features(self,inputs):
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return   x2, x5
    

    def get_sp_cam(self,logits,deconv_para):
        bg= upfeat(logits[:,0:1],deconv_para[:,:9],1,1)
        fg= upfeat(logits[:,1:],deconv_para[:,9:],1,1)
        logits =torch.cat([bg,fg],dim=1)
        return logits

    def DRM(self,probs,x5):
        q=self.get_qfeats(probs) 
        deconv_parameters = self.get_tran_conv(torch.cat([x5.detach(),q],dim=1))
        bg_para=get_noliner(F.softmax(deconv_parameters[:,:9],dim=1))#torch.sum(fg_aff).max()# fg_aff[0,:,10:20,10:20].detach().cpu().numpy()
        fg_para=get_noliner(F.softmax(deconv_parameters[:,9:],dim=1))#torch.sum(aff22).min()
        deconv_parameters= torch.cat([bg_para,fg_para],dim=1)
        return  deconv_parameters
    def forward(self, inputs,probs):
        # b,c,w,h=probs.shape
        x4,x5 =self.get_x5_features(inputs)
        logits = self.classifier(x5)
        logits_min = self.classifier(self.global_average_pooling_2d(x5, keepdims=True))
        
        deconv_parameters= self.DRM(probs,x5)
        logits = self.get_sp_cam(logits,deconv_parameters)
        return logits,logits_min
   
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
                if('qfeats' in name ):
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[4].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[5].append(value)
                elif('tran_conv' in name):
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

class SP_CAM_Model(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=32
        self.outc=9*2

        self.get_qfeats=nn.Sequential(
                        # conv_dilation(True,9,ch_q,  3, stride=1,dilation=16),
                        conv(True,9, ch_q,  4, stride=4), 
                        conv(True,ch_q, ch_q*2,  4, stride=4), 
                        conv(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv(True,ch_q*4,ch_q*4, 3, stride=1),
                        )
        # self.get_qfeats=nn.Sequential(
        # # conv_dilation(True,9,ch_q,  3, stride=1,dilation=16),
        # conv(True,9, ch_q,  4, stride=2), 
        # conv(True,ch_q, ch_q,  3, stride=2), 
        
        # conv(True,ch_q,ch_q*2, 3, stride=2),
        # conv(True,ch_q*2,ch_q*4, 3, stride=2),
        # conv(True,ch_q*4,ch_q*4, 3, stride=2),
        # )
        self.get_tran_conv=nn.Sequential(
                conv(False,ch_q*4+2048, int(1024),3),
                conv(False,1024,256,1),
                conv(False,256,  self.outc,1),
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        # self.classifier = conv(True,2048,num_classes,1)

    def get_x5_features(self,inputs):
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return   x2, x5
    

    def get_sp_cam(self,logits,deconv_para):
        bg= upfeat(logits[:,0:1],deconv_para[:,:9],1,1)
        fg= upfeat(logits[:,1:],deconv_para[:,9:],1,1)
        logits =torch.cat([bg,fg],dim=1)
        return logits

    def DRM(self,probs,x5):
        q=self.get_qfeats(probs) 
        deconv_parameters = self.get_tran_conv(torch.cat([x5.detach(),q],dim=1))
        bg_para=get_noliner(F.softmax(deconv_parameters[:,:9],dim=1))#torch.sum(fg_aff).max()# fg_aff[0,:,10:20,10:20].detach().cpu().numpy()
        fg_para=get_noliner(F.softmax(deconv_parameters[:,9:],dim=1))#torch.sum(aff22).min()
        deconv_parameters= torch.cat([bg_para,fg_para],dim=1)
        return  deconv_parameters
    def forward(self, inputs,probs,with_feat=False):
        # b,c,w,h=probs.shape
        x4,x5 =self.get_x5_features(inputs)
        logits = self.classifier(x5)
        logits_min = self.classifier(self.global_average_pooling_2d(x5, keepdims=True))
        
        deconv_parameters= self.DRM(probs,x5)
        logits = self.get_sp_cam(logits,deconv_parameters)
        if(with_feat):
            return logits,logits_min,x4
            
        else:
            
            return logits,logits_min
   
class Classifier_rethink(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix',beta=1):
        super().__init__(model_name, num_classes, mode)
        self.num_classes=num_classes
        self.beta=beta

        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

        self.fbconv= nn.Sequential(
            nn.Conv2d(2048, 2, 1, padding=0, bias=False),
        )
      
        
        self.ala1= nn.Sequential(
            nn.Conv2d(1024, 1024, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.ala2= nn.Sequential(
            nn.Conv2d(2048, 2048, 1, padding=0, bias=False),
            nn.Sigmoid()
            
        )
        self.head= nn.Sequential( 
                                 ASPP(output_stride=16, norm_fn=group_norm),
            nn.Conv2d(256, 21, 1, padding=0, bias=False),
        )
        self.initialize([self.fbconv,self.classifier])


    def  forward(self, input, with_cam=True,mask=None,pcm=1):
        x1 = self.stage1(input)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x5=x5.detach()
        x4=x4.detach()
        
        
        
        b,c,h,w=x5.shape#.view(b,20,2,h,w)

        feats=self.classifier(x5)
        logits = self.classifier(self.global_average_pooling_2d(x5, keepdims=True) ).view(-1, self.num_classes)
        pred_soft=self.head(torch.cat([x4,x5],dim=1))
        if(with_cam):
            if(pcm==1):
                # feats = self.repcm(feats,x4,1,0.5)
                pred_soft = self.repcm(pred_soft,x4,1,0.5)



        return pred_soft ,feats 
    def repcm(self,feats,x4,it=1,th=0.5,pcm=1,):
                x4=torch.cat([x4],dim=1)
                b,c,h,w=x4.shape
                x4=x4.view(b,c,-1)
                x4=F.normalize(x4,dim=1)
                aff = torch.bmm(x4.transpose(1,2),x4)
                aff[aff<th]=0
                aff[aff>th]=1
                aff = aff**pcm
                aff=aff/aff.sum(1,True)
                logits_flat=feats.view(b,feats.shape[1],-1)#aff.max()
                for i in range(it):
                     logits_flat=torch.bmm (logits_flat,aff)
                feats=logits_flat.view(b,feats.shape[1],h,w)
                return feats
                
class SP_CAM_Model2(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=32
        self.outc=9
        
        self.get_qfeats=nn.Sequential(
                        conv(True,9, ch_q,  4, stride=4), 
                        conv(True,ch_q, ch_q*4,  4, stride=4), 
                        conv(False,ch_q*4,ch_q*4, 3, stride=1),
                        )
        self.x4_feats=nn.Sequential(
                        conv(True,1024,128, 1, stride=1),
                        )     
        self.x5_feats=nn.Sequential(
                        conv(True,2048,128, 1, stride=1),

                        ) 
        self.get_tran_conv5=nn.Sequential(
                conv(False,ch_q*4+128, 128,3),
                conv(False,128,  self.outc,1),
                nn.Softmax(1)
                
            )  
        self.get_tran_conv4=nn.Sequential(
                conv(False,ch_q*4+128, 128,3),
                conv(False,128,  self.outc,1),
                nn.Softmax(1)
            )   
     
        
        self.classifier = nn.Sequential(nn.Conv2d(2048, num_classes, 1, bias=False))



    def forward(self, inputs,probs,labels=None,pcm=0,th=0.5):
        # b,c,w,h=probs.shape
        q_feat=self.get_qfeats(probs) 
        
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        x4 = self.stage4(x3)
        x4_dp=self.get_tran_conv4(torch.cat([self.x4_feats(x4.detach()),q_feat],dim=1))
        x4=upfeat(x4,x4_dp,1,1)
        x5 = self.stage5(x4)
        x5_dp=self.get_tran_conv5(torch.cat([self.x5_feats(x5.detach()),q_feat],dim=1))
        logits = self.classifier(x5)
        logits_min = self.classifier(F.adaptive_avg_pool2d(x5, 1))
        logits=upfeat(logits,x5_dp,1,1)
        if(pcm>0):
            x4=torch.cat([x4],dim=1)
            b,c,h,w=x4.shape
            x4=x4.view(b,c,-1)
            x4=F.normalize(x4,dim=1)
            aff_b = torch.bmm(x4.transpose(1,2),x4)
            aff=torch.clamp(aff_b,0.01,0.999)
            # aff[aff<th]=0
            aff=aff**th
            aff=aff/aff.sum(1,True)
            logits_flat=logits.view(b,21,-1)#aff.max()
            for i in range(pcm):
                logits_flat=torch.bmm (logits_flat,aff)
            logits=logits_flat.view(b,21,h,w)
            
        return logits,logits_min
   


class SP_CAM_Model3(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=32
        self.outc=9
        
        self.get_qfeats=nn.Sequential(
                        conv(True,9, ch_q,  4, stride=4), 
                        conv(True,ch_q, ch_q*4,  4, stride=4), 
                        conv(False,ch_q*4,ch_q*4, 3, stride=1),
                        )
        self.x4_feats=nn.Sequential(
                        conv(True,1024,128, 1, stride=1),
                        )     
        self.x5_feats=nn.Sequential(
                        conv(True,2048,128, 1, stride=1),

                        ) 
        self.get_tran_conv5=nn.Sequential(
                conv(False,128, 256,3),
                conv(False,256,  self.outc,1),
                nn.Softmax(1)
                
            )  
        self.get_tran_conv4=nn.Sequential(
                conv(False,128, 256,3),
                conv(False,256,  self.outc,1),
                nn.Softmax(1)
            )   
        self.ala2 = nn.Sequential(  nn.Conv2d(2048, 128, 1, bias=False),
                                 nn.ReLU(),
                                  nn.Conv2d(128, 2048, 1, bias=False),
                                  nn.Sigmoid())
        self.ala1 = nn.Sequential(  nn.Conv2d(1024, 64, 1, bias=False),
                                 nn.ReLU(),
                                  nn.Conv2d(64, 1024, 1, bias=False),
                                  nn.Sigmoid())
        
        self.classifier = nn.Sequential(nn.Conv2d(2048, num_classes, 1, bias=False))



    def forward(self, inputs,probs,labels=None,pcm=0,th=0.5):
        # b,c,w,h=probs.shape
        # q_feat=self.get_qfeats(probs) 
        
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        x4 = self.stage4(x3)
        x4_dp=self.get_tran_conv4(torch.cat([self.x4_feats(x4)],dim=1))
        # x4_dp=F.softmax(x4_dp,dim=1)
        ala1=self.ala1(F.adaptive_avg_pool2d( x4, 1))
        x4=x4*ala1
        x4=upfeat(x4,x4_dp,1,1)
        x5 = self.stage5(x4)
        ala2=self.ala2(F.adaptive_avg_pool2d( x5.detach(), 1))
        # # x5=x5*torch.sigmoid(ala2)
        x5_dp=self.get_tran_conv5(torch.cat([self.x5_feats(x5)],dim=1))
        x5=x5*ala2
        logits = self.classifier(x5)
        logits_min = self.classifier(F.adaptive_avg_pool2d(x5, 1))
        logits=upfeat(logits,x5_dp,1,1)
        
        # logits_min =self.classifier(F.adaptive_avg_pool2d( (x5), 1))

      #  logits_min =(self.global_average_pooling_2d( self.classifier(x5), keepdims=True))
        if(pcm>0):

            x4=torch.cat([x4],dim=1)
            b,c,h,w=x4.shape
            x4=x4.view(b,c,-1)
            x4=F.normalize(x4,dim=1)
            aff_b = torch.bmm(x4.transpose(1,2),x4)
            aff=torch.clamp(aff_b,0.01,0.999)
            # th=0.5
            aff[aff<th]=0
            # aff=F.relu(aff-th)
            # aff=F.relu()
            # aff[aff>th]=1
            #aff[aff>0.8]=0.2
            aff=aff/aff.sum(1,True)
            logits_flat=logits.view(b,21,-1)#aff.max()
            for i in range(pcm):
                logits_flat=torch.bmm (logits_flat,aff)
            logits=logits_flat.view(b,21,h,w)
            
        return logits,logits_min
   

class Classifier(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes

        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True) 
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits

class Classifier_For_Positive_Pooling(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes
        
        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True) 
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits

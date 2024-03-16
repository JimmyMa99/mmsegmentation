


from weakref import ref
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from torch.nn.modules.loss import _Loss
class SP_CAM_Loss(_Loss):
  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               reduction='mean'):
    super(SP_CAM_Loss, self).__init__(size_average, reduce, reduction)
    self.args=args
    self.fg_c_num =20 if  args.dataset == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,fg_cam,sailencys):#sailencys.max()
      
        #region cls_loss
        b, c, h, w = fg_cam.size()
        imgmin_mask=sailencys.sum(1,True)!=0
        #endregion
        #region sal_loss
        # else:
        #   affmat=calc_affmat(prob).detach()
        #   crf_inference()
        #   for i in range(20):
        #     sailencys=refine_with_affmat(sailencys,affmat)
        sailencys = F.interpolate(sailencys.float(), size=(h, w))


        # fg_cam=F.softmax(logits,dim=1)*mask
        bg=1-torch.max(fg_cam,dim=1,keepdim=True)[0]**1
        
        nnn = torch.max((1-bg.detach()*imgmin_mask).view(b,1,-1),dim=2)[0]>self.args.ig_th
        nnn2 = torch.max((bg.detach()*imgmin_mask).view(b,1,-1),dim=2)[0]>self.args.ig_th
        nnn=nnn*nnn2
        imgmin_mask=nnn.view(b,1,1,1)*imgmin_mask
        
        
        probs=torch.cat([bg,fg_cam],dim=1)
        # probs=fg_cam
        b,c,h,w=probs.shape
        
        origin_f=F.normalize(sailencys.detach(),dim=1)
        # origin_f=sailencys.detach()
        # imgmin_mask=imgmin_mask*cam_mask.detach()#cam_mask.sum()
        probs=probs*imgmin_mask
        f_min=pool_feat_2(probs,origin_f)
        up_f=up_feat_2(probs,f_min)
        # up_f=F.normalize(up_f,dim=1)
        aaa=(origin_f*up_f).sum(dim=1,keepdim=True)#aaa.min()
        aaa=torch.clamp(aaa+self.args.clamp_rate,0.01,0.99)
        
        
        
        sal_loss=-torch.log(aaa+1e-5)#*lbl.unsqueeze(-1).unsqueeze(-1).cuda()
        # sal_loss =F.mse_loss(up_f,origin_f,reduce=False)
        # sal_loss=(sal_loss*imgmin_mask).sum()/(torch.sum(imgmin_mask)+1e-5)#imgmin_mask.sum()
        sal_loss=(sal_loss*imgmin_mask).sum()/(torch.sum(imgmin_mask)+1e-5)
        return sal_loss
class SP_CAM_Loss2(_Loss):
  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               reduction='mean'):
    super(SP_CAM_Loss2, self).__init__(size_average, reduce, reduction)
    self.args=args
    self.fg_c_num =20 if  args.dataset == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,fg_cam,sailencys):#sailencys.max()
      
        #region cls_loss
        b, c, h, w = fg_cam.size()
        imgmin_mask=sailencys.sum(1,True)!=0
        #endregion
        #region sal_loss
        # else:
        #   affmat=calc_affmat(prob).detach()
        #   crf_inference()
        #   for i in range(20):
        #     sailencys=refine_with_affmat(sailencys,affmat)
        sailencys = F.interpolate(sailencys.float(), size=(h, w))


        # fg_cam=F.softmax(logits,dim=1)*mask
        bg=1-torch.max(fg_cam,dim=1,keepdim=True)[0]**1
        
        
        nnn = torch.max((1-bg.detach()*imgmin_mask).view(b,1,-1),dim=2)[0]>self.args.ig_th
        nnn2 = torch.max((bg.detach()*imgmin_mask).view(b,1,-1),dim=2)[0]>self.args.ig_th
        nnn=nnn*nnn2
        if(nnn.sum()==0):
          nnn=torch.ones(nnn.shape).cuda()
        imgmin_mask=nnn.view(b,1,1,1)*imgmin_mask
        probs=torch.cat([bg,fg_cam],dim=1)
        # probs=fg_cam
        b,c,h,w=probs.shape
        
        # imgmin_mask=imgmin_mask*cam_mask.
        # h()#cam_mask.sum()
        probs1=probs*imgmin_mask
        # origin_f=sailencys.detach()
        
        origin_f=F.normalize(sailencys.detach(),dim=1)
        origin_f=origin_f*imgmin_mask
        f_min=pool_feat_2(probs1,origin_f)
        up_f=up_feat_2(probs1,f_min)#
       

        sal_loss =F.mse_loss(up_f,origin_f,reduce=False)#sal_loss.mean()
        # sal_loss=(sal_loss*imgmin_mask).sum()/(torch.sum(imgmin_mask)+1e-5)#imgmin_mask.sum()
        sal_loss=(sal_loss*imgmin_mask).sum()/(torch.sum(imgmin_mask)+1e-3)

        return sal_loss


class SP_CAM_Loss2_aff(_Loss):
  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               reduction='mean'):
    super(SP_CAM_Loss2, self).__init__(size_average, reduce, reduction)
    self.args=args
    self.fg_c_num =20 if  args.dataset == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,fg_cam,sailencys):#sailencys.max()
      
        #region cls_loss
        b, c, h, w = fg_cam.size()
        imgmin_mask=sailencys.sum(1,True)!=0
        #endregion
        #region sal_loss
        # else:
        #   affmat=calc_affmat(prob).detach()
        #   crf_inference()
        #   for i in range(20):
        #     sailencys=refine_with_affmat(sailencys,affmat)
        sailencys = F.interpolate(sailencys.float(), size=(h, w))


        # fg_cam=F.softmax(logits,dim=1)*mask
        bg=1-torch.max(fg_cam,dim=1,keepdim=True)[0]**1
        
        
        nnn = torch.max((1-bg.detach()*imgmin_mask).view(b,1,-1),dim=2)[0]>self.args.ig_th
        nnn2 = torch.max((bg.detach()*imgmin_mask).view(b,1,-1),dim=2)[0]>self.args.ig_th
        nnn=nnn*nnn2
        if(nnn.sum()==0):
          nnn=torch.ones(nnn.shape).cuda()
        imgmin_mask=nnn.view(b,1,1,1)*imgmin_mask
        
        probs=torch.cat([bg,fg_cam],dim=1)
        # probs=fg_cam
        b,c,h,w=probs.shape
        sailencys=F.normalize(sailencys.detach(),dim=1)
        sal_f=sailencys.view(b,3,-1)
        aff=torch.bmm(sal_f.transpose(1,2),sal_f)#aff.mean()
        aff =torch.clamp(aff,0.01,0.99)
        
        th=0.85
        aff[aff<th]=0
        aff[aff>th]=1
        #aff[aff>0.8]=0.2#aff.sum()/144
        aff=aff/(aff.sum(1,True)+1e-5)
        origin_f=probs
        origin_flat=torch.bmm (origin_f.view(b,c,-1),aff)
        up_f= origin_flat.view(b,-1,h,w)#
        sal_loss=torch.abs(up_f-origin_f).sum(1,True)
        # sal_loss =F.mse_loss(up_f,origin_f,reduce=False)s
        # sal_loss=(sal_loss*imgmin_mask).sum()/(torch.sum(imgmin_mask)+1e-5)#imgmin_mask.sum()
        sal_loss=(sal_loss*imgmin_mask).sum()/(torch.sum(imgmin_mask)+1e-3)

        return sal_loss
        
class QLoss(_Loss):

  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               relu_t=0.9,
               reduction='mean'):
    super(QLoss, self).__init__(size_average, reduce, reduction)
    self.relu_t=relu_t
    self.relufn =nn.ReLU()
    self.args=args
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,prob,labxy_feat,imgids,pos_weight= 0.003, kernel_size=16):


            # this wrt the slic paper who used sqrt of (mse)

            # rgbxy1_feat: B*50+2*H*W
            # output : B*9*H*w
            # NOTE: this loss is only designed for one level structure

            # todo: currently we assume the downsize scale in x,y direction are always same
            S = kernel_size
            m = pos_weight

            b, c, h, w = labxy_feat.shape
            pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
            reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

            loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]
            if(labxy_feat.shape[1]==5):
                loss_map_sem = reconstr_feat[:, :-2, :, :] - labxy_feat[:, :-2, :, :]
                loss_sem =  torch.norm(loss_map_sem, p=2, dim=1).sum() / (b * S)
            else:
                # self def cross entropy  -- the official one combined softmax
                logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
                loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b    
                
          
            
            loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

            # empirically we find timing 0.005 tend to better performance
            loss_sem_sum =   0.005 * loss_sem
            loss_pos_sum = 0.005 * loss_pos
            loss_sum =   loss_pos_sum +loss_sem_sum


            return loss_sum, loss_sem_sum,loss_pos_sum
    
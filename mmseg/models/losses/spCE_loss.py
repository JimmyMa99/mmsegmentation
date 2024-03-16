# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from tools.ASAMrefine import ASAMrefiner
import sys
sys.path.append('/media/ders/mazhiming/eps_test/EPS-main/')
from utils.loss import CrossEntropy2d
from torchvision import ops
import pdb


def _stack_batch_globalcam(batch_data_samples: SampleList) -> Tensor:
    globalcam = [
        data_sample.cam.data for data_sample in batch_data_samples
    ]
    return torch.stack(globalcam, dim=0)

def _stack_batch_crop_imgs(batch_data_samples: SampleList) -> Tensor:
    crop_imgs = [
        data_sample.crop_imgs.data for data_sample in batch_data_samples
    ]
    return torch.stack(crop_imgs, dim=0)

def _stack_batch_boxes(batch_data_samples: SampleList) -> Tensor:
    boxes = [
        data_sample.crop_boxes.data for data_sample in batch_data_samples
    ]
    return torch.stack(boxes, dim=0)



@MODELS.register_module()
class SuperpixelCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_sp',
                 wsss=False,
                 depth=False,
                 superpixel=False,
                 ws_thr=0.8,
                 weight_sp=0.5,
                 renum=70,
                 total_patch=16,
                 avg_non_ignore=False):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.wsss = wsss
        self.depth = depth
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self.superpixel = superpixel
        self.ws_thr = ws_thr
        self.weight_sp=weight_sp
        self.renum=renum
        self.total_patch=total_patch
        self.L_CE=CrossEntropy2d(255)
        if self.superpixel:
            self.refiner = ASAMrefiner(model_path="/media/ders/mazhiming/eps_test/EPS-main/models_ckpt/Q_model_final.pth")
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')


        self.cls_criterion = self.spCEloss
        
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                siamese=False,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            siamese=siamese,
            **kwargs)
        return loss_cls

    def spCEloss(self,pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None,
            ignore_index=-100,
            avg_non_ignore=False,
            siamese=False,
            **kwargs):
        """
        Calculate the binary CrossEntropy loss.
        """

        # pdb.set_trace()
        if siamese:
            datasample=label[3]
            cam_local=label[2]
            label=label[0]
            cam=_stack_batch_globalcam(datasample)
            bs,c,h,w=cam.shape
            bxs=datasample[0].patch_size
            crop_imgs=datasample[0].crop_imgs.data
            boxes=_stack_batch_boxes(datasample)
            boxes = boxes.reshape(bs * bxs, 5)
            box_ind = torch.cat([torch.zeros(bxs).fill_(i) for i in range(bs)])
            boxes[:, 0] = box_ind
            boxes = boxes.cuda(non_blocking=True).type_as(cam)

            n, c, h, w = cam_local.shape
            cam_soft=F.sigmoid(cam)
            feat_aligned = ops.roi_align(cam_soft, boxes, (h, w), 1 / 8.0)
            cam_local_s=cam_local
            cam_local_s=F.sigmoid(cam_local_s)
            # pdb.set_trace()
            cam_local_s=self.refiner.run_infer(crop_imgs, cam_local_s, self.renum)
            cam_local=F.interpolate(cam_local_s,(h,w))
            # label, weight, valid_mask = _expand_onehot_labels(
            #         label, weight, pred.shape, ignore_index)
            # pdb.set_trace()



            feat_aligned[:,:-1]=feat_aligned[:,:-1]#*label_local.unsqueeze(2).unsqueeze(3)
            pselb_crop=(cam_local==cam_local[:,:].max(dim=1,keepdim=True)[0]).to(dtype=torch.float)
            pselb_global=(feat_aligned==feat_aligned.max(dim=1,keepdim=True)[0]).to(dtype=torch.float)
            pselb_diff=torch.pow(torch.sub(pselb_crop,pselb_global),2)/(h*w)#差异矩阵(比例)
            weight_same=torch.sum(torch.sum(pselb_diff,dim=2),dim=2)#16=bs*ps
            weight_same=torch.sub(1,weight_same)
            weight_same=torch.where(weight_same>self.ws_thr,torch.ones_like(weight_same),weight_same)
            cam_local=F.softmax(cam_local,dim=1)
            pselb=torch.argmax(cam_local[:,:],dim=1)
            loss=torch.sum(torch.stack([self.L_CE(feat_aligned[i,:].unsqueeze(0),pselb[i,:].unsqueeze(0)\
                ,weight=weight_same[i])*self.weight_sp for i in range(self.total_patch)]))/self.total_patch

        else:
            loss=torch.tensor(0.0).cuda()
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

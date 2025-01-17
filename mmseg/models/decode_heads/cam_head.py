# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from typing import List
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead, BaseDecodeHead_wsss

import torch.nn.functional as F
from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from ..losses import accuracy
import pdb

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from mmengine.model import BaseModule

@MODELS.register_module()
class CAMHead(BaseDecodeHead_wsss):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        # pdb.set_trace()
        # self.fc8=torch.nn.Conv2d(4096, self.num_classes, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.conv_seg.weight)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # x = self._transform_inputs(inputs)
        #pdb.set_trace()
        if isinstance(inputs, list):
            x = inputs[-1]#取最后一层4096
        else:
            x = inputs[-1]
            # raise TypeError('inputs must be a list of Tensor')
        feats = self.conv_seg(x)
  
        return feats
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType,siamese: bool=False) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples,siamese)

        return losses
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        #get output from feature map
        # n,c,h,w=feats.size()
        # output=F.adaptive_avg_pool2d(feats, kernel_size=(h, w), padding=0)
        # output = output.view(output.size(0), -1)
        #calculate 
        # output = self.cls_seg(output)
        return output
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    def _stack_batch_sal(self, batch_data_samples: SampleList) -> Tensor:
        gt_sal_maps = [
            data_sample.sal_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_sal_maps, dim=0)
    
    def _stack_batch_depth(self, batch_data_samples: SampleList) -> Tensor:
        gt_depth_maps = [
            data_sample.gt_depth_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_depth_maps, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList,siamese) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        sal_map = self._stack_batch_sal(batch_data_samples)
        # depth_map = self._stack_batch_depth(batch_data_samples)
        # seg_label=torch.stack(batch_data_samples, dim=0)
        # sal_map=torch.stack(batch_data_samples, dim=0)
        loss = dict()
        cam=seg_logits.clone()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        # pdb.set_trace()
        seg_label = seg_label.squeeze(1)
        if not siamese:
            for i in range(seg_label.size(0)):
                cam_detached = cam[i].detach()
                batch_data_samples[i].set_metainfo({'cam': cam_detached})
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    [seg_label,sal_map,cam,batch_data_samples],#,depth_map],
                    weight=seg_weight,
                    ignore_index=self.ignore_index,siamese=siamese)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        if not siamese:
            loss['acc_seg'] = accuracy(
                seg_logits, seg_label, ignore_index=self.ignore_index)
        else:
            loss['acc_seg'] = torch.tensor(0.0).cuda()
        return loss,batch_data_samples
    def predict(self, inputs: List[Tensor],
            batch_img_metas: List[dict]):
        """Forward function for testing.

        Args:
            inputs (List[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs)
        # pdb.set_trace()
        # seg_logits_=self.predict_by_feat(seg_logits, batch_img_metas)
        seg_logits = torch.roll(seg_logits, shifts=1, dims=1)

        return seg_logits
    

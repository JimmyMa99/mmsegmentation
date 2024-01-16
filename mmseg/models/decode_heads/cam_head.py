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
        # pdb.set_trace()
        if isinstance(inputs, list):
            x = inputs[-1]#取最后一层4096
        else:
            raise TypeError('inputs must be a list of Tensor')
        feats = self.conv_seg(x)
  
        return feats

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

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
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
        # seg_label=torch.stack(batch_data_samples, dim=0)
        # sal_map=torch.stack(batch_data_samples, dim=0)
        loss = dict()
        cam=seg_logits
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    [seg_label,sal_map,cam],
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss
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

        return self.predict_by_feat(seg_logits, batch_img_metas)
    

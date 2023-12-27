# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

import torch.nn.functional as F
import pdb


@MODELS.register_module()
class CAMHead(BaseDecodeHead):
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
        self.fc8=torch.nn.Conv2d(4096, self.num_classes, 1)
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
        if isinstance(inputs, list):
            x = inputs[-1]#取最后一层4096
        feats = self.fc8(x)
  
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

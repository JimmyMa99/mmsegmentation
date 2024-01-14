# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS, VISUALIZERS
from mmseg.utils import register_all_modules

from mmseg.datasets.transforms import LoadAnnotations_SAL, RandomFlip, RandomCrop, PhotoMetricDistortion, PackSegInputs
from mmcv.transforms import RandomResize,LoadImageFromFile
from mmseg.utils.misc import stack_batch

import torch

img_path = '/media/ders/mazhiming/mmseg4wsss/mmsegmentation/data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
gt_path = '/media/ders/mazhiming/mmseg4wsss/mmsegmentation/data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'
sal_path = '/media/ders/mazhiming/mmseg4wsss/mmsegmentation/data/VOCdevkit/VOC2012/saliency_map/2007_000032.png'

randomCrop=RandomCrop(crop_size=(448,448),cat_max_ratio=0.75)
randomFlip=RandomFlip(prob=0.5)
# randomResize=RandomResize(scale_range=(0.5, 2.0),interpolation='bilinear',scale=(448,448),keep_ratio=True)
loadImg=LoadImageFromFile()
loadAnn=LoadAnnotations_SAL()
photoMetricDistortion=PhotoMetricDistortion()
packSegInputs=PackSegInputs()


results = dict(
    img_path=img_path,
    seg_map_path=gt_path,
    sal_path=sal_path,
    reduce_zero_label=False,
    seg_fields=[])

x=loadImg(results)
x=loadAnn(x)
x=randomCrop(x)
x=randomFlip(x)
x=photoMetricDistortion(x)
x=packSegInputs(x)

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
pad_val=0
seg_pad_val=255
size=(448,448)
type='SegDataPreProcessor'
dataSamples=x['data_samples']
inputs=x['inputs']

outputdata,padded_samples=stack_batch(inputs=[inputs], data_samples=[dataSamples],  size=size)

padded_samples[0].sal_map.data.size()
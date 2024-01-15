# %%
# Copyright (c) OpenMMLab. All rights reserved.
import random
random.seed(0)
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS, VISUALIZERS
from mmseg.utils import register_all_modules

from mmseg.datasets.transforms import LoadAnnotations_SAL, RandomFlip, RandomCrop, PhotoMetricDistortion, PackSegInputs
from mmcv.transforms import RandomResize,LoadImageFromFile
from mmseg.utils.misc import stack_batch

from mmseg.models.data_preprocessor import SegDataPreProcessor

import numpy as np


# %%
img_path = '/media/ders/mazhiming/mmseg4wsss/mmsegmentation/data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
gt_path = '/media/ders/mazhiming/mmseg4wsss/mmsegmentation/data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'
sal_path = '/media/ders/mazhiming/mmseg4wsss/mmsegmentation/data/VOCdevkit/VOC2012/saliency_map/2007_000032.png'

# %%
class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_arr = np.asarray(img)
        normalized_img = np.empty_like(img_arr, np.float32)

        normalized_img[..., 0] = (img_arr[..., 0] / 255. - self.mean[0]) / self.std[0]
        normalized_img[..., 1] = (img_arr[..., 1] / 255. - self.mean[1]) / self.std[1]
        normalized_img[..., 2] = (img_arr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return normalized_img

# %%
randomCrop=RandomCrop(crop_size=(448,448),cat_max_ratio=0.75)
randomFlip=RandomFlip(prob=0.5)
randomResize=RandomResize(ratio_range=(1.0,1.0),scale=(256,512),keep_ratio=True)
loadImg=LoadImageFromFile()
loadAnn=LoadAnnotations_SAL()
photoMetricDistortion=PhotoMetricDistortion(brightness_delta=77,contrast_range=(0.7,1.3),saturation_range=(0.7,1.3),hue_delta=26)#brightness_delta=77,contrast_range=(0.7,1.3),saturation_range=(0.7,1.3),hue_delta=0.1
packSegInputs=PackSegInputs()
normalize = Normalize()



# %%
results = dict(
    img_path=img_path,
    seg_map_path=gt_path,
    sal_path=sal_path,
    reduce_zero_label=False,
    seg_fields=[])

# %%
test_x=loadImg(results)
test_x=loadAnn(test_x)
print(test_x['img'].shape,test_x['sal_map'].shape)



# %%
test_x=randomResize(test_x)
print(test_x['img'].shape,test_x['sal_map'].shape,test_x['gt_seg_map'].shape)

# %%
print(test_x['img'].shape,test_x['sal_map'].shape,test_x['gt_seg_map'].shape)

# %%
import matplotlib.pyplot as plt
plt.imshow(test_x['img'], cmap='gray',)

# %%
plt.imshow(test_x['sal_map'], cmap='gray',)

# %%
test_x=randomFlip(test_x)
# print(test_x['img'].shape,test_x['sal_map'].shape)

# %%
print(test_x['img'].shape,test_x['sal_map'].shape)

# %%
test_x=photoMetricDistortion(test_x)
print(test_x['img'].shape,test_x['sal_map'].shape)

# %%
test_x=randomCrop(test_x)
print(test_x['img'].shape,test_x['sal_map'].shape)

# %%
test_x

# %%
test_x=packSegInputs(test_x)
# print(test_x['input'].shape,test_x['sal_map'].shape)

# %%
print(test_x['inputs'].shape,test_x['data_samples'].sal_map.data.shape)
segDataPpreprocessor=SegDataPreProcessor(bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        448,
        448,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],)

# %%
test_x['inputs']=[test_x['inputs']]
test_x['data_samples']=[test_x['data_samples']]
test_x=segDataPpreprocessor(test_x,training=True)

# %%
print(test_x['inputs'].shape,test_x['data_samples'][0].sal_map.data.shape)

# %%
def transform_img(results):
    x=loadImg(results)
    x=loadAnn(x)
    print('--------load-------------------')
    print(x['img'].shape,x['sal_map'].shape)
    x=randomResize(x)
    print('--------resize-------------------')
    print(x['img'].shape,x['sal_map'].shape)
    x=randomFlip(x)
    print('--------flip-------------------')
    print(x['img'].shape,x['sal_map'].shape)
    x=photoMetricDistortion(x)#jetter
    print('--------jetter-------------------')
    print(x['img'].shape,x['sal_map'].shape)
    x=randomCrop(x)
    print('--------crop-------------------')
    print(x['img'].shape,x['sal_map'].shape)
    x=packSegInputs(x)
    print('--------pack-------------------')
    print(x['inputs'].shape,x['data_samples'].sal_map.data.shape)
    return x

# %%
results = dict(
    img_path=img_path,
    seg_map_path=gt_path,
    sal_path=sal_path,
    reduce_zero_label=False,
    seg_fields=[])
x=transform_img(results)
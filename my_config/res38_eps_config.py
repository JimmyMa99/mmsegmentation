crop_size = (
    448,
    448,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
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
    ],
    type='SegDataPreProcessor')
dataset_type = 'PascalVOCDataset'
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(type='WideRes38'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        # dropout_ratio=0.1,
        in_channels=4096,
        in_index=3,
        loss_decode=[dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False,use_mask=False,wsss=True),
                        dict(loss_weight=1.0, type='EPSLoss', use_sigmoid=False,use_mask=False,eps_wsss=True),],
        # norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=21,
        pool_scales=(
            1,
            2,
            3,
            1,#56*56->4096channels
        ),
        type='CAMHead'),
    pretrained='data/models/res38d_mxnet.pth',
    test_cfg=dict(mode='whole'),
    type='WSSSEncoderDecoder')
# norm_cfg = dict(requires_grad=True, type='SyncBN')

optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005,)
optim_wrapper = dict(
    clip_grad=None,
    optimizer=optimizer,
    type='OptimWrapper',
    paramwise_cfg = dict(
    custom_keys={
    'decode_head.fc8': dict(lr_mult=10.),
    }))

#lr衰减
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=20000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
# test_cfg = dict(
#      crop_size=
#         256
#     , mode='slide', stride=
#         170
#     )
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/train.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='data/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='CAMIoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(keep_ratio=True, scale=(
    #     448,
    #     448,
    # ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=500)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_SAL'),
    dict(
        keep_ratio=True,
        ratio_range=(
            1.0,
            1.0,
        ),
        scale=(
            256,
            512,
        ),
        type='RandomResize'),

    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion',brightness_delta=77, contrast_range=(0.7, 1.3),  # 调整对比度的范围
    saturation_range=(0.7, 1.3),  # 调整饱和度的范围
    hue_delta=26),
    dict(cat_max_ratio=0.75, crop_size=(
        448,
        448,
    ), type='RandomCrop'),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='ImageSets/Segmentation/aug.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassAug',sal_path='saliency_map'),
        data_root='data/VOCdevkit/VOC2012',
        pipeline=train_pipeline,
        type='PascalVOCDataset_Sal'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))

tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/train.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='data/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            # dict(keep_ratio=True, scale=(
            #     448,
            #     448,
            # ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
#####################不用改#####################
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))
################################################

work_dir = 'work_dirs/wsss_voc12_res38'
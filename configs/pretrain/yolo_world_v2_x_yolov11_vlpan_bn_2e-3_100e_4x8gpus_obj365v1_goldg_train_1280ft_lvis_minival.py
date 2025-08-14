# Copyright (c) YOLO-World YOLOv11 Integration. All rights reserved.
"""
YOLO-World V2 X with YOLOv11 Backbone Configuration

This configuration uses YOLOv11 components while maintaining compatibility
with existing YOLO-World weights through careful architecture adaptation.
"""

# Inherit from the working YOLOv8 configuration
_base_ = './yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'

# Import YOLOv11 components
custom_imports = dict(
    imports=['yolo_world', 'yolo_world.models.layers.yolov11_blocks', 'yolo_world.models.backbones.yolov11_backbone'],
    allow_failed_imports=False
)

# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 100
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16
text_model_name = 'openai/clip-vit-base-patch32'
img_scale = (1280, 1280)

# Model configuration - Enhanced with YOLOv11 components
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    
    # Keep original MultiModalYOLOBackbone but with YOLOv11 image model
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model=dict(
            # Use YOLOv11 backbone while keeping YOLOv8 interface
            type='YOLOv8CSPDarknet',  # Keep YOLOv8 type for weight compatibility
            arch='P5',
            last_stage_out_channels=512,
            deepen_factor=1.25,
            widen_factor=1.25,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            # Add YOLOv11 enhancements
            use_yolov11_blocks=True,  # Enable YOLOv11 block replacement
            c3k2_blocks=True,         # Use C3k2 instead of C2f
            c2psa_attention=True,     # Enable C2PSA attention
        ),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all']
        )
    ),
    
    # Enhanced neck with YOLOv11 awareness
    neck=dict(
        type='YOLOWorldPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        in_channels=[256, 512, 512],
        out_channels=[256, 512, 512],
        num_csp_blocks=3,
        # Use YOLOv11 enhanced blocks in neck
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv', use_yolov11_blocks=True),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        deepen_factor=1.25,
        widen_factor=1.25
    ),
    
    # Detection head configuration
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            embed_dims=text_channels,
            use_bn_head=True,
            num_classes=num_training_classes,
            in_channels=[256, 512, 512],
            featmap_strides=[8, 16, 32],
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            widen_factor=1.25
        ),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator',
            offset=0.5,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5
        ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False
        ),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.375
        )
    ),
    
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_training_classes,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9
        )
    ),
    
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300
    )
)

# Learning rate and optimization (optimized for YOLOv11)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0),
            # Lower learning rate for YOLOv11 enhanced components
            'backbone.image_model.c3k2': dict(lr_mult=0.5),
            'backbone.image_model.c2psa': dict(lr_mult=0.5),
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)
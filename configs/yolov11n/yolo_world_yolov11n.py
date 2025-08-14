# Copyright (c) YOLO-World YOLOv11n Integration. All rights reserved.
"""
YOLO-World with YOLOv11n (Nano) Configuration

This configuration adapts YOLO-World to use YOLOv11n backbone weights.
YOLOv11n is the smallest and fastest version of YOLOv11.
"""

# Inherit from base configuration
_base_ = '../pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'

# Import YOLOv11 components
custom_imports = dict(
    imports=['yolo_world', 'yolo_world.models.layers.yolov11_blocks', 'yolo_world.models.backbones.yolov11_backbone'],
    allow_failed_imports=False
)

# YOLOv11n specific parameters
# YOLOv11n: depth=0.33, width=0.25 (much smaller than YOLOv11x)
depth_multiple = 0.33  # Model depth scaling
width_multiple = 0.25  # Model width scaling

# Channel configurations for YOLOv11n
backbone_out_channels = [64, 128, 256]  # YOLOv11n output channels
neck_in_channels = [64, 128, 256]       # Corresponding neck input channels
neck_out_channels = [64, 128, 256]      # Neck output channels

# Text and embedding dimensions
text_channels = 512
neck_embed_channels = [32, 64, 128]     # Smaller for nano version
neck_num_heads = [2, 4, 8]              # Smaller attention heads

# Training parameters
base_lr = 1e-3          # Lower learning rate for smaller model
weight_decay = 0.01     # Lower weight decay
max_epochs = 100
batch_size_per_gpu = 32 # Can use larger batch size due to smaller model

# Model configuration
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=80,
    num_test_classes=1203,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    
    # Multi-modal backbone with YOLOv11n
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='YOLOv8CSPDarknet',  # Use YOLOv8 interface for compatibility
            arch='P5',
            last_stage_out_channels=256,  # YOLOv11n last stage
            deepen_factor=depth_multiple,
            widen_factor=width_multiple,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        )
    ),
    
    # Adapted neck for smaller channels
    neck=dict(
        type='YOLOWorldPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        in_channels=neck_in_channels,
        out_channels=neck_out_channels,
        num_csp_blocks=2,  # Fewer blocks for nano version
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        deepen_factor=depth_multiple,
        widen_factor=width_multiple
    ),
    
    # Detection head adapted for smaller channels
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            embed_dims=text_channels,
            use_bn_head=True,
            num_classes=80,
            in_channels=neck_out_channels,
            featmap_strides=[8, 16, 32],
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            widen_factor=width_multiple
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
            num_classes=80,
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

# Optimizer configuration (adapted for nano model)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=batch_size_per_gpu
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # Lower LR for text model
            'logit_scale': dict(weight_decay=0.0),
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)

# Training dataloader (larger batch size for nano model)
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    # ... other dataloader configs remain the same
)

# Input image scale (can be smaller for nano model)
img_scale = (640, 640)  # Standard input size
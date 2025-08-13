# Copyright (c) YOLO-World Integration Project. All rights reserved.
"""
YOLO-World V2 Large with YOLOv11 Backbone Configuration

This configuration integrates YOLOv11's improved backbone with YOLO-World's
open-vocabulary detection capabilities.

Key improvements over YOLOv8-based YOLO-World:
- C3k2 blocks for better efficiency
- C2PSA attention mechanism for spatial awareness
- Enhanced feature extraction
- Improved multi-scale processing
"""

# Inherit base configuration from original YOLO-World
_base_ = '../pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'

# Custom imports for YOLOv11 components
custom_imports = dict(
    imports=['yolo_world', 'yolo_world.models.layers.yolov11_blocks', 'yolo_world.models.backbones.yolov11_backbone'],
    allow_failed_imports=False
)

# Model configuration with YOLOv11 backbone
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=80,
    num_test_classes=1203,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    
    # YOLOv11 Multi-Modal Backbone
    backbone=dict(
        type='YOLOv11MultiModalBackbone',
        image_model=dict(
            type='YOLOv11Backbone',
            arch='YOLOv11l',
            depth_multiple=1.0,
            width_multiple=1.0,
            out_indices=(4, 6, 10),  # P3, P4, P5 equivalent layers
            frozen_stages=-1,
            norm_eval=False,
        ),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'],
            dropout=0.0,
            training_use_cache=True,
        ),
        frozen_stages=-1,
        with_text_model=True,
        # Feature fusion configuration
        feature_fusion_cfg=dict(
            type='attention',  # or 'adaptive', 'simple'
            text_channels=512,
        )
    ),
    
    # Enhanced PAFPN for YOLOv11 features
    neck=dict(
        type='YOLOWorldPAFPN',
        guide_channels=512,
        embed_channels=[128, 256, 512],  # Adjusted for YOLOv11 backbone
        num_heads=[4, 8, 16],
        in_channels=[256, 512, 1024],    # YOLOv11 backbone output channels
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        block_cfg=dict(type='CSPLayerWithTwoConv'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    
    # Detection head remains compatible
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            embed_dims=512,
            use_bn_head=True,
            bn_head_cfg=dict(
                type='BNContrastiveHead',
                embed_dims=512,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                use_einsum=True
            ),
            num_classes=1203,
            in_channels=[256, 512, 1024],
            featmap_strides=[8, 16, 32],
            num_base_priors=1
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
            loss_weight=1.0
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
            loss_weight=1.5 / 4
        )
    ),
    
    # Training configuration
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=80,
            use_ciou=True,
            topk=10,
            alpha=1,
            beta=6,
            eps=1e-9
        )
    ),
    
    # Testing configuration  
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300
    )
)

# Training hyperparameters (optimized for YOLOv11)
base_lr = 1e-3  # Slightly lower learning rate for stability
weight_decay = 0.025  # Reduced weight decay
max_epochs = 80  # Fewer epochs due to better convergence

# Optimizer configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # Lower LR for frozen text model
            'backbone.image_model': dict(lr_mult=1.0),  # Full LR for YOLOv11 backbone
        }
    )
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1000,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - 10, 1)]
)

# Evaluation settings
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='auto'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Logging configuration
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# Loading settings
load_from = None
resume = False

# Experiment tracking
experiment_name = 'yolo_world_v2_l_yolov11_backbone'
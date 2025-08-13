# Copyright (c) Tencent Inc. All rights reserved.
# Basic brick modules for PAFPN based on CSPLayers

from .yolo_bricks import (
    CSPLayerWithTwoConv,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    ImagePoolingAttentionModule,
    RepConvMaxSigmoidCSPLayerWithTwoConv,
    RepMaxSigmoidCSPLayerWithTwoConv
    )

# YOLOv11 Components
from .yolov11_blocks import (
    YOLOv11Conv,
    YOLOv11Bottleneck,
    C3k,
    C3k2,
    PSABlock,
    MultiHeadAttention,
    C2PSA,
    SPPF
)

__all__ = ['CSPLayerWithTwoConv',
           'MaxSigmoidAttnBlock',
           'MaxSigmoidCSPLayerWithTwoConv',
           'RepConvMaxSigmoidCSPLayerWithTwoConv',
           'RepMaxSigmoidCSPLayerWithTwoConv',
           'ImagePoolingAttentionModule',
           # YOLOv11 Components
           'YOLOv11Conv',
           'YOLOv11Bottleneck', 
           'C3k',
           'C3k2',
           'PSABlock',
           'MultiHeadAttention',
           'C2PSA',
           'SPPF']

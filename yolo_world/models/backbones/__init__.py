# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet / YOLOv11 (New Integration)
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    PseudoLanguageBackbone)

# YOLOv11 Integration Components
from .yolov11_backbone import (
    YOLOv11Backbone,
    YOLOv11CSPBackbone,
    YOLOv11MultiScaleBackbone)

from .yolov11_mm_backbone import (
    YOLOv11MultiModalBackbone,
    AttentionFusion,
    AdaptiveFusion)

__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    # YOLOv11 Components
    'YOLOv11Backbone',
    'YOLOv11CSPBackbone', 
    'YOLOv11MultiScaleBackbone',
    'YOLOv11MultiModalBackbone',
    'AttentionFusion',
    'AdaptiveFusion'
]

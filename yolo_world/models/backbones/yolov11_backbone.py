# Copyright (c) YOLO-World Integration Project. All rights reserved.
"""
YOLOv11 Backbone for YOLO-World Integration

This module provides a YOLOv11-based backbone that can be integrated
into YOLO-World's multi-modal architecture while maintaining 
compatibility with text processing components.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptConfigType

from ..layers.yolov11_blocks import (
    YOLOv11Conv, C3k2, C2PSA, SPPF
)


@MODELS.register_module()
class YOLOv11Backbone(BaseModule):
    """
    YOLOv11 Backbone adapted for YOLO-World
    
    This backbone extracts multi-scale features compatible with 
    YOLO-World's text-image fusion requirements.
    
    Args:
        depth_multiple (float): Depth multiplier for scaling model depth
        width_multiple (float): Width multiplier for scaling model width
        out_indices (tuple): Output feature map indices
        frozen_stages (int): Stages to freeze during training
        norm_eval (bool): Whether to set norm layers to eval mode
    """
    
    # YOLOv11 architecture configuration
    # Format: [from, repeats, module, args]
    arch_settings = {
        'YOLOv11n': [
            [-1, 1, 'YOLOv11Conv', [64, 3, 2]],     # 0-P1/2
            [-1, 1, 'YOLOv11Conv', [128, 3, 2]],    # 1-P2/4
            [-1, 2, 'C3k2', [128, False, 0.25]],    # 2
            [-1, 1, 'YOLOv11Conv', [256, 3, 2]],    # 3-P3/8
            [-1, 2, 'C3k2', [256, False, 0.25]],    # 4
            [-1, 1, 'YOLOv11Conv', [512, 3, 2]],    # 5-P4/16
            [-1, 2, 'C3k2', [512, False, 0.5]],     # 6
            [-1, 1, 'YOLOv11Conv', [1024, 3, 2]],   # 7-P5/32
            [-1, 2, 'C3k2', [1024, False, 0.5]],    # 8
            [-1, 1, 'SPPF', [1024, 5]],             # 9
            [-1, 1, 'C2PSA', [1024, 1]],            # 10
        ],
        'YOLOv11s': [
            [-1, 1, 'YOLOv11Conv', [64, 3, 2]],
            [-1, 1, 'YOLOv11Conv', [128, 3, 2]],
            [-1, 2, 'C3k2', [128, False, 0.25]],
            [-1, 1, 'YOLOv11Conv', [256, 3, 2]],
            [-1, 2, 'C3k2', [256, False, 0.25]],
            [-1, 1, 'YOLOv11Conv', [512, 3, 2]],
            [-1, 2, 'C3k2', [512, False, 0.5]],
            [-1, 1, 'YOLOv11Conv', [1024, 3, 2]],
            [-1, 2, 'C3k2', [1024, False, 0.5]],
            [-1, 1, 'SPPF', [1024, 5]],
            [-1, 1, 'C2PSA', [1024, 1]],
        ],
        'YOLOv11l': [
            [-1, 1, 'YOLOv11Conv', [64, 3, 2]],
            [-1, 1, 'YOLOv11Conv', [128, 3, 2]],
            [-1, 2, 'C3k2', [128, False, 0.25]],
            [-1, 1, 'YOLOv11Conv', [256, 3, 2]],
            [-1, 2, 'C3k2', [256, False, 0.25]],
            [-1, 1, 'YOLOv11Conv', [512, 3, 2]],
            [-1, 2, 'C3k2', [512, False, 0.5]],
            [-1, 1, 'YOLOv11Conv', [1024, 3, 2]],
            [-1, 2, 'C3k2', [1024, False, 0.5]],
            [-1, 1, 'SPPF', [1024, 5]],
            [-1, 1, 'C2PSA', [1024, 1]],
        ]
    }
    
    def __init__(self,
                 arch: str = 'YOLOv11l',
                 depth_multiple: float = 1.0,
                 width_multiple: float = 1.0,
                 out_indices: Tuple[int] = (4, 6, 10),  # P3, P4, P5 equivalent
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        self.arch = arch
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        # Build layers from architecture settings
        self.layers = self._build_layers()
        
        # Initialize frozen stages
        self._freeze_stages()
    
    def _build_layers(self) -> nn.ModuleList:
        """Build YOLOv11 layers from architecture settings"""
        
        if self.arch not in self.arch_settings:
            raise ValueError(f"Unknown architecture: {self.arch}")
        
        layers = nn.ModuleList()
        channels = [3]  # Input channels
        
        for i, (f, n, m, args) in enumerate(self.arch_settings[self.arch]):
            # Calculate input channels
            if f == -1:
                in_channels = channels[-1]
            else:
                in_channels = channels[f]
            
            # Scale depth and width
            n = max(round(n * self.depth_multiple), 1)
            out_channels = self._make_divisible(args[0] * self.width_multiple, 8)
            
            # Create layer based on module type
            if m == 'YOLOv11Conv':
                layer = YOLOv11Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=args[1],
                    stride=args[2]
                )
            elif m == 'C3k2':
                layer = C3k2(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=n,
                    shortcut=args[1],
                    expansion=args[2]
                )
            elif m == 'C2PSA':
                layer = C2PSA(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=n
                )
            elif m == 'SPPF':
                layer = SPPF(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=args[1]
                )
            else:
                raise ValueError(f"Unknown module: {m}")
            
            layers.append(layer)
            channels.append(out_channels)
        
        return layers
    
    @staticmethod
    def _make_divisible(v: float, divisor: int = 8) -> int:
        """Make channels divisible by divisor"""
        return max(divisor, int(v + divisor / 2) // divisor * divisor)
    
    def _freeze_stages(self):
        """Freeze parameters in specified stages"""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                if i < len(self.layers):
                    m = self.layers[i]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through YOLOv11 backbone
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W)
            
        Returns:
            tuple: Multi-scale feature maps at specified indices
        """
        outs = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Set training mode and handle frozen stages"""
        super().train(mode)
        self._freeze_stages()
        
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


@MODELS.register_module()
class YOLOv11CSPBackbone(YOLOv11Backbone):
    """
    YOLOv11 CSP Backbone optimized for YOLO-World integration
    
    This variant focuses on Cross Stage Partial connections
    for better feature reuse and gradient flow.
    """
    
    def __init__(self, **kwargs):
        # Use more C3k2 blocks for better feature extraction
        super().__init__(**kwargs)
    
    def get_feature_channels(self) -> List[int]:
        """Get output channel dimensions for each output feature map"""
        channels = []
        temp_channels = [3]
        
        for i, (f, n, m, args) in enumerate(self.arch_settings[self.arch]):
            out_channels = self._make_divisible(args[0] * self.width_multiple, 8)
            temp_channels.append(out_channels)
            
            if i in self.out_indices:
                channels.append(out_channels)
        
        return channels


@MODELS.register_module()
class YOLOv11MultiScaleBackbone(YOLOv11Backbone):
    """
    YOLOv11 Multi-Scale Backbone for enhanced multi-modal fusion
    
    This version outputs additional intermediate features
    for better text-image alignment in YOLO-World.
    """
    
    def __init__(self, 
                 out_indices: Tuple[int] = (2, 4, 6, 8, 10),  # More outputs
                 **kwargs):
        super().__init__(out_indices=out_indices, **kwargs)
    
    def forward_with_intermediate(self, x: torch.Tensor) -> dict:
        """
        Forward pass with intermediate feature collection
        
        Returns:
            dict: Feature maps with their corresponding layer indices
        """
        features = {}
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            features[f'layer_{i}'] = x
            
            if i in self.out_indices:
                features[f'out_{i}'] = x
        
        return features
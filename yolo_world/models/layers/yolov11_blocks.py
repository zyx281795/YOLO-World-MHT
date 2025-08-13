# Copyright (c) YOLO-World Integration Project. All rights reserved.
"""
YOLOv11 Core Components for YOLO-World Integration

This module extracts and adapts key YOLOv11 components for integration
with YOLO-World's multi-modal architecture.

Key Components:
- C3k2: Improved version of C2f with smaller kernels
- C2PSA: Spatial attention mechanism
- Enhanced Conv and Bottleneck layers
"""

import torch
import torch.nn as nn
from typing import Union
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptConfigType


@MODELS.register_module()
class YOLOv11Conv(BaseModule):
    """Standard convolution with BatchNorm and activation from YOLOv11"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple] = 1,
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = None,
                 groups: int = 1,
                 dilation: int = 1,
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=norm_cfg['eps'], momentum=norm_cfg['momentum'])
        self.act = nn.SiLU(inplace=True) if act_cfg['type'] == 'SiLU' else nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


@MODELS.register_module()
class YOLOv11Bottleneck(BaseModule):
    """YOLOv11 Bottleneck with 3x3 convolutions"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool = True,
                 groups: int = 1,
                 kernel_size: Union[int, tuple] = (3, 3),
                 expansion: float = 0.5,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        hidden_channels = int(out_channels * expansion)
        self.cv1 = YOLOv11Conv(in_channels, hidden_channels, kernel_size[0], 1)
        self.cv2 = YOLOv11Conv(hidden_channels, out_channels, kernel_size[1], 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@MODELS.register_module()
class C3k(BaseModule):
    """C3k block - core component of C3k2"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 shortcut: bool = True,
                 groups: int = 1,
                 expansion: float = 0.5,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        hidden_channels = int(out_channels * expansion)
        self.cv1 = YOLOv11Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = YOLOv11Conv(in_channels, hidden_channels, 1, 1)
        self.cv3 = YOLOv11Conv(2 * hidden_channels, out_channels, 1, 1)
        
        self.m = nn.Sequential(*(
            YOLOv11Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 
                             kernel_size=(3, 3), expansion=1.0)
            for _ in range(num_blocks)
        ))
    
    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))


@MODELS.register_module()
class C3k2(BaseModule):
    """C3k2 block from YOLOv11 - replacement for C2f with improved efficiency"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 shortcut: bool = False,
                 groups: int = 1,
                 expansion: float = 0.5,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        hidden_channels = int(out_channels * expansion)
        self.cv1 = YOLOv11Conv(in_channels, 2 * hidden_channels, 1, 1)
        self.cv2 = YOLOv11Conv((2 + num_blocks) * hidden_channels, out_channels, 1, 1)
        
        self.m = nn.ModuleList([
            C3k(hidden_channels, hidden_channels, 2, True, groups) if i == 0 else
            YOLOv11Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 
                             kernel_size=(3, 3), expansion=1.0)
            for i in range(num_blocks)
        ])
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


@MODELS.register_module()
class PSABlock(BaseModule):
    """Position-Sensitive Attention Block for C2PSA"""
    
    def __init__(self,
                 channels: int,
                 expansion: float = 0.5,
                 num_heads: int = 4,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Multi-head attention
        self.attn = MultiHeadAttention(channels, num_heads)
        
        # Feed-forward network
        hidden_channels = int(channels * expansion * 2)
        self.ffn = nn.Sequential(
            YOLOv11Conv(channels, hidden_channels, 1, 1),
            YOLOv11Conv(hidden_channels, channels, 1, 1, act_cfg=dict(type='Identity'))
        )
    
    def forward(self, x):
        # Self-attention
        x_attn = self.attn(x)
        x = x + x_attn
        
        # Feed-forward with residual
        x = x + self.ffn(x)
        return x


@MODELS.register_module()
class MultiHeadAttention(BaseModule):
    """Multi-Head Attention for PSA"""
    
    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = YOLOv11Conv(channels, channels * 2, 1, 1, act_cfg=dict(type='Identity'))
        self.proj = YOLOv11Conv(channels, channels, 1, 1, act_cfg=dict(type='Identity'))
        
        # Position encoding
        self.pe = YOLOv11Conv(channels, channels, 3, 1, groups=channels, act_cfg=dict(type='Identity'))
    
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Generate QKV
        qkv = self.qkv(x)
        q, k = qkv.chunk(2, dim=1)
        v = x
        
        # Position encoding
        v = v + self.pe(v)
        
        # Reshape for attention
        q = q.flatten(2).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.flatten(2).reshape(B, self.num_heads, self.head_dim, N)
        v = v.flatten(2).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        # Attention computation
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        
        return self.proj(out)


@MODELS.register_module()
class C2PSA(BaseModule):
    """C2PSA block from YOLOv11 with Position-Sensitive Attention"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 expansion: float = 0.5,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        hidden_channels = int(out_channels * expansion)
        self.cv1 = YOLOv11Conv(in_channels, 2 * hidden_channels, 1, 1)
        self.cv2 = YOLOv11Conv(2 * hidden_channels, out_channels, 1, 1)
        
        self.m = nn.Sequential(*(
            PSABlock(hidden_channels, expansion=2.0)
            for _ in range(num_blocks)
        ))
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y[1:], 1))


@MODELS.register_module()
class SPPF(BaseModule):
    """Spatial Pyramid Pooling - Fast (SPPF) layer from YOLOv11"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        hidden_channels = in_channels // 2
        self.cv1 = YOLOv11Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = YOLOv11Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# Identity activation for layers that don't need activation
class Identity(nn.Module):
    def forward(self, x):
        return x
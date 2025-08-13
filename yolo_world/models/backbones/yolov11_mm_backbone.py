# Copyright (c) YOLO-World Integration Project. All rights reserved.
"""
YOLOv11 Multi-Modal Backbone for YOLO-World

This module integrates YOLOv11 backbone with YOLO-World's text processing
capabilities, creating a unified multi-modal architecture.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptConfigType, ConfigType

from .yolov11_backbone import YOLOv11Backbone
from .mm_backbone import HuggingCLIPLanguageBackbone, PseudoLanguageBackbone


@MODELS.register_module()
class YOLOv11MultiModalBackbone(BaseModule):
    """
    YOLOv11-based Multi-Modal Backbone for YOLO-World
    
    This backbone combines YOLOv11's improved visual feature extraction
    with YOLO-World's text processing capabilities.
    
    Args:
        image_model (ConfigType): YOLOv11 backbone configuration
        text_model (ConfigType): Text encoder configuration  
        frozen_stages (int): Number of frozen backbone stages
        with_text_model (bool): Whether to include text processing
        feature_fusion_cfg (dict): Configuration for feature fusion
    """
    
    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 feature_fusion_cfg: Optional[dict] = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        self.with_text_model = with_text_model
        self.frozen_stages = frozen_stages
        
        # Build YOLOv11 image backbone
        self.image_model = MODELS.build(image_model)
        
        # Build text model if required
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        
        # Feature fusion configuration
        self.feature_fusion_cfg = feature_fusion_cfg or {}
        self._build_fusion_layers()
        
        # Freeze stages
        self._freeze_stages()
    
    def _build_fusion_layers(self):
        """Build feature fusion layers for text-image alignment"""
        
        if not self.with_text_model or not self.feature_fusion_cfg:
            return
        
        fusion_type = self.feature_fusion_cfg.get('type', 'simple')
        
        if fusion_type == 'attention':
            self._build_attention_fusion()
        elif fusion_type == 'adaptive':
            self._build_adaptive_fusion()
        else:
            # Simple fusion - no additional layers needed
            pass
    
    def _build_attention_fusion(self):
        """Build attention-based fusion layers"""
        
        # Get image feature channels from YOLOv11 backbone
        if hasattr(self.image_model, 'get_feature_channels'):
            img_channels = self.image_model.get_feature_channels()
        else:
            # Default channels for YOLOv11l
            img_channels = [256, 512, 1024]
        
        text_channels = self.feature_fusion_cfg.get('text_channels', 512)
        
        self.fusion_layers = nn.ModuleList([
            AttentionFusion(img_ch, text_channels)
            for img_ch in img_channels
        ])
    
    def _build_adaptive_fusion(self):
        """Build adaptive fusion layers"""
        
        # Get image feature channels
        if hasattr(self.image_model, 'get_feature_channels'):
            img_channels = self.image_model.get_feature_channels()
        else:
            img_channels = [256, 512, 1024]
        
        text_channels = self.feature_fusion_cfg.get('text_channels', 512)
        
        self.fusion_layers = nn.ModuleList([
            AdaptiveFusion(img_ch, text_channels)
            for img_ch in img_channels
        ])
    
    def _freeze_stages(self):
        """Freeze specified stages in the image backbone"""
        if hasattr(self.image_model, '_freeze_stages'):
            self.image_model.frozen_stages = self.frozen_stages
            self.image_model._freeze_stages()
    
    def forward(self, 
                image: torch.Tensor, 
                text: Optional[List[List[str]]] = None) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through multi-modal backbone
        
        Args:
            image (torch.Tensor): Input images with shape (B, 3, H, W)
            text (List[List[str]], optional): Text prompts for each image
            
        Returns:
            tuple: (image_features, text_features)
                - image_features: Multi-scale image features 
                - text_features: Text embeddings or None
        """
        
        # Extract image features using YOLOv11
        img_feats = self.image_model(image)
        
        # Extract text features if available
        if text is not None and self.with_text_model:
            if hasattr(self.text_model, 'forward'):
                # For HuggingCLIPLanguageBackbone
                txt_feats = self.text_model(text)
            else:
                txt_feats = None
        else:
            txt_feats = None
        
        # Apply feature fusion if configured
        if hasattr(self, 'fusion_layers') and txt_feats is not None:
            img_feats = self._apply_fusion(img_feats, txt_feats)
        
        return img_feats, txt_feats
    
    def _apply_fusion(self, 
                     img_feats: Tuple[torch.Tensor], 
                     txt_feats: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply feature fusion between image and text features"""
        
        fused_feats = []
        
        for i, img_feat in enumerate(img_feats):
            if i < len(self.fusion_layers):
                fused_feat = self.fusion_layers[i](img_feat, txt_feats)
                fused_feats.append(fused_feat)
            else:
                fused_feats.append(img_feat)
        
        return tuple(fused_feats)
    
    def forward_text(self, text: List[List[str]]) -> torch.Tensor:
        """Forward text only - for pre-computing text embeddings"""
        
        assert self.with_text_model, "Text model not available"
        return self.text_model(text)
    
    def forward_image(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward image only - for inference with pre-computed text features"""
        
        return self.image_model(image)
    
    def train(self, mode: bool = True):
        """Set training mode"""
        super().train(mode)
        self._freeze_stages()


@MODELS.register_module()
class AttentionFusion(BaseModule):
    """Attention-based fusion for image and text features"""
    
    def __init__(self,
                 img_channels: int,
                 text_channels: int,
                 hidden_channels: Optional[int] = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        hidden_channels = hidden_channels or min(img_channels, text_channels)
        
        self.img_proj = nn.Linear(img_channels, hidden_channels)
        self.text_proj = nn.Linear(text_channels, hidden_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=8)
        self.output_proj = nn.Linear(hidden_channels, img_channels)
    
    def forward(self, 
                img_feat: torch.Tensor, 
                text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: (B, C, H, W)
            text_feat: (B, N, D) where N is number of text tokens
        """
        
        B, C, H, W = img_feat.shape
        
        # Reshape image features for attention
        img_flat = img_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        img_proj = self.img_proj(img_flat)  # (B, HW, hidden)
        
        # Project text features
        text_proj = self.text_proj(text_feat)  # (B, N, hidden)
        
        # Apply cross-attention (text attends to image)
        attended, _ = self.attention(
            text_proj.transpose(0, 1),   # (N, B, hidden)
            img_proj.transpose(0, 1),    # (HW, B, hidden) 
            img_proj.transpose(0, 1)     # (HW, B, hidden)
        )
        
        # Aggregate attended features
        attended = attended.transpose(0, 1).mean(1, keepdim=True)  # (B, 1, hidden)
        attended = attended.expand(-1, H*W, -1)  # (B, HW, hidden)
        
        # Project back and reshape
        output = self.output_proj(attended)  # (B, HW, C)
        output = output.transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection
        return img_feat + output


@MODELS.register_module()
class AdaptiveFusion(BaseModule):
    """Adaptive fusion that learns to combine image and text features"""
    
    def __init__(self,
                 img_channels: int,
                 text_channels: int,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        
        self.text_adapter = nn.Sequential(
            nn.Linear(text_channels, img_channels),
            nn.ReLU(inplace=True),
            nn.Linear(img_channels, img_channels)
        )
        
        self.fusion_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, 
                img_feat: torch.Tensor, 
                text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: (B, C, H, W)
            text_feat: (B, N, D)
        """
        
        # Adapt text features to image channel dimensions
        text_adapted = self.text_adapter(text_feat.mean(1))  # (B, C)
        text_adapted = text_adapted.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Adaptive combination
        fused = img_feat + self.fusion_weight * text_adapted
        
        return fused
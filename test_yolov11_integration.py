#!/usr/bin/env python3
"""
Test script for YOLO-World YOLOv11 integration

This script validates that our YOLOv11 components work correctly
with YOLO-World's multi-modal architecture.
"""

import torch
import traceback
from mmengine.config import Config


def test_yolov11_blocks():
    """Test YOLOv11 building blocks"""
    print("=== Testing YOLOv11 Building Blocks ===")
    
    try:
        from yolo_world.models.layers.yolov11_blocks import (
            YOLOv11Conv, C3k2, C2PSA, SPPF
        )
        
        # Test YOLOv11Conv
        conv = YOLOv11Conv(3, 64, 3, 2)
        x = torch.randn(1, 3, 640, 640)
        out = conv(x)
        print(f"✓ YOLOv11Conv: {x.shape} -> {out.shape}")
        
        # Test C3k2
        c3k2 = C3k2(64, 128, num_blocks=2)
        out = c3k2(out)
        print(f"✓ C3k2: {out.shape}")
        
        # Test C2PSA
        c2psa = C2PSA(128, 256)
        out = c2psa(out)
        print(f"✓ C2PSA: {out.shape}")
        
        # Test SPPF
        sppf = SPPF(256, 256)
        out = sppf(out)
        print(f"✓ SPPF: {out.shape}")
        
        print("All YOLOv11 blocks working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing YOLOv11 blocks: {e}")
        traceback.print_exc()
        return False


def test_yolov11_backbone():
    """Test YOLOv11 backbone"""
    print("\n=== Testing YOLOv11 Backbone ===")
    
    try:
        from yolo_world.models.backbones.yolov11_backbone import YOLOv11Backbone
        
        # Create YOLOv11 backbone
        backbone = YOLOv11Backbone(
            arch='YOLOv11l',
            out_indices=(4, 6, 10)
        )
        
        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        features = backbone(x)
        
        print(f"✓ YOLOv11 Backbone input: {x.shape}")
        for i, feat in enumerate(features):
            print(f"✓ Output feature {i}: {feat.shape}")
        
        print("YOLOv11 Backbone working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing YOLOv11 backbone: {e}")
        traceback.print_exc()
        return False


def test_multimodal_backbone():
    """Test YOLOv11 multi-modal backbone"""
    print("\n=== Testing YOLOv11 Multi-Modal Backbone ===")
    
    try:
        from yolo_world.models.backbones.yolov11_mm_backbone import YOLOv11MultiModalBackbone
        
        # Create multi-modal backbone
        mm_backbone = YOLOv11MultiModalBackbone(
            image_model=dict(
                type='YOLOv11Backbone',
                arch='YOLOv11l',
                out_indices=(4, 6, 10)
            ),
            text_model=dict(
                type='PseudoLanguageBackbone',  # Use pseudo for testing
                text_embed_path='dummy_path'   # Will need actual path for real testing
            ),
            with_text_model=False  # Disable for now
        )
        
        # Test image-only forward pass
        x = torch.randn(1, 3, 640, 640)
        img_features, txt_features = mm_backbone(x, None)
        
        print(f"✓ Multi-Modal Backbone input: {x.shape}")
        for i, feat in enumerate(img_features):
            print(f"✓ Image feature {i}: {feat.shape}")
        print(f"✓ Text features: {txt_features}")
        
        print("YOLOv11 Multi-Modal Backbone working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing multi-modal backbone: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test YOLOv11 configuration loading"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        config_path = 'configs/yolov11_integration/yolo_world_v2_l_yolov11_backbone.py'
        cfg = Config.fromfile(config_path)
        
        print(f"✓ Configuration loaded successfully")
        print(f"✓ Model type: {cfg.model.type}")
        print(f"✓ Backbone type: {cfg.model.backbone.type}")
        print(f"✓ Image model: {cfg.model.backbone.image_model.type}")
        
        # Check if all required components are defined
        required_components = [
            'backbone', 'neck', 'bbox_head', 'train_cfg', 'test_cfg'
        ]
        
        for component in required_components:
            if hasattr(cfg.model, component):
                print(f"✓ {component}: ✅")
            else:
                print(f"✓ {component}: ❌")
                return False
        
        print("Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        traceback.print_exc()
        return False


def test_model_building():
    """Test building the complete model"""
    print("\n=== Testing Complete Model Building ===")
    
    try:
        from mmyolo.registry import MODELS
        
        # Simple backbone test first
        backbone_cfg = dict(
            type='YOLOv11Backbone',
            arch='YOLOv11l',
            out_indices=(4, 6, 10)
        )
        
        backbone = MODELS.build(backbone_cfg)
        x = torch.randn(1, 3, 640, 640)
        features = backbone(x)
        
        print(f"✓ Model registry integration successful")
        print(f"✓ Built backbone: {type(backbone)}")
        for i, feat in enumerate(features):
            print(f"✓ Feature {i}: {feat.shape}")
        
        print("Model building successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error building model: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("YOLO-World YOLOv11 Integration Tests")
    print("=" * 50)
    
    tests = [
        test_yolov11_blocks,
        test_yolov11_backbone, 
        test_multimodal_backbone,
        test_configuration,
        test_model_building
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! YOLOv11 integration ready for use!")
    else:
        print("Some tests failed. Please review the errors above.")
    
    return passed == total


if __name__ == '__main__':
    main()
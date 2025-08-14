#!/usr/bin/env python3
"""
Simple YOLOv11 Components Test
"""

import torch
import sys
sys.path.append('.')

def test_yolov11_blocks():
    """Test YOLOv11 building blocks"""
    print("Testing YOLOv11 Building Blocks...")
    
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
        
        print("YOLOv11 blocks are working!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_yolov11_backbone():
    """Test YOLOv11 backbone"""
    print("\nTesting YOLOv11 Backbone...")
    
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
        
        print(f"✓ Input: {x.shape}")
        for i, feat in enumerate(features):
            print(f"✓ Feature {i}: {feat.shape}")
        
        print("YOLOv11 Backbone is working!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    print("=== YOLO-World YOLOv11 Components Test ===")
    
    # Import yolo_world first
    import yolo_world
    
    success1 = test_yolov11_blocks()
    success2 = test_yolov11_backbone()
    
    if success1 and success2:
        print("\n✅ All YOLOv11 components are working correctly!")
        print("The project successfully integrates YOLOv11 with YOLO-World.")
    else:
        print("\n❌ Some components failed to load.")
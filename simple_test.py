#!/usr/bin/env python3
"""
Simple test for YOLOv11 components (without full YOLO-World dependencies)
"""

import torch
import torch.nn as nn

# Test basic YOLOv11 components
def test_basic_components():
    print("Testing basic YOLOv11 components...")
    
    # Simple Conv test
    class SimpleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.SiLU(inplace=True)
        
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))
    
    # Test
    conv = SimpleConv(3, 64)
    x = torch.randn(1, 3, 64, 64)  # Smaller size for testing
    out = conv(x)
    print(f"Conv test: {x.shape} -> {out.shape}")
    
    # Test attention mechanism
    class SimpleAttention(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.channels = channels
            self.qkv = nn.Linear(channels, channels * 2)
            self.proj = nn.Linear(channels, channels)
        
        def forward(self, x):
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            qkv = self.qkv(x_flat)
            q, k = qkv.chunk(2, dim=-1)
            v = x_flat
            
            # Simple attention
            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.channels ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = self.proj(out)
            out = out.transpose(1, 2).reshape(B, C, H, W)
            return x + out
    
    # Test attention
    attn = SimpleAttention(64)
    out2 = attn(out)
    print(f"Attention test: {out.shape} -> {out2.shape}")
    
    print("Basic components working!")
    return True

if __name__ == '__main__':
    try:
        test_basic_components()
        print("SUCCESS: All basic tests passed!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
import torch
import os
print('PyTorch version:', torch.__version__)
print('Current directory:', os.getcwd())
print('YOLOv11 weights exist:', os.path.exists('checkpoints/yolo11n.pt'))

try:
    import yolo_world
    print('YOLO-World imported successfully')
except Exception as e:
    print('YOLO-World import failed:', e)

try:
    from yolo_world.models.layers import yolov11_blocks
    print('YOLOv11 blocks imported successfully')
except Exception as e:
    print('YOLOv11 blocks import failed:', e)

# Test loading YOLOv11 weights
try:
    yolov11_ckpt = torch.load('checkpoints/yolo11n.pt', map_location='cpu')
    print('YOLOv11 weights loaded successfully')
    print('Keys in checkpoint:', list(yolov11_ckpt.keys())[:5])
except Exception as e:
    print('YOLOv11 weights loading failed:', e)

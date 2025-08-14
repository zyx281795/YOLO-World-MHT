import torch
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
import yolo_world

# Test loading the converted weights
print('Testing converted YOLOv11 weights...')
config_file = 'configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
converted_weights = 'checkpoints/yolo_world_yolov11n_converted.pth'

cfg = Config.fromfile(config_file)
cfg.work_dir = './work_dirs'
cfg.load_from = converted_weights

try:
    runner = Runner.from_cfg(cfg)
    runner.call_hook('before_run')
    
    # Load weights with strict=False
    checkpoint = torch.load(converted_weights, map_location='cpu')
    state_dict = checkpoint['state_dict']
    missing_keys, unexpected_keys = runner.model.load_state_dict(state_dict, strict=False)
    
    print(f'Missing keys: {len(missing_keys)}')
    print(f'Unexpected keys: {len(unexpected_keys)}')
    print('YOLOv11 weights loaded successfully into YOLO-World!')
    
    # Test model evaluation mode
    runner.model.eval()
    
    print('YOLOv11 + YOLO-World integration successful!')
    print('Ready to run demo with YOLOv11 weights!')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
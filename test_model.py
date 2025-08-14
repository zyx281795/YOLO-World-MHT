#!/usr/bin/env python3
import sys
sys.path.append('.')

# Import all YOLO-World modules to register them
import yolo_world
from mmengine.config import Config
from mmengine.registry import MODELS
import torch

config_file = 'configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
checkpoint = 'checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth'

print("Loading config...")
cfg = Config.fromfile(config_file)
print('Config loaded successfully!')

print("Building model...")
model = MODELS.build(cfg.model)
print('Model built successfully!')

print("Loading checkpoint...")
ckpt = torch.load(checkpoint, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=False)
print('Checkpoint loaded successfully!')

print("All steps completed successfully!")
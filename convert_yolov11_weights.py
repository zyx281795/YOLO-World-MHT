#!/usr/bin/env python3
"""
YOLOv11 權重轉換器
將純YOLOv11權重轉換為YOLO-World多模態架構可用的權重
"""

import torch
import os
from collections import OrderedDict
from mmengine.config import Config
from mmengine.runner import Runner
import yolo_world

def convert_yolov11_to_yoloworld(yolov11_path, output_path, config_path):
    """
    轉換YOLOv11權重為YOLO-World格式
    
    Args:
        yolov11_path: YOLOv11權重文件路徑 (yolo11n.pt)
        output_path: 輸出的YOLO-World權重路徑
        config_path: YOLO-World配置文件路徑
    """
    print(f"🔄 正在轉換 {yolov11_path} 到 YOLO-World 格式...")
    
    # 載入YOLOv11權重
    print("📥 載入YOLOv11權重...")
    yolov11_ckpt = torch.load(yolov11_path, map_location='cpu')
    
    # YOLOv11權重通常在'model'鍵下
    if 'model' in yolov11_ckpt:
        yolov11_state_dict = yolov11_ckpt['model'].state_dict()
    else:
        yolov11_state_dict = yolov11_ckpt
    
    print(f"📊 YOLOv11權重包含 {len(yolov11_state_dict)} 個參數")
    
    # 載入YOLO-World配置以了解架構
    cfg = Config.fromfile(config_path)
    
    # 創建新的YOLO-World state_dict
    new_state_dict = OrderedDict()
    
    # 映射YOLOv11權重到YOLO-World架構
    backbone_mapping = {
        # YOLOv11 backbone -> YOLO-World image_model
        'model.': 'backbone.image_model.backbone.',
        'backbone.': 'backbone.image_model.backbone.',
    }
    
    # 處理backbone權重
    for old_key, value in yolov11_state_dict.items():
        new_key = old_key
        
        # 映射backbone權重
        for old_prefix, new_prefix in backbone_mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix)
                break
        
        # 特殊處理某些層
        if 'backbone' in new_key or 'neck' in new_key:
            new_state_dict[new_key] = value
            print(f"✓ 映射: {old_key} -> {new_key}")
    
    # 注意：text_model權重將從CLIP預訓練模型自動載入
    print("💬 Text model權重將使用CLIP預訓練權重")
    
    # 保存轉換後的權重
    checkpoint = {
        'state_dict': new_state_dict,
        'meta': {
            'converted_from': yolov11_path,
            'original_type': 'YOLOv11',
            'target_type': 'YOLO-World',
            'note': 'Converted backbone weights only, text_model uses CLIP pretrained weights'
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    
    print(f"✅ 轉換完成！")
    print(f"💾 輸出權重: {output_path}")
    print(f"📊 轉換了 {len(new_state_dict)} 個參數")
    
    return output_path

def test_converted_weights(config_path, checkpoint_path):
    """測試轉換後的權重是否可以載入"""
    print("\n🧪 測試轉換後的權重...")
    
    try:
        cfg = Config.fromfile(config_path)
        cfg.load_from = checkpoint_path
        
        # 嘗試建立模型
        runner = Runner.from_cfg(cfg)
        print("✅ 權重載入測試成功！")
        return True
        
    except Exception as e:
        print(f"❌ 權重載入測試失敗: {e}")
        return False

if __name__ == '__main__':
    # 設定路徑
    yolov11_weight = 'checkpoints/yolo11n.pt'
    output_weight = 'checkpoints/yolo_world_yolov11n_converted.pth'
    config_file = 'configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
    
    print("🚀 YOLOv11 -> YOLO-World 權重轉換器")
    print("=" * 50)
    
    # 檢查輸入文件
    if not os.path.exists(yolov11_weight):
        print(f"❌ YOLOv11權重文件不存在: {yolov11_weight}")
        exit(1)
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        exit(1)
    
    try:
        # 執行轉換
        converted_path = convert_yolov11_to_yoloworld(
            yolov11_weight, output_weight, config_file
        )
        
        # 測試載入
        success = test_converted_weights(config_file, converted_path)
        
        if success:
            print("\n🎉 轉換成功！您現在可以使用以下命令啟動demo:")
            print(f"python demo/gradio_demo.py {config_file} {converted_path}")
        else:
            print("\n⚠️ 轉換完成但測試失敗，可能需要調整配置")
            
    except Exception as e:
        print(f"❌ 轉換過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
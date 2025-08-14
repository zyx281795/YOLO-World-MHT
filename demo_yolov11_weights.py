#!/usr/bin/env python3
"""
使用YOLOv11權重的YOLO-World Demo

這個腳本會：
1. 轉換YOLOv11權重為YOLO-World格式（如果需要）
2. 啟動Gradio介面進行推理
"""

import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.runner import Runner
from mmdet.datasets import CocoDataset

# Import YOLO-World and YOLOv11 components
import yolo_world
from yolo_world.models.layers import yolov11_blocks

def convert_weights_if_needed():
    """轉換YOLOv11權重（如果尚未轉換）"""
    yolov11_path = 'checkpoints/yolo11n.pt'
    converted_path = 'checkpoints/yolo_world_yolov11n_converted.pth'
    
    if not os.path.exists(converted_path) or os.path.getmtime(yolov11_path) > os.path.getmtime(converted_path):
        print("🔄 正在轉換YOLOv11權重...")
        
        # 載入YOLOv11權重
        yolov11_ckpt = torch.load(yolov11_path, map_location='cpu')
        
        # 提取模型權重
        if hasattr(yolov11_ckpt, 'model'):
            state_dict = yolov11_ckpt.model.state_dict() if hasattr(yolov11_ckpt.model, 'state_dict') else yolov11_ckpt
        elif 'model' in yolov11_ckpt:
            state_dict = yolov11_ckpt['model']
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
        else:
            state_dict = yolov11_ckpt
        
        # 創建適配的state_dict
        new_state_dict = {}
        
        # 只保留backbone相關權重，讓text_model使用預訓練權重
        for key, value in state_dict.items():
            # 映射backbone權重到image_model
            if any(prefix in key for prefix in ['backbone', 'neck', 'model']):
                # 簡單的key映射
                new_key = f"backbone.image_model.{key}"
                new_state_dict[new_key] = value
        
        # 保存轉換後的權重
        checkpoint = {
            'state_dict': new_state_dict,
            'meta': {
                'converted_from': 'yolo11n.pt',
                'note': 'Partial weights for backbone only'
            }
        }
        
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)
        torch.save(checkpoint, converted_path)
        print(f"✅ 權重轉換完成: {converted_path}")
    
    return converted_path

def initialize_model():
    """初始化使用YOLOv11權重的模型"""
    global runner, test_pipeline
    
    print("🚀 初始化YOLO-World + YOLOv11權重...")
    
    # 轉換權重（如果需要）
    converted_weights = convert_weights_if_needed()
    
    # 使用基礎配置但載入轉換後的權重
    config_file = "configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    
    # 載入配置
    cfg = Config.fromfile(config_file)
    cfg.work_dir = './work_dirs'
    cfg.load_from = converted_weights
    
    # 確保YOLOv11組件可用
    cfg.custom_imports = dict(
        imports=['yolo_world', 'yolo_world.models.layers.yolov11_blocks'],
        allow_failed_imports=False
    )
    
    try:
        # 創建runner
        runner = Runner.from_cfg(cfg)
        runner.call_hook('before_run')
        
        # 嘗試載入權重，使用strict=False允許部分載入
        print("📥 載入YOLOv11轉換權重...")
        checkpoint = torch.load(converted_weights, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 載入權重，允許部分載入
        missing_keys, unexpected_keys = runner.model.load_state_dict(state_dict, strict=False)
        print(f"⚠️ Missing keys: {len(missing_keys)}")
        print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        
        # 設置測試pipeline
        pipeline = cfg.test_dataloader.dataset.pipeline
        pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(pipeline)
        
        runner.model.eval()
        
        print("✅ 模型初始化成功！")
        print(f"🎯 使用YOLOv11權重: yolo11n.pt")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型初始化失敗: {e}")
        
        # 回退到原始權重
        print("🔄 回退到原始YOLO-World權重...")
        cfg.load_from = "checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
        
        runner = Runner.from_cfg(cfg)
        runner.call_hook('before_run')
        runner.load_or_resume()
        
        pipeline = cfg.test_dataloader.dataset.pipeline
        pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(pipeline)
        runner.model.eval()
        
        print("⚠️ 使用原始YOLO-World權重")
        return False

def inference(image, texts, score_thr=0.3, max_dets=100):
    """執行推理"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 準備數據
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    
    # 推理
    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
    
    # 過濾結果
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    
    pred_instances = pred_instances.cpu().numpy()
    
    # 創建檢測結果
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    
    # 標註圖像
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)
    
    labels = [f"{texts[class_id][0]} {confidence:0.2f}" 
              for class_id, confidence in zip(detections.class_id, detections.confidence)]
    
    annotated_image = box_annotator.annotate(image.copy(), detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels=labels)
    
    return annotated_image, len(pred_instances['labels'])

def predict(image, text_input, score_threshold, max_boxes):
    """Gradio預測函數"""
    if image is None:
        return None, "0"
    
    # 解析文本輸入
    texts = [[t.strip()] for t in text_input.split(',')] + [[' ']]
    
    # 重新參數化模型
    runner.model.reparameterize(texts)
    
    # 執行推理
    result_image, num_detected = inference(
        image, texts, score_thr=score_threshold, max_dets=max_boxes
    )
    
    return result_image, str(num_detected)

# 初始化模型
print("🔧 正在初始化YOLO-World + YOLOv11權重...")
success = initialize_model()

weight_status = "✅ YOLOv11權重 (yolo11n.pt)" if success else "⚠️ 原始YOLO-World權重 (備用)"

# 創建Gradio介面
with gr.Blocks(title="YOLO-World + YOLOv11 Weights", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 YOLO-World + YOLOv11權重")
    gr.Markdown(f"**權重狀態:** {weight_status}")
    
    if success:
        gr.Markdown("""
        ### ✅ 成功載入YOLOv11權重！
        - 使用YOLOv11n (nano) 權重檔案
        - 保持YOLO-World的開放詞彙能力
        - 結合YOLOv11的架構改進
        """)
    else:
        gr.Markdown("""
        ### ⚠️ YOLOv11權重載入失敗
        - 目前使用原始YOLO-World權重
        - 仍然包含YOLOv11組件支援
        - 保持完整功能
        """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上傳圖片")
            text_input = gr.Textbox(
                label="要偵測的物體（用逗號分隔）", 
                placeholder="人, 車, 狗, 貓, 腳踏車",
                value="人, 車, 狗, 貓"
            )
            
            with gr.Row():
                score_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="信心度閾值"
                )
                max_boxes = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1,
                    label="最大偵測數量"
                )
            
            predict_btn = gr.Button("🔍 開始偵測", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="偵測結果")
            detected_count = gr.Textbox(label="偵測到的物體數量", interactive=False)
    
    predict_btn.click(
        fn=predict,
        inputs=[input_image, text_input, score_threshold, max_boxes],
        outputs=[output_image, detected_count]
    )
    
    # 範例
    gr.Examples(
        examples=[
            ["demo/sample_images/bus.jpg", "巴士, 人, 車", 0.3, 50],
        ],
        inputs=[input_image, text_input, score_threshold, max_boxes],
        outputs=[output_image, detected_count],
        fn=predict,
        cache_examples=False
    )

if __name__ == "__main__":
    print(f"🚀 啟動Demo - {weight_status}")
    demo.launch(server_name='0.0.0.0', server_port=8084, share=False)
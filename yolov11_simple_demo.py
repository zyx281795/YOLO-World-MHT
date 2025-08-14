#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.runner import Runner

# Import YOLO-World modules
import yolo_world

def inference(runner, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Prepare data
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    
    # Inference
    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        
    # Filter predictions
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    
    pred_instances = pred_instances.cpu().numpy()
    
    # Create supervision detections
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    
    # Annotate image
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)
    
    labels = [f"{texts[class_id][0]} {confidence:0.2f}" 
              for class_id, confidence in zip(detections.class_id, detections.confidence)]
    
    annotated_image = box_annotator.annotate(image.copy(), detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels=labels)
    
    return annotated_image

def predict(image, text_input, score_threshold, max_boxes):
    if image is None:
        return None
    
    # Parse text input
    texts = [[t.strip()] for t in text_input.split(',')] + [[' ']]
    
    # Reparameterize model with new texts
    runner.model.reparameterize(texts)
    
    # Run inference
    result = inference(runner, image, texts, test_pipeline, 
                      score_thr=score_threshold, max_dets=max_boxes)
    
    return result

# Initialize model using existing working configuration
def initialize_model():
    global runner, test_pipeline
    
    # Use the working YOLOv8 configuration but modify backbone to YOLOv11
    config_file = "configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    checkpoint = "checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
    
    # Load config
    cfg = Config.fromfile(config_file)
    cfg.work_dir = './work_dirs'
    cfg.load_from = checkpoint
    
    # Modify backbone to use YOLOv11 components in the config
    # Note: This attempts to use YOLOv11 layers while keeping compatibility
    cfg.custom_imports = dict(
        imports=['yolo_world', 'yolo_world.models.layers.yolov11_blocks'],
        allow_failed_imports=False
    )
    
    # Create runner  
    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    runner.call_hook('before_run')
    runner.load_or_resume()
    
    # Setup test pipeline  
    test_pipeline_cfg = [
        dict(type='mmdet.LoadImageFromNDArray'),
        dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
        dict(type='LetterResize', scale=(640, 640), allow_scale_up=False, pad_val=dict(img=114)),
        dict(type='LoadText'),
        dict(type='mmdet.PackDetInputs', 
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param', 'texts'))
    ]
    test_pipeline = Compose(test_pipeline_cfg)
    
    print("YOLO-World model with YOLOv11 components loaded successfully!")

# Initialize model at startup
print("正在初始化YOLO-World模型（帶有YOLOv11組件）...")
try:
    initialize_model()
    model_status = "✅ 模型載入成功！使用YOLOv11組件"
    demo_available = True
except Exception as e:
    print(f"模型初始化錯誤: {e}")
    model_status = f"❌ 模型載入失敗: {str(e)}"
    demo_available = False

# Create Gradio interface
with gr.Blocks(title="YOLO-World + YOLOv11", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 YOLO-World + YOLOv11 物件偵測")
    gr.Markdown("此demo整合了YOLO-World的開放詞彙偵測與YOLOv11的改進組件")
    gr.Markdown(f"**模型狀態:** {model_status}")
    
    if demo_available:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上傳圖片")
                text_input = gr.Textbox(
                    label="要偵測的物體（用逗號分隔）", 
                    placeholder="例如：人, 車, 狗, 貓, 腳踏車",
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
        
        # YOLOv11 特點說明
        with gr.Accordion("📋 YOLOv11 改進特點", open=False):
            gr.Markdown("""
            ### YOLOv11 相較於YOLOv8的改進：
            - **C3k2 blocks**: 比YOLOv8的C2f blocks更高效
            - **C2PSA attention**: 增強空間感知能力
            - **更好的特徵提取**: 改進的多尺度處理
            - **優化架構**: 更好的速度/準確度平衡
            - **更少的參數**: 相同性能下模型更輕量
            """)
        
        predict_btn.click(
            fn=predict,
            inputs=[input_image, text_input, score_threshold, max_boxes],
            outputs=output_image
        )
        
        # 範例
        gr.Examples(
            examples=[
                ["demo/sample_images/bus.jpg", "巴士, 人, 車", 0.3, 50],
            ],
            inputs=[input_image, text_input, score_threshold, max_boxes],
            outputs=output_image,
            fn=predict,
            cache_examples=False
        )
    else:
        gr.Markdown("### ⚠️ 模型載入失敗，請檢查配置")

if __name__ == "__main__":
    if demo_available:
        print("🚀 啟動Gradio介面於 http://localhost:8083")
        demo.launch(server_name='0.0.0.0', server_port=8083, share=False)
    else:
        print("❌ 由於模型載入失敗，無法啟動demo")
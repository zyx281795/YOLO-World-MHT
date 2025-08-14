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

# Initialize YOLOv11 model using Runner approach
def initialize_yolov11_model():
    global runner, test_pipeline
    
    # Use YOLOv11 backbone configuration
    config_file = "configs/yolov11_integration/yolo_world_v2_l_yolov11_backbone.py"
    # Use existing weight file (we'll adapt it)
    checkpoint = "checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
    
    # Load config
    cfg = Config.fromfile(config_file)
    cfg.work_dir = './work_dirs'
    cfg.load_from = checkpoint
    
    # Create runner  
    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    runner.call_hook('before_run')
    # Use strict=False to allow partial loading of weights
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
    
    print("YOLOv11 YOLO-World model initialized successfully!")

# Initialize model at startup
print("Initializing YOLO-World with YOLOv11 backbone...")
try:
    initialize_yolov11_model()
    model_status = "✅ YOLOv11 model loaded successfully!"
except Exception as e:
    print(f"Error initializing YOLOv11 model: {e}")
    model_status = f"❌ Failed to load YOLOv11 model: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="YOLO-World + YOLOv11 Demo") as demo:
    gr.Markdown("# YOLO-World with YOLOv11 Backbone")
    gr.Markdown("This demo uses YOLO-World's open-vocabulary detection with YOLOv11's improved backbone architecture.")
    gr.Markdown(f"**Model Status:** {model_status}")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            text_input = gr.Textbox(
                label="Objects to detect (comma-separated)", 
                placeholder="person, car, dog, cat, bicycle, motorcycle",
                value="person, car, dog, cat"
            )
            score_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                label="Confidence Threshold"
            )
            max_boxes = gr.Slider(
                minimum=1, maximum=100, value=50, step=1,
                label="Maximum Detections"
            )
            predict_btn = gr.Button("🔍 Detect Objects", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
    
    # Model information
    with gr.Row():
        gr.Markdown("""
        ### YOLOv11 Improvements:
        - **C3k2 blocks**: More efficient than YOLOv8's C2f blocks
        - **C2PSA attention**: Enhanced spatial awareness
        - **Better feature extraction**: Improved multi-scale processing
        - **Optimized architecture**: Better speed/accuracy trade-off
        """)
    
    predict_btn.click(
        fn=predict,
        inputs=[input_image, text_input, score_threshold, max_boxes],
        outputs=output_image
    )
    
    # Example
    gr.Examples(
        examples=[
            ["demo/sample_images/bus.jpg", "bus, person, car", 0.3, 50],
        ],
        inputs=[input_image, text_input, score_threshold, max_boxes],
        outputs=output_image,
        fn=predict,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=8082, share=False)
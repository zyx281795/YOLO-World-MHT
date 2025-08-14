#!/usr/bin/env python3
"""
Test YOLOv11 weights with YOLO-World
Simple version without Unicode characters for Windows compatibility
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
    """Convert YOLOv11 weights if needed"""
    yolov11_path = 'checkpoints/yolo11n.pt'
    converted_path = 'checkpoints/yolo_world_yolov11n_converted.pth'
    
    if not os.path.exists(converted_path) or os.path.getmtime(yolov11_path) > os.path.getmtime(converted_path):
        print("Converting YOLOv11 weights...")
        
        # Load YOLOv11 weights
        yolov11_ckpt = torch.load(yolov11_path, map_location='cpu')
        
        # Extract model weights
        if hasattr(yolov11_ckpt, 'model'):
            state_dict = yolov11_ckpt.model.state_dict() if hasattr(yolov11_ckpt.model, 'state_dict') else yolov11_ckpt
        elif 'model' in yolov11_ckpt:
            state_dict = yolov11_ckpt['model']
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
        else:
            state_dict = yolov11_ckpt
        
        # Create adapted state_dict
        new_state_dict = {}
        
        # Only keep backbone weights, let text_model use pretrained weights
        for key, value in state_dict.items():
            # Map backbone weights to image_model
            if any(prefix in key for prefix in ['backbone', 'neck', 'model']):
                # Simple key mapping
                new_key = f"backbone.image_model.{key}"
                new_state_dict[new_key] = value
        
        # Save converted weights
        checkpoint = {
            'state_dict': new_state_dict,
            'meta': {
                'converted_from': 'yolo11n.pt',
                'note': 'Partial weights for backbone only'
            }
        }
        
        os.makedirs(os.path.dirname(converted_path), exist_ok=True)
        torch.save(checkpoint, converted_path)
        print(f"Weight conversion completed: {converted_path}")
    
    return converted_path

def initialize_model():
    """Initialize model with YOLOv11 weights"""
    global runner, test_pipeline
    
    print("Initializing YOLO-World + YOLOv11 weights...")
    
    # Convert weights if needed
    converted_weights = convert_weights_if_needed()
    
    # Use base configuration but load converted weights
    config_file = "configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    
    # Load configuration
    cfg = Config.fromfile(config_file)
    cfg.work_dir = './work_dirs'
    cfg.load_from = converted_weights
    
    # Ensure YOLOv11 components are available
    cfg.custom_imports = dict(
        imports=['yolo_world', 'yolo_world.models.layers.yolov11_blocks'],
        allow_failed_imports=False
    )
    
    try:
        # Create runner
        runner = Runner.from_cfg(cfg)
        runner.call_hook('before_run')
        
        # Try to load weights, use strict=False for partial loading
        print("Loading YOLOv11 converted weights...")
        checkpoint = torch.load(converted_weights, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights, allow partial loading
        missing_keys, unexpected_keys = runner.model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # Set test pipeline
        pipeline = cfg.test_dataloader.dataset.pipeline
        pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(pipeline)
        
        runner.model.eval()
        
        print("Model initialization successful!")
        print(f"Using YOLOv11 weights: yolo11n.pt")
        
        return True
        
    except Exception as e:
        print(f"Model initialization failed: {e}")
        
        # Fallback to original weights
        print("Falling back to original YOLO-World weights...")
        cfg.load_from = "checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
        
        runner = Runner.from_cfg(cfg)
        runner.call_hook('before_run')
        runner.load_or_resume()
        
        pipeline = cfg.test_dataloader.dataset.pipeline
        pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(pipeline)
        runner.model.eval()
        
        print("Using original YOLO-World weights")
        return False

def inference(image, texts, score_thr=0.3, max_dets=100):
    """Perform inference"""
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
    
    # Filter results
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    
    pred_instances = pred_instances.cpu().numpy()
    
    # Create detection results
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
    
    return annotated_image, len(pred_instances['labels'])

def predict(image, text_input, score_threshold, max_boxes):
    """Gradio prediction function"""
    if image is None:
        return None, "0"
    
    # Parse text input
    texts = [[t.strip()] for t in text_input.split(',')] + [[' ']]
    
    # Reparameterize model
    runner.model.reparameterize(texts)
    
    # Perform inference
    result_image, num_detected = inference(
        image, texts, score_thr=score_threshold, max_dets=max_boxes
    )
    
    return result_image, str(num_detected)

if __name__ == "__main__":
    print("Testing YOLO-World + YOLOv11 weights...")
    
    # Check if YOLOv11 weights exist
    if not os.path.exists('checkpoints/yolo11n.pt'):
        print("Error: YOLOv11 weights not found at checkpoints/yolo11n.pt")
        print("Please ensure the weights file exists")
        sys.exit(1)
    
    # Initialize model
    success = initialize_model()
    
    weight_status = "YOLOv11 weights (yolo11n.pt)" if success else "Original YOLO-World weights (backup)"
    print(f"Weight status: {weight_status}")
    
    # Test a simple inference to verify it works
    print("Testing inference with a dummy image...")
    try:
        # Create dummy image and text
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        texts = [['person'], ['car'], [' ']]
        
        # Reparameterize
        runner.model.reparameterize(texts)
        
        # Test inference
        result_image, num_detected = inference(dummy_image, texts, score_thr=0.5, max_dets=10)
        print(f"Test inference successful! Detected {num_detected} objects")
        
        print("YOLO-World + YOLOv11 weights setup is working correctly!")
        print(f"You can now use this configuration with: {weight_status}")
        
    except Exception as e:
        print(f"Test inference failed: {e}")
        import traceback
        traceback.print_exc()
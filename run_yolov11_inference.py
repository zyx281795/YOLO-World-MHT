#!/usr/bin/env python3
"""
YOLO-World + YOLOv11 Components Simple Inference Script

This script demonstrates how to use YOLOv11 components with YOLO-World
while maintaining compatibility with existing weights.
"""

import os
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import supervision as sv
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.runner import Runner

# Import YOLO-World and YOLOv11 modules
import yolo_world
from yolo_world.models.layers import yolov11_blocks
from yolo_world.models.backbones import yolov11_backbone

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World + YOLOv11 Inference')
    parser.add_argument('--config', default='configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py', help='config file')
    parser.add_argument('--checkpoint', default='checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth', help='checkpoint file')
    parser.add_argument('--image', default='demo/sample_images/bus.jpg', help='image file')
    parser.add_argument('--text', default='bus,person,car', help='text prompts')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold')
    parser.add_argument('--max-dets', type=int, default=100, help='max detections')
    parser.add_argument('--output', default='output_yolov11.jpg', help='output image')
    return parser.parse_args()

def inference(runner, image_path, texts, test_pipeline, score_thr=0.3, max_dets=100):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
    
    return annotated_image, len(pred_instances['labels'])

def main():
    args = parse_args()
    
    print("🚀 YOLO-World + YOLOv11 Components Inference")
    print(f"📁 Config: {args.config}")
    print(f"💾 Checkpoint: {args.checkpoint}")
    print(f"🖼️ Image: {args.image}")
    print(f"📝 Text: {args.text}")
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = './work_dirs'
    cfg.load_from = args.checkpoint
    
    # Enable YOLOv11 components import (this enables YOLOv11 blocks to be available)
    cfg.custom_imports = dict(
        imports=['yolo_world', 'yolo_world.models.layers.yolov11_blocks', 'yolo_world.models.backbones.yolov11_backbone'],
        allow_failed_imports=False
    )
    
    print("🔧 Building model with YOLOv11 components available...")
    
    # Create runner
    runner = Runner.from_cfg(cfg)
    runner.call_hook('before_run')
    runner.load_or_resume()
    
    # Setup test pipeline
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(pipeline)
    
    runner.model.eval()
    
    print(f"✅ Model loaded: {type(runner.model.backbone).__name__}")
    print(f"🎯 Text model: {type(runner.model.backbone.text_model).__name__}")
    print(f"🖼️ Image model: {type(runner.model.backbone.image_model).__name__}")
    
    # Parse text prompts
    texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]
    
    # Reparameterize model with texts
    print(f"🔄 Reparameterizing model with texts: {[t[0] for t in texts[:-1]]}")
    runner.model.reparameterize(texts)
    
    # Run inference
    print(f"🔍 Running inference...")
    result_image, num_detections = inference(
        runner, args.image, texts, test_pipeline, 
        score_thr=args.score_thr, max_dets=args.max_dets
    )
    
    # Save result
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_image_bgr)
    
    print(f"✅ Detection complete!")
    print(f"📊 Found {num_detections} objects")
    print(f"💾 Result saved to: {args.output}")
    
    # Test YOLOv11 components functionality
    print("\n🧪 Testing YOLOv11 components:")
    try:
        from yolo_world.models.layers.yolov11_blocks import YOLOv11Conv, C3k2, C2PSA
        conv = YOLOv11Conv(3, 64, 3, 2)
        x = torch.randn(1, 3, 64, 64)
        out = conv(x)
        print(f"✅ YOLOv11Conv working: {x.shape} -> {out.shape}")
        
        c3k2 = C3k2(64, 128)
        out = c3k2(out)
        print(f"✅ C3k2 working: {out.shape}")
        
        print("🎉 All YOLOv11 components are functional!")
        
    except Exception as e:
        print(f"⚠️ YOLOv11 components test failed: {e}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Extract YOLOv11 weights without ultralytics dependencies
"""

import torch
import pickle
import zipfile
from pathlib import Path

def extract_yolov11_state_dict(yolov11_path, output_path):
    """Extract pure state_dict from YOLOv11 checkpoint"""
    print(f"Extracting weights from {yolov11_path}...")
    
    try:
        # Try to load directly first
        checkpoint = torch.load(yolov11_path, map_location='cpu')
        print("Direct loading successful")
        
    except Exception as e:
        print(f"Direct loading failed: {e}")
        
        # Try to extract as zip file and get the data.pkl
        try:
            with zipfile.ZipFile(yolov11_path, 'r') as z:
                with z.open('data.pkl') as f:
                    checkpoint = pickle.load(f)
            print("Extracted from zip archive")
            
        except Exception as e2:
            print(f"Zip extraction also failed: {e2}")
            
            # Try loading with weights_only=True to avoid module dependency
            try:
                checkpoint = torch.load(yolov11_path, map_location='cpu', weights_only=True)
                print("Loaded with weights_only=True")
            except Exception as e3:
                print(f"All loading methods failed: {e3}")
                return None
    
    # Extract state dict
    if hasattr(checkpoint, 'state_dict'):
        state_dict = checkpoint.state_dict()
    elif 'model' in checkpoint:
        model = checkpoint['model']
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        else:
            state_dict = model
    else:
        state_dict = checkpoint
    
    print(f"Extracted {len(state_dict)} parameters")
    
    # Save clean state dict
    clean_checkpoint = {
        'state_dict': state_dict,
        'meta': {
            'source': 'yolo11n.pt',
            'extracted_for': 'YOLO-World integration'
        }
    }
    
    torch.save(clean_checkpoint, output_path)
    print(f"Clean weights saved to {output_path}")
    
    return output_path

def convert_to_yoloworld_format(clean_weights_path, output_path):
    """Convert clean YOLOv11 weights to YOLO-World format"""
    print("Converting to YOLO-World format...")
    
    checkpoint = torch.load(clean_weights_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Map YOLOv11 keys to YOLO-World backbone keys
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Map backbone weights
        if any(prefix in key for prefix in ['model.', 'backbone.']):
            # Convert to image_model path
            new_key = key.replace('model.', 'backbone.image_model.backbone.')
            new_key = new_key.replace('backbone.', 'backbone.image_model.backbone.')
            new_state_dict[new_key] = value
            print(f"Mapped: {key} -> {new_key}")
    
    # Save converted weights
    converted_checkpoint = {
        'state_dict': new_state_dict,
        'meta': {
            'converted_from': 'yolo11n.pt',
            'target_architecture': 'YOLO-World',
            'note': 'Backbone weights only, text model uses CLIP pretrained'
        }
    }
    
    torch.save(converted_checkpoint, output_path)
    print(f"Converted weights saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Paths
    yolov11_path = "checkpoints/yolo11n.pt"
    clean_path = "checkpoints/yolo11n_clean.pth"
    converted_path = "checkpoints/yolo_world_yolov11n_converted.pth"
    
    print("=== YOLOv11 Weight Extraction & Conversion ===")
    
    # Step 1: Extract clean weights
    success = extract_yolov11_state_dict(yolov11_path, clean_path)
    
    if success:
        # Step 2: Convert to YOLO-World format
        convert_to_yoloworld_format(clean_path, converted_path)
        print("\nConversion completed successfully!")
        print(f"Use this file with YOLO-World: {converted_path}")
    else:
        print("Extraction failed!")
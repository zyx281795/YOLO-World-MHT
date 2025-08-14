#!/usr/bin/env python3
"""
Launch YOLO-World demo with converted YOLOv11 weights
"""

import os
import sys

def main():
    print("=== YOLO-World + YOLOv11 Weights Demo ===")
    print("Starting demo with converted YOLOv11 weights...")
    
    # Check if converted weights exist
    converted_weights = "checkpoints/yolo_world_yolov11n_converted.pth"
    if not os.path.exists(converted_weights):
        print(f"Error: Converted weights not found at {converted_weights}")
        print("Please run extract_yolov11_weights.py first")
        return
    
    # Configuration file
    config_file = "configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    
    # Run the original gradio demo with converted weights
    print(f"Config: {config_file}")
    print(f"Weights: {converted_weights}")
    print("Launching Gradio demo...")
    
    # Set up sys.argv for the demo script
    sys.argv = [
        'demo/gradio_demo.py',
        config_file,
        converted_weights
    ]
    
    # Import and run the demo
    with open('demo/gradio_demo.py') as f:
        demo_code = f.read()
    
    # Execute the demo code
    exec(demo_code)

if __name__ == "__main__":
    main()
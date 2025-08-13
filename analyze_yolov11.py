#!/usr/bin/env python3
"""
YOLOv11 Architecture Analysis Script
This script analyzes YOLOv11 structure for YOLO-World integration
"""

try:
    from ultralytics import YOLO
    import torch.nn as nn
    
    print("=== YOLOv11 Architecture Analysis ===")
    
    # Load YOLOv11 model 
    model = YOLO('yolo11n.yaml')
    
    print(f"Model type: {type(model.model)}")
    print(f"Model device: {model.device}")
    
    print("\n=== YOLOv11 Architecture Layers ===")
    
    # Get model architecture
    for i, (name, module) in enumerate(model.model.named_modules()):
        if len(list(module.children())) == 0 and name:  # leaf modules only
            print(f"Layer {i}: {name} -> {module.__class__.__name__}")
            
            # Look for key YOLOv11 components
            if 'C3k2' in str(module.__class__.__name__) or 'C2PSA' in str(module.__class__.__name__):
                print(f"  *** Found YOLOv11 specific component: {module.__class__.__name__} ***")
                print(f"  Module details: {module}")
        
        if i > 50:  # Limit output for readability
            print("  ... (truncated for readability)")
            break
    
    print("\n=== YOLOv11 Model Structure ===")
    try:
        print(model.model)
    except:
        print("Could not print full model structure")
    
    print("\n=== Important Components for Integration ===")
    
    # Look for specific YOLOv11 improvements
    key_components = []
    for name, module in model.model.named_modules():
        class_name = module.__class__.__name__
        if any(keyword in class_name for keyword in ['C3k2', 'C2PSA', 'SPFF']):
            key_components.append((name, class_name))
    
    if key_components:
        print("Found YOLOv11 specific components:")
        for name, class_name in key_components:
            print(f"  - {name}: {class_name}")
    else:
        print("No YOLOv11 specific components found in layer names")
        print("Note: Components might be embedded within standard Conv/Sequential layers")
    
    print("\n=== Analysis Complete ===")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure ultralytics is installed: pip install ultralytics")
except Exception as e:
    print(f"Error analyzing YOLOv11: {e}")
    print("This might be due to missing model files or configuration issues")
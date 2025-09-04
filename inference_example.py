#!/usr/bin/env python3
"""
Simple example of using MobileOne reparameterization for inference
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def inference_example():
    """Simple inference example with reparameterization."""
    
    try:
        from models.yolo import DetectionModel
        import yaml
        
        print("=== MobileOne Inference Example ===")
        
        # 1. Load configuration
        with open("models/yolo-mobileone-500k.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        
        # 2. Create model
        model = DetectionModel(cfg, ch=3, nc=cfg['nc'], anchors=cfg['anchors'])
        model.eval()
        
        print(f"✓ Model created with {cfg['nc']} classes")
        
        # 3. Reparameterize for inference (IMPORTANT!)
        print("Reparameterizing for inference...")
        model.reparameterize_mobileone()
        
        # 4. Prepare input
        input_tensor = torch.randn(1, 3, 320, 320)
        print(f"✓ Input shape: {input_tensor.shape}")
        
        # 5. Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print(f"✓ Inference completed!")
        print(f"✓ Output type: {type(outputs)}")
        
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            detections = outputs[0]
            if isinstance(detections, torch.Tensor):
                print(f"✓ Detections shape: {detections.shape}")
                print(f"✓ Format: [batch, detections, 8] (x, y, w, h, conf, head, front, back)")
        
        print("\n✅ Inference example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    inference_example()

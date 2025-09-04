#!/usr/bin/env python3
"""
Quick test for three-class model
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    try:
        from models.yolo import DetectionModel
        import yaml
        
        print("Quick test for three-class model...")
        
        # Load config
        with open("models/yolo-mobileone-500k.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        
        print(f"Classes: {cfg['nc']}")
        
        # Create model
        model = DetectionModel(cfg, ch=3, nc=cfg['nc'], anchors=cfg['anchors'])
        
        # Test forward pass
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ Model works!")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if isinstance(output, (list, tuple)) and len(output) > 0:
            det = output[0]
            if isinstance(det, torch.Tensor):
                print(f"Detection shape: {det.shape}")
                print(f"Expected: [1, 25200, 8] for 3 classes")
                if det.shape[-1] == 8:
                    print("✓ Correct output format!")
                else:
                    print(f"⚠ Wrong channels: {det.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_test()

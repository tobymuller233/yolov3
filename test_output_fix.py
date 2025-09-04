#!/usr/bin/env python3
"""
Test script to verify YOLO output handling fix
"""

import torch
import yaml
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_yolo_output():
    """Test YOLO model output handling."""
    
    try:
        from models.yolo import Model
        
        # Load configuration
        config_path = "models/yolo-mobileone-500k.yaml"
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print("Creating model...")
        model = Model(cfg)
        print("✓ Model created successfully!")
        
        # Test forward pass
        print("Testing forward pass...")
        dummy_input = torch.randn(1, 3, 320, 320)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful!")
            print(f"✓ Input shape: {dummy_input.shape}")
            
            # Handle YOLO model output format
            print(f"\nOutput type: {type(output)}")
            
            if isinstance(output, (list, tuple)):
                print(f"Output is a {type(output).__name__} with {len(output)} elements")
                
                for i, out in enumerate(output):
                    print(f"\nOutput {i}:")
                    if isinstance(out, torch.Tensor):
                        print(f"  Type: torch.Tensor")
                        print(f"  Shape: {out.shape}")
                        print(f"  Dtype: {out.dtype}")
                        
                        # Analyze detection output format
                        if len(out.shape) == 3:  # [batch, num_detections, features]
                            batch_size, num_detections, features = out.shape
                            print(f"  Batch size: {batch_size}")
                            print(f"  Number of detections: {num_detections}")
                            print(f"  Features per detection: {features}")
                            
                            if features == 6:  # Typical YOLO format: [x, y, w, h, conf, class]
                                print(f"  Format: [x, y, w, h, confidence, class]")
                            elif features == 5:  # Alternative format: [x, y, w, h, conf]
                                print(f"  Format: [x, y, w, h, confidence]")
                            else:
                                print(f"  Custom format with {features} features")
                                
                    elif isinstance(out, (list, tuple)):
                        print(f"  Type: {type(out).__name__} with {len(out)} sub-elements")
                        for j, sub_out in enumerate(out):
                            if isinstance(sub_out, torch.Tensor):
                                print(f"    Sub-output {j}: {sub_out.shape}")
                            else:
                                print(f"    Sub-output {j}: {type(sub_out)}")
                    else:
                        print(f"  Type: {type(out)}")
                        if hasattr(out, '__len__'):
                            print(f"  Length: {len(out)}")
            else:
                print(f"Output type: {type(output)}")
                if hasattr(output, 'shape'):
                    print(f"Output shape: {output.shape}")
        
        print("\n🎉 YOLO output handling test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing YOLO output handling fix...")
    success = test_yolo_output()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")

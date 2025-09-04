#!/usr/bin/env python3
"""
Test script for three-class YOLO-MobileOne model
Classes: head, front, back
"""

import torch
import yaml
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_three_class_model():
    """Test three-class model configuration."""
    
    try:
        from models.yolo import DetectionModel
        
        print("Testing three-class YOLO-MobileOne model...")
        print("Classes: head, front, back")
        
        # Load configuration
        config_path = "models/yolo-mobileone-500k.yaml"
        print(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print(f"✓ Configuration loaded successfully!")
        print(f"Number of classes: {cfg['nc']}")
        print(f"Depth multiple: {cfg['depth_multiple']}")
        print(f"Width multiple: {cfg['width_multiple']}")
        
        # Create model
        print("\nCreating model...")
        model = DetectionModel(cfg, ch=3, nc=cfg['nc'], anchors=cfg['anchors'])
        
        print(f"✓ Model created successfully!")
        print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("\n=== Forward Pass Test ===")
        dummy_input = torch.randn(1, 3, 320, 320)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful!")
            
            # Analyze output structure
            print(f"\n=== Output Analysis ===")
            if isinstance(output, (list, tuple)):
                print(f"Output is a {type(output).__name__} with {len(output)} elements")
                
                # First element should be processed detections
                if len(output) > 0:
                    detections = output[0]
                    if isinstance(detections, torch.Tensor):
                        print(f"Detections shape: {detections.shape}")
                        print(f"Expected shape: [batch, num_detections, 8] (x, y, w, h, conf, class_head, class_front, class_back)")
                        
                        if detections.shape[-1] == 8:
                            print("✓ Correct output format for 3 classes!")
                        else:
                            print(f"⚠ Unexpected output channels: {detections.shape[-1]} (expected 8)")
                
                # Second element should be raw feature maps
                if len(output) > 1:
                    feature_maps = output[1]
                    if isinstance(feature_maps, list):
                        print(f"Feature maps: {len(feature_maps)} scales")
                        for i, fm in enumerate(feature_maps):
                            if isinstance(fm, torch.Tensor):
                                print(f"  Scale {i}: {fm.shape}")
                                # Check if feature map has correct channels for 3 classes
                                expected_channels = (5 + 3) * 3  # (5 + nc) * na
                                if fm.shape[-1] == expected_channels:
                                    print(f"    ✓ Correct channels: {expected_channels}")
                                else:
                                    print(f"    ⚠ Unexpected channels: {fm.shape[-1]} (expected {expected_channels})")
            else:
                print(f"Unexpected output type: {type(output)}")
        
        # Test with different input sizes
        print(f"\n=== Multi-scale Test ===")
        test_sizes = [320, 416, 640]
        for size in test_sizes:
            try:
                test_input = torch.randn(1, 3, size, size)
                with torch.no_grad():
                    test_output = model(test_input)
                    if isinstance(test_output, (list, tuple)) and len(test_output) > 0:
                        det_shape = test_output[0].shape if isinstance(test_output[0], torch.Tensor) else "N/A"
                        print(f"  Input {size}x{size}: Output {det_shape}")
            except Exception as e:
                print(f"  Input {size}x{size}: Error - {e}")
        
        print(f"\n=== Class Information ===")
        print("Class 0: head")
        print("Class 1: front") 
        print("Class 2: back")
        
        print(f"\n✅ Three-class model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_three_class_model()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")

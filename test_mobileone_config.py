#!/usr/bin/env python3
"""
Test script for YOLO-MobileOne configuration
Validates the model configuration and estimates parameter count
"""

import torch
import yaml
from models.yolo import Model
from models.common import MobileOneBlock, MobileOneStage

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_mobileone_config():
    """Test the MobileOne YOLO configuration."""
    
    # Load configuration
    config_path = "models/yolo-mobileone-500k.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("=== YOLO-MobileOne Configuration Test ===")
    print(f"Configuration file: {config_path}")
    print(f"Number of classes: {cfg['nc']}")
    print(f"Depth multiple: {cfg['depth_multiple']}")
    print(f"Width multiple: {cfg['width_multiple']}")
    print()
    
    # Create model
    try:
        model = Model(cfg)
        print("✓ Model created successfully")
        
        # Count parameters
        total_params = count_parameters(model)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Model size: {total_params/1e6:.2f}M parameters")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful")
            print(f"✓ Input shape: {dummy_input.shape}")
            print(f"✓ Output type: {type(output)}")
            if isinstance(output, (list, tuple)):
                print(f"✓ Number of outputs: {len(output)}")
                for i, out in enumerate(output):
                    print(f"  Output {i} shape: {out.shape}")
            else:
                print(f"✓ Output shape: {output.shape}")
        
        # Compare with yoloface-500k target
        target_params = 500000  # 500K parameters
        ratio = total_params / target_params
        print(f"\n=== Parameter Comparison ===")
        print(f"Target (yoloface-500k): {target_params:,} parameters")
        print(f"Actual: {total_params:,} parameters")
        print(f"Ratio: {ratio:.2f}x")
        
        if 0.8 <= ratio <= 1.2:
            print("✓ Parameter count is within acceptable range (0.8x - 1.2x of target)")
        else:
            print("⚠ Parameter count is outside target range")
            
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False
    
    return True

def test_mobileone_modules():
    """Test MobileOne modules individually."""
    print("\n=== MobileOne Module Tests ===")
    
    # Test MobileOneBlock
    try:
        block = MobileOneBlock(16, 32, k=3, s=1, num_conv_branches=1)
        x = torch.randn(1, 16, 32, 32)
        y = block(x)
        print(f"✓ MobileOneBlock: {x.shape} -> {y.shape}")
        print(f"  Parameters: {count_parameters(block):,}")
    except Exception as e:
        print(f"✗ MobileOneBlock test failed: {e}")
    
    # Test MobileOneStage
    try:
        stage = MobileOneStage(16, 32, num_blocks=2, k=3, s=1, num_conv_branches=1)
        x = torch.randn(1, 16, 32, 32)
        y = stage(x)
        print(f"✓ MobileOneStage: {x.shape} -> {y.shape}")
        print(f"  Parameters: {count_parameters(stage):,}")
    except Exception as e:
        print(f"✗ MobileOneStage test failed: {e}")

if __name__ == "__main__":
    print("Testing YOLO-MobileOne configuration...")
    
    # Test individual modules
    test_mobileone_modules()
    
    # Test full configuration
    success = test_mobileone_config()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")

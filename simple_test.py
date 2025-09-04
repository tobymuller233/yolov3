#!/usr/bin/env python3
"""
Simple test for YOLO-MobileOne configuration
"""

import torch
import yaml
import sys
import os
from models.mobileone import reparameterize_model

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop library not available. Install with: pip install thop")

def calculate_flops(model, input_size=(1, 3, 320, 320)):
    """Calculate FLOPs and parameters using thop library."""
    if not THOP_AVAILABLE:
        return None, None, None, None
    
    try:
        # Use the same approach as YOLO's model_info function
        from copy import deepcopy
        
        # Get device from model parameters
        p = next(model.parameters())
        device = p.device
        
        # Create input tensor on the same device as model
        dummy_input = torch.empty(input_size, device=device)
        
        # Calculate FLOPs using thop.profile (same as YOLO's approach)
        flops = profile(deepcopy(model), inputs=(dummy_input,), verbose=False)[0] / 1e9 * 2  # GFLOPs
        
        # Calculate parameters
        params = sum(x.numel() for x in model.parameters())
        
        # Format the results
        flops_formatted = f"{flops:.3f}G"
        params_formatted = f"{params/1e3:.1f}K" if params < 1e6 else f"{params/1e6:.1f}M"
        
        return flops * 1e9, params, flops_formatted, params_formatted
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None, None, None, None

def estimate_inference_time(model, input_size=(1, 3, 368, 640), num_runs=100):
    """Estimate inference time."""
    model.eval()
    
    # Get device from model parameters
    p = next(model.parameters())
    device = p.device
    
    # Create input tensor on the same device as model
    dummy_input = torch.randn(input_size, device=device)
    model = reparameterize_model(model)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"Warning: Warmup failed: {e}")
                break
    
    # Measure time
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"Warning: Inference failed: {e}")
                break
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return avg_time * 1000, fps  # Return time in ms and FPS

def test_config():
    """Test the MobileOne YOLO configuration."""
    
    try:
        # Import after adding path
        from models.yolo import Model
        
        # Load configuration
        config_path = "models/yolo-mobileone-500k.yaml"
        # config_path = "models/yoloface-500kp-layer21-dim120-3class.yaml"
        # config_path = "models/yoloface-500k.yaml"
        print(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print("Configuration loaded successfully!")
        print(f"Number of classes: {cfg['nc']}")
        print(f"Depth multiple: {cfg['depth_multiple']}")
        print(f"Width multiple: {cfg['width_multiple']}")
        
        # Create model
        print("\nCreating model...")
        model = Model(cfg)
        print("✓ Model created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Model size: {total_params/1e6:.2f}M parameters")
        
        # Calculate FLOPs and computational complexity
        print("\n=== Computational Analysis ===")
        flops, params_thop, flops_formatted, params_formatted = calculate_flops(model)
        
        if flops is not None:
            print(f"✓ FLOPs: {flops_formatted}")
            print(f"✓ Parameters (thop): {params_formatted}")
            print(f"✓ FLOPs per parameter: {flops/total_params:.2f}")
        else:
            print("⚠ FLOPs calculation not available (install thop library)")
        
        # Estimate inference time
        print("\n=== Performance Analysis ===")
        try:
            avg_time_ms, fps = estimate_inference_time(model, num_runs=50)
            print(f"✓ Average inference time: {avg_time_ms:.2f} ms")
            print(f"✓ FPS: {fps:.1f}")
        except Exception as e:
            print(f"⚠ Performance test failed: {e}")
        
        # Test forward pass
        print("\n=== Forward Pass Test ===")
        # Get device from model parameters
        p = next(model.parameters())
        device = p.device
        dummy_input = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful!")
            print(f"✓ Input shape: {dummy_input.shape}")
            print(f"✓ Device: {device}")
            
            # Analyze YOLO output structure
            print("\n=== YOLO Output Analysis ===")
            
            if isinstance(output, (list, tuple)):
                print(f"Output is a {type(output).__name__} with {len(output)} elements")
                
                for i, out in enumerate(output):
                    print(f"\nOutput {i}:")
                    if isinstance(out, torch.Tensor):
                        print(f"  Type: torch.Tensor")
                        print(f"  Shape: {out.shape}")
                        print(f"  Dtype: {out.dtype}")
                        print(f"  Device: {out.device}")
                        
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
        
        # Compare with target
        target_params = 500000  # 500K parameters
        ratio = total_params / target_params
        print(f"\n=== Parameter Comparison ===")
        print(f"Target (yoloface-500k): {target_params:,} parameters")
        print(f"Actual: {total_params:,} parameters")
        print(f"Ratio: {ratio:.2f}x")
        
        if 0.8 <= ratio <= 1.2:
            print("✓ Parameter count is within acceptable range!")
        else:
            print("⚠ Parameter count is outside target range")
        
        # Additional computational analysis
        print(f"\n=== Detailed Analysis ===")
        if flops is not None:
            # Calculate efficiency metrics
            flops_per_million = flops / 1e6
            params_per_million = total_params / 1e6
            efficiency_score = flops_per_million / params_per_million if params_per_million > 0 else 0
            
            print(f"✓ FLOPs per million parameters: {flops_per_million/params_per_million:.2f}")
            print(f"✓ Computational efficiency score: {efficiency_score:.2f}")
            
            # Memory estimation (rough)
            model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
            print(f"✓ Estimated model size: {model_size_mb:.2f} MB")
            
            # Compare with typical models
            print(f"\n=== Comparison with Typical Models ===")
            print(f"MobileNetV2 (3.4M params): ~300M FLOPs")
            print(f"EfficientNet-B0 (5.3M params): ~390M FLOPs")
            print(f"YOLOv5n (1.9M params): ~4.1G FLOPs")
            print(f"Our model ({total_params/1e6:.1f}M params): {flops/1e6:.1f}M FLOPs")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing YOLO-MobileOne configuration...")
    success = test_config()
    
    if success:
        print("\n🎉 Test passed!")
    else:
        print("\n❌ Test failed!")

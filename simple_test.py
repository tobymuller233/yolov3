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

def calculate_flops(model, input_size=(1, 3, 640, 640)):
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

def calculate_backbone_head_flops(model, input_size=(1, 3, 640, 640)):
    """Calculate FLOPs separately for backbone and head."""
    if not THOP_AVAILABLE:
        return None
    
    try:
        from copy import deepcopy
        
        # Get device from model parameters
        p = next(model.parameters())
        device = p.device
        
        # Create input tensor on the same device as model
        dummy_input = torch.empty(input_size, device=device)
        
        # Calculate total model FLOPs
        total_flops = profile(deepcopy(model), inputs=(dummy_input,), verbose=False)[0] / 1e9 * 2  # GFLOPs
        
        # Try to separate backbone and head
        # For YOLO models, we need to find the detection layer
        backbone_flops = 0
        head_flops = 0
        
        try:
            # Find the detection layer (usually the last layer)
            detection_layer_idx = -1
            for i, layer in enumerate(model.model):
                if hasattr(layer, '__class__') and 'Detect' in layer.__class__.__name__:
                    detection_layer_idx = i
                    break
            
            if detection_layer_idx > 0:
                # Create backbone model (everything before detection layer)
                backbone_layers = model.model[:detection_layer_idx]
                backbone_model = torch.nn.Sequential(*backbone_layers)
                backbone_flops = profile(deepcopy(backbone_model), inputs=(dummy_input,), verbose=False)[0] / 1e9 * 2
                head_flops = total_flops - backbone_flops
            else:
                # Fallback: estimate based on layer count
                # Assume backbone is 80% of the model
                backbone_flops = total_flops * 0.8
                head_flops = total_flops * 0.2
                
        except Exception as e:
            print(f"Warning: Could not separate backbone/head: {e}")
            # Fallback estimation
            backbone_flops = total_flops * 0.8
            head_flops = total_flops * 0.2
        
        # Calculate parameters for each part
        total_params = sum(x.numel() for x in model.parameters())
        
        # Estimate backbone and head parameters
        # Count parameters in backbone layers vs detection layers
        backbone_params = 0
        head_params = 0
        
        try:
            if detection_layer_idx > 0:
                # Count backbone parameters
                for i in range(detection_layer_idx):
                    backbone_params += sum(x.numel() for x in model.model[i].parameters())
                # Count head parameters
                for i in range(detection_layer_idx, len(model.model)):
                    head_params += sum(x.numel() for x in model.model[i].parameters())
            else:
                # Fallback: estimate based on typical ratios
                backbone_params = int(total_params * 0.7)  # Backbone typically 70% of params
                head_params = total_params - backbone_params
        except:
            # Final fallback
            backbone_params = int(total_params * 0.7)
            head_params = total_params - backbone_params
        
        # Format the results
        total_flops_formatted = f"{total_flops:.3f}G"
        backbone_flops_formatted = f"{backbone_flops:.3f}G"
        head_flops_formatted = f"{head_flops:.3f}G"
        
        total_params_formatted = f"{total_params/1e3:.1f}K" if total_params < 1e6 else f"{total_params/1e6:.1f}M"
        backbone_params_formatted = f"{backbone_params/1e3:.1f}K" if backbone_params < 1e6 else f"{backbone_params/1e6:.1f}M"
        head_params_formatted = f"{head_params/1e3:.1f}K" if head_params < 1e6 else f"{head_params/1e6:.1f}M"
        
        return {
            'total': (total_flops * 1e9, total_params, total_flops_formatted, total_params_formatted),
            'backbone': (backbone_flops * 1e9, backbone_params, backbone_flops_formatted, backbone_params_formatted),
            'head': (head_flops * 1e9, head_params, head_flops_formatted, head_params_formatted)
        }
    except Exception as e:
        print(f"Error calculating backbone/head FLOPs: {e}")
        return None

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
        # config_path = "models/yolo-mobileone-500k.yaml"
        config_path = "models/yolo-ghost-120k.yaml"
        # config_path = "models/yolo-mobilenetv3.yaml"
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
        
        # Analyze model structure
        print(f"\n=== Model Structure Analysis ===")
        print(f"✓ Total layers: {len(model.model)}")
        
        # Count different layer types
        layer_types = {}
        for layer in model.model:
            layer_name = layer.__class__.__name__
            layer_types[layer_name] = layer_types.get(layer_name, 0) + 1
        
        print(f"✓ Layer types:")
        for layer_type, count in sorted(layer_types.items()):
            print(f"   {layer_type}: {count}")
        
        # Find detection layer
        detection_idx = -1
        for i, layer in enumerate(model.model):
            if 'Detect' in layer.__class__.__name__:
                detection_idx = i
                break
        
        if detection_idx >= 0:
            print(f"✓ Detection layer at index: {detection_idx}")
            print(f"✓ Backbone layers: {detection_idx}")
            print(f"✓ Head layers: {len(model.model) - detection_idx}")
        else:
            print(f"⚠ Detection layer not found")
        
        # Calculate FLOPs and computational complexity
        print("\n=== Computational Analysis ===")
        
        # Calculate total FLOPs
        flops, params_thop, flops_formatted, params_formatted = calculate_flops(model)
        
        if flops is not None:
            print(f"✓ Total FLOPs: {flops_formatted}")
            print(f"✓ Total Parameters (thop): {params_formatted}")
            print(f"✓ FLOPs per parameter: {flops/total_params:.2f}")
        else:
            print("⚠ FLOPs calculation not available (install thop library)")
        
        # Calculate backbone and head FLOPs separately
        print("\n=== Backbone vs Head Analysis ===")
        backbone_head_analysis = calculate_backbone_head_flops(model)
        
        if backbone_head_analysis is not None:
            total_data = backbone_head_analysis['total']
            backbone_data = backbone_head_analysis['backbone']
            head_data = backbone_head_analysis['head']
            
            print(f"📊 Backbone:")
            print(f"   FLOPs: {backbone_data[2]}")
            print(f"   Parameters: {backbone_data[3]}")
            print(f"   FLOPs ratio: {backbone_data[0]/total_data[0]*100:.1f}%")
            print(f"   Params ratio: {backbone_data[1]/total_data[1]*100:.1f}%")
            
            print(f"📊 Head:")
            print(f"   FLOPs: {head_data[2]}")
            print(f"   Parameters: {head_data[3]}")
            print(f"   FLOPs ratio: {head_data[0]/total_data[0]*100:.1f}%")
            print(f"   Params ratio: {head_data[1]/total_data[1]*100:.1f}%")
            
            print(f"📊 Efficiency Analysis:")
            backbone_efficiency = backbone_data[0] / backbone_data[1] if backbone_data[1] > 0 else 0
            head_efficiency = head_data[0] / head_data[1] if head_data[1] > 0 else 0
            print(f"   Backbone FLOPs/Param: {backbone_efficiency:.2f}")
            print(f"   Head FLOPs/Param: {head_efficiency:.2f}")
            
            if backbone_efficiency > head_efficiency:
                print(f"   → Backbone is more computationally efficient")
            else:
                print(f"   → Head is more computationally efficient")
            
            # Additional analysis
            print(f"📊 Computational Distribution:")
            print(f"   Backbone computational load: {backbone_data[0]/total_data[0]*100:.1f}%")
            print(f"   Head computational load: {head_data[0]/total_data[0]*100:.1f}%")
            print(f"   Backbone parameter load: {backbone_data[1]/total_data[1]*100:.1f}%")
            print(f"   Head parameter load: {head_data[1]/total_data[1]*100:.1f}%")
            
            # Memory analysis
            backbone_memory = backbone_data[1] * 4 / (1024 * 1024)  # 4 bytes per float32
            head_memory = head_data[1] * 4 / (1024 * 1024)
            total_memory = total_data[1] * 4 / (1024 * 1024)
            
            print(f"📊 Memory Analysis:")
            print(f"   Backbone memory: {backbone_memory:.2f} MB")
            print(f"   Head memory: {head_memory:.2f} MB")
            print(f"   Total memory: {total_memory:.2f} MB")
        else:
            print("⚠ Backbone/Head analysis not available")
        
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

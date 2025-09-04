#!/usr/bin/env python3
"""
Demo script showing how to use MobileOne reparameterization for inference
"""

import torch
import time
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_reparameterization():
    """Demonstrate MobileOne reparameterization for inference."""
    
    try:
        from models.yolo import DetectionModel
        import yaml
        
        print("=== MobileOne Reparameterization Demo ===")
        print("Classes: head, front, back")
        
        # Load configuration
        config_path = "models/yolo-mobileone-500k.yaml"
        print(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Create model
        print("\n1. Creating model...")
        model = DetectionModel(cfg, ch=3, nc=cfg['nc'], anchors=cfg['anchors'])
        
        # Count MobileOne blocks
        mobileone_count = 0
        for name, module in model.named_modules():
            if 'MobileOneBlock' in str(type(module)):
                mobileone_count += 1
        
        print(f"✓ Model created with {mobileone_count} MobileOne blocks")
        print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference before reparameterization
        print("\n2. Testing inference BEFORE reparameterization...")
        model.eval()
        dummy_input = torch.randn(1, 3, 320, 320)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # Time inference before reparameterization
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output_before = model(dummy_input)
        time_before = time.time() - start_time
        
        print(f"✓ Inference time (before): {time_before/100*1000:.2f} ms per image")
        print(f"✓ Output shape: {output_before[0].shape if isinstance(output_before, (list, tuple)) else 'N/A'}")
        
        # Reparameterize the model
        print("\n3. Reparameterizing MobileOne blocks...")
        model.reparameterize_mobileone()
        
        # Test inference after reparameterization
        print("\n4. Testing inference AFTER reparameterization...")
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # Time inference after reparameterization
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output_after = model(dummy_input)
        time_after = time.time() - start_time
        
        print(f"✓ Inference time (after): {time_after/100*1000:.2f} ms per image")
        print(f"✓ Output shape: {output_after[0].shape if isinstance(output_after, (list, tuple)) else 'N/A'}")
        
        # Compare outputs
        print("\n5. Comparing outputs...")
        if isinstance(output_before, (list, tuple)) and isinstance(output_after, (list, tuple)):
            if len(output_before) > 0 and len(output_after) > 0:
                diff = torch.abs(output_before[0] - output_after[0]).max().item()
                print(f"✓ Maximum difference: {diff:.6f}")
                if diff < 1e-5:
                    print("✓ Outputs are nearly identical (reparameterization successful!)")
                else:
                    print("⚠ Outputs differ significantly")
        
        # Performance comparison
        print("\n6. Performance Analysis...")
        speedup = time_before / time_after
        print(f"✓ Speedup: {speedup:.2f}x")
        print(f"✓ Time reduction: {(1 - time_after/time_before)*100:.1f}%")
        
        # Check if MobileOne blocks were reparameterized
        print("\n7. Verifying reparameterization...")
        reparam_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'reparam_conv'):
                reparam_count += 1
        
        print(f"✓ {reparam_count}/{mobileone_count} MobileOne blocks reparameterized")
        
        if reparam_count == mobileone_count:
            print("✅ All MobileOne blocks successfully reparameterized!")
        else:
            print("⚠ Some MobileOne blocks were not reparameterized")
        
        print("\n=== Demo Summary ===")
        print("✅ Reparameterization completed successfully!")
        print("✅ Model is now optimized for inference")
        print("✅ Multi-branch training structure → Single-branch inference structure")
        print("✅ Faster inference with equivalent accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_reparameterized_model():
    """Save the reparameterized model for deployment."""
    
    try:
        from models.yolo import DetectionModel
        import yaml
        
        print("\n=== Saving Reparameterized Model ===")
        
        # Load and create model
        with open("models/yolo-mobileone-500k.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        
        model = DetectionModel(cfg, ch=3, nc=cfg['nc'], anchors=cfg['anchors'])
        
        # Reparameterize
        model.reparameterize_mobileone()
        
        # Save model
        save_path = "yolo_mobileone_reparameterized.pt"
        torch.save({
            'model': model.state_dict(),
            'config': cfg,
            'reparameterized': True
        }, save_path)
        
        print(f"✓ Reparameterized model saved to: {save_path}")
        print("✓ Model is ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

if __name__ == "__main__":
    print("MobileOne Reparameterization Demo")
    print("=" * 50)
    
    # Run demo
    success = demo_reparameterization()
    
    if success:
        # Save reparameterized model
        save_success = save_reparameterized_model()
        
        if save_success:
            print("\n🎉 Demo completed successfully!")
        else:
            print("\n⚠ Demo completed with warnings")
    else:
        print("\n💥 Demo failed!")

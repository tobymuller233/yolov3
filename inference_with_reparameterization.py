#!/usr/bin/env python3
"""
Proper inference script with MobileOne reparameterization
This script shows the correct way to use reparameterization during inference.
"""

import torch
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_model_for_inference(weights_path, device='cpu'):
    """Load model properly for inference with reparameterization."""
    
    try:
        from models.experimental import attempt_load
        
        print(f"Loading model from: {weights_path}")
        
        # Load model without automatic fusing
        model = attempt_load(weights_path, device=device, inplace=True, fuse=False)
        
        # Set to evaluation mode
        model.eval()
        
        print("✓ Model loaded successfully")
        print(f"✓ Model device: {next(model.parameters()).device}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def reparameterize_and_fuse(model):
    """Properly reparameterize MobileOne blocks and then fuse the model."""
    
    try:
        print("\n=== Reparameterization Process ===")
        
        # Step 1: Reparameterize MobileOne blocks
        print("1. Reparameterizing MobileOne blocks...")
        if hasattr(model, 'reparameterize_mobileone'):
            model.reparameterize_mobileone()
        else:
            # Manual reparameterization if method doesn't exist
            def _reparameterize_module(module):
                if hasattr(module, 'reparameterize'):
                    module.reparameterize()
                for child in module.children():
                    _reparameterize_module(child)
            
            _reparameterize_module(model)
            print("✓ MobileOne blocks reparameterized")
        
        # Step 2: Fuse Conv+BN layers
        print("2. Fusing Conv+BN layers...")
        model.fuse()
        
        print("✓ Model optimization completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return False

def run_inference(model, input_tensor):
    """Run inference with the optimized model."""
    
    try:
        print("\n=== Running Inference ===")
        
        # Ensure input is on the correct device
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Input device: {input_tensor.device}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print("✓ Inference completed successfully!")
        
        # Analyze outputs
        if isinstance(outputs, (list, tuple)):
            print(f"Output type: {type(outputs).__name__} with {len(outputs)} elements")
            if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                print(f"Detection output shape: {outputs[0].shape}")
                print(f"Expected format: [batch, detections, 8] for 3 classes")
        else:
            print(f"Output type: {type(outputs)}")
            if isinstance(outputs, torch.Tensor):
                print(f"Output shape: {outputs.shape}")
        
        return outputs
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main inference function."""
    
    print("=== MobileOne YOLO Inference with Reparameterization ===")
    
    # Configuration
    weights_path = "runs/mobileone_yolo/exp2/weights/best.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (1, 3, 640, 640)
    
    print(f"Device: {device}")
    print(f"Input size: {input_size}")
    
    # Check if weights file exists
    if not Path(weights_path).exists():
        print(f"❌ Weights file not found: {weights_path}")
        print("Please provide the correct path to your trained model.")
        return False
    
    # Step 1: Load model
    model = load_model_for_inference(weights_path, device)
    if model is None:
        return False
    
    # Step 2: Optimize model for inference
    success = reparameterize_and_fuse(model)
    if not success:
        return False
    
    # Step 3: Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Step 4: Run inference
    outputs = run_inference(model, dummy_input)
    if outputs is None:
        return False
    
    print("\n✅ Inference pipeline completed successfully!")
    print("\n=== Usage Summary ===")
    print("1. Load model with fuse=False")
    print("2. Reparameterize MobileOne blocks")
    print("3. Fuse Conv+BN layers")
    print("4. Run inference")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

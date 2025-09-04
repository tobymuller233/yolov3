#!/usr/bin/env python3
"""
Verify the layer indices in the YAML configuration
"""

import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_indices():
    """Verify layer indices are correct."""
    
    try:
        print("Verifying layer indices in YAML configuration...")
        
        # Load config
        with open("models/yolo-mobileone-500k.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        
        print("=== Backbone Layers ===")
        backbone = cfg['backbone']
        for i, layer in enumerate(backbone):
            print(f"Layer {i}: {layer}")
        
        print(f"\n=== Head Layers ===")
        head = cfg['head']
        for i, layer in enumerate(head):
            layer_idx = len(backbone) + i
            print(f"Layer {layer_idx}: {layer}")
        
        print(f"\n=== Total Layers ===")
        total_layers = len(backbone) + len(head)
        print(f"Total layers: {total_layers}")
        print(f"Last layer index: {total_layers - 1}")
        
        # Check Detect layer indices
        detect_layer = head[-1]
        if detect_layer[2] == 'Detect':
            detect_indices = detect_layer[0]
            print(f"\n=== Detect Layer Analysis ===")
            print(f"Detect layer indices: {detect_indices}")
            print(f"Expected: [29, 24, 19] (P3, P4, P5)")
            
            if detect_indices == [29, 24, 19]:
                print("✓ Detect layer indices are correct!")
            else:
                print("⚠ Detect layer indices may be incorrect!")
        
        print(f"\n✅ Index verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_indices()

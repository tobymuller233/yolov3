#!/usr/bin/env python3
"""
Test script to verify val.py reparameterization integration
"""

import subprocess
import sys
import os

def test_val_with_reparameterize():
    """Test val.py with reparameterization flag"""
    
    print("=== Testing val.py with reparameterization ===")
    
    # Test command
    cmd = [
        "python3", "val.py",
        "--weights", "runs/mobileone_yolo/exp2/weights/best.pt",
        "--data", "data/SCUT_HEAD_A_B_stu_three_v2.yaml",
        "--reparameterize",
        "--batch-size", "1",
        "--imgsz", "640"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Test passed! Reparameterization integration works correctly.")
            return True
        else:
            print(f"❌ Test failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_val_without_reparameterize():
    """Test val.py without reparameterization flag (should still fail with original error)"""
    
    print("\n=== Testing val.py without reparameterization ===")
    
    # Test command
    cmd = [
        "python3", "val.py",
        "--weights", "runs/mobileone_yolo/exp2/weights/best.pt",
        "--data", "data/SCUT_HEAD_A_B_stu_three_v2.yaml",
        "--batch-size", "1",
        "--imgsz", "640"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if "RuntimeError: Sizes of tensors must match except in dimension 1" in result.stderr:
            print("✅ Expected error occurred (Concat layer size mismatch)")
            print("This confirms the original issue still exists without reparameterization")
            return True
        else:
            print("❌ Unexpected result")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Main test function"""
    
    print("Testing val.py reparameterization integration...")
    
    # Check if weights file exists
    weights_path = "runs/mobileone_yolo/exp2/weights/best.pt"
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        print("Please ensure the model weights exist before running this test.")
        return False
    
    # Test without reparameterization (should show original error)
    test1_passed = test_val_without_reparameterize()
    
    # Test with reparameterization (should work)
    test2_passed = test_val_with_reparameterize()
    
    print("\n=== Test Summary ===")
    print(f"Without reparameterization: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"With reparameterization: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Reparameterization integration is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


#!/usr/bin/env python
"""
Test script for InternVL debug logging functionality.

This script runs a minimal inference to verify that debug logging
is working correctly for token pruning research.
"""

import sys
import os

# Add VLMEvalKit to path
sys.path.insert(0, os.path.dirname(__file__))

from vlmeval.config import supported_VLM
from PIL import Image
import requests
from io import BytesIO

def download_test_image():
    """Download a test image."""
    url = "https://raw.githubusercontent.com/open-mmlab/mmpretrain/main/demo/demo.JPEG"
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        # Fallback: create a simple test image
        print("Creating synthetic test image...")
        return Image.new('RGB', (640, 480), color='red')

def test_internvl_debug_logging():
    """Test InternVL debug logging."""
    print("="*80)
    print("InternVL Debug Logging Test")
    print("="*80)
    
    # Initialize model (smallest one for testing)
    print("\n1. Initializing InternVL model...")
    try:
        model_cls = supported_VLM['InternVL3_5-8B']
        model = model_cls()
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False
    
    # Check if debug logger is initialized
    if hasattr(model, 'debug_logger'):
        print("✓ Debug logger initialized")
        print(f"  Log file: {model.debug_logger.log_file}")
    else:
        print("✗ Debug logger not found")
        return False
    
    # Prepare test input
    print("\n2. Preparing test input...")
    img = download_test_image()
    img_path = "/tmp/test_image.jpg"
    img.save(img_path)
    print(f"✓ Test image saved to {img_path}")
    
    # Create test message
    message = [
        {'type': 'image', 'value': img_path},
        {'type': 'text', 'value': 'Describe this image briefly.'}
    ]
    
    # Run inference
    print("\n3. Running inference with debug logging...")
    try:
        response = model.generate(message)
        print(f"✓ Inference completed")
        print(f"  Response: {response[:100]}...")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check log file
    print("\n4. Checking log file...")
    if os.path.exists(model.debug_logger.log_file):
        with open(model.debug_logger.log_file, 'r') as f:
            content = f.read()
            print(f"✓ Log file created ({len(content)} bytes)")
            
            # Check for expected log stages
            expected_stages = [
                "1. Image Preprocessing - Input",
                "1. Image Preprocessing - Output",
                "2. Token Generation - Input",
                "3. ViT Encoding",
                "4. Embedding Fusion",
                "5. LLM Input"
            ]
            
            found_stages = []
            for stage in expected_stages:
                if stage in content:
                    found_stages.append(stage)
            
            print(f"  Found {len(found_stages)}/{len(expected_stages)} expected log stages:")
            for stage in found_stages:
                print(f"    ✓ {stage}")
            
            missing = set(expected_stages) - set(found_stages)
            if missing:
                print(f"  Missing stages:")
                for stage in missing:
                    print(f"    ✗ {stage}")
    else:
        print(f"✗ Log file not found: {model.debug_logger.log_file}")
        return False
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print(f"Check the log file for detailed output: {model.debug_logger.log_file}")
    print("="*80)
    
    return True

if __name__ == '__main__':
    success = test_internvl_debug_logging()
    sys.exit(0 if success else 1)


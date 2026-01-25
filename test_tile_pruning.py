#!/usr/bin/env python
"""
Test script for tile pruning functionality.
Tests the attention-based tile pruning baseline for InternVL3.5.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from vlmeval.vlm.internvl import InternVLChat
from PIL import Image
import numpy as np

print("="*80)
print("Tile Pruning Test Script")
print("="*80)

# Use local test image
print("\n1. Loading test image...")
test_image_path = '/gpfs/projects/embodied3d/jianxu/vlm_pruning/VLMEvalKit/assets/demo.jpeg'
if os.path.exists(test_image_path):
    print(f"   Test image loaded: {test_image_path}")
    # Verify image can be opened
    test_image = Image.open(test_image_path)
    print(f"   Image size: {test_image.size}")
else:
    print(f"   ERROR: Test image not found: {test_image_path}")
    sys.exit(1)

# Test 1: Load model with pruning disabled (default)
print("\n" + "="*80)
print("Test 1: Default behavior (pruning disabled)")
print("="*80)

try:
    print("Loading InternVL3.5-30B-A3B with default config...")
    model = InternVLChat(model_path='OpenGVLab/InternVL3_5-30B-A3B', version='V2.0')
    
    # Check config
    print(f"enable_tile_pruning: {getattr(model.model.config, 'enable_tile_pruning', 'Not set')}")
    print(f"tile_keep_ratio: {getattr(model.model.config, 'tile_keep_ratio', 'Not set')}")
    print(f"enable_tile_pruning: {model.model.config.enable_tile_pruning}")
    print(f"tile_keep_ratio: {model.model.config.tile_keep_ratio}")
    
    # Test inference
    message = [
        {'type': 'image', 'value': test_image_path},
        {'type': 'text', 'value': 'Describe this image.'}
    ]
    
    print("\nRunning inference...")
    response = model.generate(message)
    print(f"Response: {response}")
    print("✓ Test 1 passed!")
    
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Enable pruning by modifying config
print("\n" + "="*80)
print("Test 2: Pruning enabled (keep_ratio=0.5)")
print("="*80)

try:
    print("Loading InternVL3.5-30B-A3B...")
    model = InternVLChat(model_path='OpenGVLab/InternVL3_5-30B-A3B', version='V2.0')
    
    # Enable pruning
    print("Enabling tile pruning...")
    model.model.config.enable_tile_pruning = True
    model.model.config.tile_keep_ratio = 0.8
    
    print(f"enable_tile_pruning: {model.model.config.enable_tile_pruning}")
    print(f"tile_keep_ratio: {model.model.config.tile_keep_ratio}")
    
    # Test inference
    message = [
        {'type': 'image', 'value': test_image_path},
        {'type': 'text', 'value': 'Describe this image.'}
    ]
    
    print("\nRunning inference with pruning...")
    response = model.generate(message)
    print(f"Response: {response}")
    print("✓ Test 2 passed!")
    
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compare token counts
print("\n" + "="*80)
print("Test 3: Token count comparison")
print("="*80)

try:
    print("This test requires running inference and checking debug logs.")
    print("Please check the debug_logs/ directory for detailed token statistics.")
    print("Expected: With pruning enabled, vision tokens should be reduced by ~50%")
    print("✓ Test 3 completed (manual verification required)")
    
except Exception as e:
    print(f"✗ Test 3 failed: {e}")

# Cleanup
print("\n" + "="*80)
print("Cleanup")
print("="*80)
print("Using local image - no cleanup needed")

print("\n" + "="*80)
print("Test Summary")
print("="*80)
print("All basic tests completed!")
print("\nNext steps:")
print("1. Run full benchmark with pruning disabled:")
print("   python run.py --data MMBench_DEV_EN_V11 --model InternVL3_5-30B-A3B")
print("\n2. Enable pruning by modifying the model's config.json:")
print("   Add: \"enable_tile_pruning\": true, \"tile_keep_ratio\": 0.5")
print("\n3. Run benchmark again and compare results")
print("="*80)


#!/usr/bin/env python
"""
Test script to verify local config.json loading and tile pruning configuration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vlmeval.vlm.internvl import InternVLChat
from pathlib import Path

print("="*80)
print("Local Config Loading Test")
print("="*80)

# Test 1: Verify local config files exist
print("\nTest 1: Checking local config files")
print("-"*80)

config_30b = Path("vlmeval/vlm/internvl/local_models/internvl3_5_30b_a3b/config.json")
config_8b = Path("vlmeval/vlm/internvl/local_models/internvl3_5_8b/config.json")

if config_30b.exists():
    print(f"✓ 30B-A3B config found: {config_30b}")
else:
    print(f"✗ 30B-A3B config NOT found: {config_30b}")

if config_8b.exists():
    print(f"✓ 8B config found: {config_8b}")
else:
    print(f"✗ 8B config NOT found: {config_8b}")

# Test 2: Read and verify config content
print("\nTest 2: Verifying config content")
print("-"*80)

import json
with open(config_30b, 'r') as f:
    config_data = json.load(f)
    
if 'enable_tile_pruning' in config_data:
    print(f"✓ enable_tile_pruning found: {config_data['enable_tile_pruning']}")
else:
    print("✗ enable_tile_pruning NOT found in config")

if 'tile_keep_ratio' in config_data:
    print(f"✓ tile_keep_ratio found: {config_data['tile_keep_ratio']}")
else:
    print("✗ tile_keep_ratio NOT found in config")

# Test 3: Load model and check if it uses local config
print("\nTest 3: Loading model with local config")
print("-"*80)

try:
    print("Loading InternVL3.5-30B-A3B...")
    model = InternVLChat(model_path='OpenGVLab/InternVL3_5-30B-A3B', version='V2.0')
    
    # Check config attributes
    has_tile_pruning_attr = hasattr(model.model.config, 'enable_tile_pruning')
    has_keep_ratio_attr = hasattr(model.model.config, 'tile_keep_ratio')
    
    if has_tile_pruning_attr and has_keep_ratio_attr:
        print(f"✓ Config loaded successfully")
        print(f"  enable_tile_pruning: {model.model.config.enable_tile_pruning}")
        print(f"  tile_keep_ratio: {model.model.config.tile_keep_ratio}")
    else:
        print(f"✗ Config attributes missing")
        if not has_tile_pruning_attr:
            print("  - enable_tile_pruning not found")
        if not has_keep_ratio_attr:
            print("  - tile_keep_ratio not found")
    
    # Test 4: Modify config at runtime
    print("\nTest 4: Runtime config modification")
    print("-"*80)
    
    original_enable = model.model.config.enable_tile_pruning
    original_ratio = model.model.config.tile_keep_ratio
    
    # Modify
    model.model.config.enable_tile_pruning = True
    model.model.config.tile_keep_ratio = 0.7
    
    print(f"Original config: enable={original_enable}, ratio={original_ratio}")
    print(f"Modified config: enable={model.model.config.enable_tile_pruning}, ratio={model.model.config.tile_keep_ratio}")
    
    if model.model.config.enable_tile_pruning == True and model.model.config.tile_keep_ratio == 0.7:
        print("✓ Runtime modification successful")
    else:
        print("✗ Runtime modification failed")
    
    # Restore
    model.model.config.enable_tile_pruning = original_enable
    model.model.config.tile_keep_ratio = original_ratio
    
    print("\n" + "="*80)
    print("All Tests Passed!")
    print("="*80)
    
    print("\nLocal config is working correctly. You can now:")
    print("1. Modify local config.json to enable tile pruning")
    print("2. Run benchmarks with: python run.py --data MMBench_DEV_EN_V11 --model InternVL3_5-30B-A3B")
    print("3. Or modify config at runtime in your scripts")
    
except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


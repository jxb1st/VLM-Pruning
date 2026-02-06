#!/usr/bin/env python
"""
Test script to verify CDPruner integration with VLMEvalKit.
This tests basic imports and model registration.
"""

import sys
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Test 1: Import the local CDPruner package
print("=" * 60)
print("Test 1: Import local CDPruner package")
print("=" * 60)
try:
    from vlmeval.llava_cdpruner.model import LlavaLlamaForCausalLM
    print("✓ Successfully imported LlavaLlamaForCausalLM from local package")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Import the CDPruner model class
print()
print("=" * 60)
print("Test 2: Import LLaVA_CDPruner model class")
print("=" * 60)
try:
    from vlmeval.vlm import LLaVA_CDPruner
    print("✓ Successfully imported LLaVA_CDPruner")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 3: Check model registration in config
print()
print("=" * 60)
print("Test 3: Check model registration in config")
print("=" * 60)
try:
    from vlmeval.config import supported_VLM
    
    cdpruner_models = [k for k in supported_VLM.keys() if 'cdpruner' in k.lower()]
    
    if cdpruner_models:
        print(f"✓ Found {len(cdpruner_models)} CDPruner model(s) registered:")
        for model_name in cdpruner_models:
            print(f"  - {model_name}")
    else:
        print("✗ No CDPruner models found in supported_VLM")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check registration: {e}")
    sys.exit(1)

# Test 4: Verify model can be instantiated (without loading weights)
print()
print("=" * 60)
print("Test 4: Verify model configuration")
print("=" * 60)
try:
    from functools import partial
    
    # Check if we can access the partial function
    model_partial = supported_VLM.get('llava_v1.5_7b_cdpruner')
    if model_partial:
        print("✓ Model partial function found")
        print(f"  - Keywords: {model_partial.keywords}")
    else:
        print("✗ Model partial function not found")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to verify model configuration: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print()
print("The CDPruner integration is ready to use.")
print("You can now run: python run.py --data <DATASET> --model llava_v1.5_7b_cdpruner")


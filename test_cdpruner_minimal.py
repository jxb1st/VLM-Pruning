#!/usr/bin/env python
"""
Minimal test to verify CDPruner core integration.
"""

import sys
print(f"Python: {sys.version}")
print()

# Test 1: Import the local CDPruner model files
print("=" * 60)
print("Test 1: Import CDPruner local model package")
print("=" * 60)
try:
    from vlmeval.llava_cdpruner.model import LlavaLlamaForCausalLM
    from vlmeval.llava_cdpruner.model.builder import load_pretrained_model
    from vlmeval.llava_cdpruner.constants import IMAGE_TOKEN_INDEX
    print("✓ Successfully imported CDPruner local package components")
    print(f"  - LlavaLlamaForCausalLM: {LlavaLlamaForCausalLM}")
    print(f"  - load_pretrained_model: {load_pretrained_model}")
    print(f"  - IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import the CDPruner model class wrapper
print()
print("=" * 60)
print("Test 2: Import LLaVA_CDPruner wrapper class")
print("=" * 60)
try:
    from vlmeval.vlm.llava.llava_cdpruner import LLaVA_CDPruner
    print(f"✓ Successfully imported LLaVA_CDPruner")
    print(f"  - Class: {LLaVA_CDPruner}")
    print(f"  - INSTALL_REQ: {LLaVA_CDPruner.INSTALL_REQ}")
    print(f"  - INTERLEAVE: {LLaVA_CDPruner.INTERLEAVE}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
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
        for model_name in sorted(cdpruner_models):
            model_partial = supported_VLM[model_name]
            visual_token_num = model_partial.keywords.get('visual_token_num', 'N/A')
            print(f"  - {model_name:40s} (visual_token_num={visual_token_num})")
    else:
        print("✗ No CDPruner models found in supported_VLM")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✓ All core integration tests passed!")
print("=" * 60)
print()
print("The CDPruner integration is successfully deployed.")
print()
print("You can now run evaluations with:")
print("  python run.py --data MMBench_DEV_EN --model llava_v1.5_7b_cdpruner")
print()
print("Available CDPruner models:")
for model_name in sorted(cdpruner_models):
    print(f"  - {model_name}")


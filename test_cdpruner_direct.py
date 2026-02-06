#!/usr/bin/env python
"""
Direct import test bypassing VLMEvalKit's complex dependencies.
"""

import sys
import os

# Add to path
sys.path.insert(0, '/gpfs/projects/embodied3d/jianxu/vlm_pruning/VLMEvalKit')

print("=" * 60)
print("Direct import test for CDPruner integration")
print("=" * 60)
print()

# Test 1: Import CDPruner local package directly
print("Test 1: Import CDPruner local model package")
try:
    from vlmeval.llava_cdpruner.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig
    from vlmeval.llava_cdpruner.constants import IMAGE_TOKEN_INDEX
    print("✓ Successfully imported CDPruner local package")
    print(f"  - LlavaLlamaForCausalLM: {LlavaLlamaForCausalLM}")
    print(f"  - IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check visual_token_num parameter
print()
print("Test 2: Verify visual_token_num in __init__")
try:
    import inspect
    sig = inspect.signature(LlavaLlamaForCausalLM.__init__)
    params = list(sig.parameters.keys())
    print(f"  Parameters: {params}")
    if 'visual_token_num' in params:
        print("✓ visual_token_num parameter found in LlavaLlamaForCausalLM.__init__")
    else:
        print("✗ visual_token_num parameter NOT found")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Check CDPruner modifications exist
print()
print("Test 3: Verify CDPruner modifications")
try:
    # Check if get_visual_token_num method exists
    if hasattr(LlavaLlamaForCausalLM, 'get_visual_token_num'):
        print("✓ get_visual_token_num method found")
    else:
        print("✗ get_visual_token_num method NOT found")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Verify file structure
print()
print("Test 4: Verify CDPruner file structure")
base_path = '/gpfs/projects/embodied3d/jianxu/vlm_pruning/VLMEvalKit/vlmeval'
required_files = [
    'llava_cdpruner/__init__.py',
    'llava_cdpruner/constants.py',
    'llava_cdpruner/mm_utils.py',
    'llava_cdpruner/model/__init__.py',
    'llava_cdpruner/model/builder.py',
    'llava_cdpruner/model/llava_arch.py',
    'llava_cdpruner/model/language_model/llava_llama.py',
    'llava_cdpruner/model/multimodal_encoder/clip_encoder.py',
    'vlm/llava/llava_cdpruner.py',
]

all_exist = True
for file in required_files:
    full_path = os.path.join(base_path, file)
    if os.path.exists(full_path):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} MISSING")
        all_exist = False

if not all_exist:
    sys.exit(1)

# Test 5: Verify LLaVA_CDPruner wrapper can be imported
print()
print("Test 5: Import LLaVA_CDPruner wrapper")
try:
    # Direct import without going through vlmeval.__init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "llava_cdpruner",
        "/gpfs/projects/embodied3d/jianxu/vlm_pruning/VLMEvalKit/vlmeval/vlm/llava/llava_cdpruner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    LLaVA_CDPruner = module.LLaVA_CDPruner
    print(f"✓ Successfully imported LLaVA_CDPruner")
    print(f"  - Class: {LLaVA_CDPruner}")
    print(f"  - INSTALL_REQ: {LLaVA_CDPruner.INSTALL_REQ}")
    print(f"  - INTERLEAVE: {LLaVA_CDPruner.INTERLEAVE}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✓✓✓ All integration tests passed! ✓✓✓")
print("=" * 60)
print()
print("CDPruner integration is successfully deployed in VLMEvalKit.")
print()
print("Next steps:")
print("  1. Ensure you activate the correct environment (with all dependencies)")
print("  2. Run: python run.py --data <DATASET> --model llava_v1.5_7b_cdpruner")
print()
print("Available model names:")
print("  - llava_v1.5_7b_cdpruner        (64 tokens)")
print("  - llava_v1.5_7b_cdpruner_32     (32 tokens)")
print("  - llava_v1.5_7b_cdpruner_128    (128 tokens)")


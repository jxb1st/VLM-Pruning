#!/bin/bash
# Test script to verify all CDPruner files are in place

echo "============================================================"
echo "CDPruner Integration - File Structure Verification"
echo "============================================================"
echo ""

BASE="/gpfs/projects/embodied3d/jianxu/vlm_pruning/VLMEvalKit/vlmeval"

echo "Checking CDPruner local package files:"
echo ""

files=(
    "llava_cdpruner/__init__.py"
    "llava_cdpruner/constants.py"
    "llava_cdpruner/mm_utils.py"
    "llava_cdpruner/model/__init__.py"
    "llava_cdpruner/model/builder.py"
    "llava_cdpruner/model/llava_arch.py"
    "llava_cdpruner/model/utils.py"
    "llava_cdpruner/model/language_model/__init__.py"
    "llava_cdpruner/model/language_model/llava_llama.py"
    "llava_cdpruner/model/multimodal_encoder/__init__.py"
    "llava_cdpruner/model/multimodal_encoder/builder.py"
    "llava_cdpruner/model/multimodal_encoder/clip_encoder.py"
    "llava_cdpruner/model/multimodal_projector/__init__.py"
    "llava_cdpruner/model/multimodal_projector/builder.py"
    "vlm/llava/llava_cdpruner.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$BASE/$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file MISSING"
        all_exist=false
    fi
done

echo ""
echo "Checking key modifications:"
echo ""

# Check for visual_token_num in llava_llama.py
if grep -q "visual_token_num" "$BASE/llava_cdpruner/model/language_model/llava_llama.py"; then
    echo "✓ visual_token_num found in llava_llama.py"
else
    echo "✗ visual_token_num NOT found in llava_llama.py"
    all_exist=false
fi

# Check for DPP algorithm in llava_arch.py
if grep -q "Conditional DPP" "$BASE/llava_cdpruner/model/llava_arch.py" || grep -q "kernel = relevance" "$BASE/llava_cdpruner/model/llava_arch.py"; then
    echo "✓ DPP pruning algorithm found in llava_arch.py"
else
    echo "✗ DPP algorithm NOT found in llava_arch.py"
    all_exist=false
fi

# Check for CLIP text tower in clip_encoder.py
if grep -q "load_text_tower" "$BASE/llava_cdpruner/model/multimodal_encoder/clip_encoder.py"; then
    echo "✓ CLIP text tower loading found in clip_encoder.py"
else
    echo "✗ CLIP text tower NOT found in clip_encoder.py"
    all_exist=false
fi

# Check for CDPruner in config.py
if grep -q "llava_v1.5_7b_cdpruner" "$BASE/config.py"; then
    echo "✓ CDPruner models registered in config.py"
else
    echo "✗ CDPruner models NOT registered in config.py"
    all_exist=false
fi

# Check for CDPruner in vlm/__init__.py
if grep -q "LLaVA_CDPruner" "$BASE/vlm/__init__.py"; then
    echo "✓ LLaVA_CDPruner imported in vlm/__init__.py"
else
    echo "✗ LLaVA_CDPruner NOT imported in vlm/__init__.py"
    all_exist=false
fi

echo ""
echo "============================================================"
if [ "$all_exist" = true ]; then
    echo "✓✓✓ All files and modifications are in place! ✓✓✓"
    echo "============================================================"
    echo ""
    echo "CDPruner integration is successfully deployed."
    echo ""
    echo "To use CDPruner with VLMEvalKit:"
    echo "  1. Activate an environment with all VLMEvalKit dependencies"
    echo "  2. Run: python run.py --data <DATASET> --model llava_v1.5_7b_cdpruner"
    echo ""
    echo "Available CDPruner models:"
    echo "  - llava_v1.5_7b_cdpruner        (64 visual tokens)"
    echo "  - llava_v1.5_7b_cdpruner_32     (32 visual tokens)"
    echo "  - llava_v1.5_7b_cdpruner_128    (128 visual tokens)"
    exit 0
else
    echo "✗✗✗ Some files or modifications are missing ✗✗✗"
    echo "============================================================"
    exit 1
fi


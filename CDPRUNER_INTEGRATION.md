# CDPruner Integration with VLMEvalKit

## Overview

CDPruner's Conditional DPP-based visual token pruning algorithm has been successfully integrated into VLMEvalKit. This integration allows you to use CDPruner's token pruning capabilities with LLaVA-1.5-7B through VLMEvalKit's convenient evaluation framework.

## What Was Done

### 1. Local LLaVA Package (`vlmeval/llava_cdpruner/`)

Created a complete local copy of the modified LLaVA model files from CDPruner:

```
vlmeval/llava_cdpruner/
├── __init__.py
├── constants.py
├── mm_utils.py
├── model/
│   ├── __init__.py
│   ├── builder.py                          # Modified: visual_token_num parameter
│   ├── llava_arch.py                       # Core: DPP pruning algorithm
│   ├── utils.py
│   ├── language_model/
│   │   ├── __init__.py
│   │   └── llava_llama.py                  # Modified: visual_token_num storage
│   ├── multimodal_encoder/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── clip_encoder.py                 # Modified: CLIP text tower loading
│   └── multimodal_projector/
│       ├── __init__.py
│       └── builder.py
```

### 2. Model Wrapper (`vlmeval/vlm/llava/llava_cdpruner.py`)

Created `LLaVA_CDPruner` class that:
- Inherits from VLMEvalKit's `BaseModel`
- Accepts `visual_token_num` as a configurable parameter (default: 64)
- Uses the local CDPruner package for model loading
- Passes text prompts to the model for relevance calculation
- Compatible with VLMEvalKit's evaluation pipeline

### 3. Model Registration (`vlmeval/config.py`)

Registered three CDPruner model variants:
- `llava_v1.5_7b_cdpruner` - 64 visual tokens (default)
- `llava_v1.5_7b_cdpruner_32` - 32 visual tokens (more aggressive)
- `llava_v1.5_7b_cdpruner_128` - 128 visual tokens (less aggressive)

### 4. Key Modifications Preserved

All three core CDPruner modifications are intact:

#### a) `llava_llama.py` (Line 44-52)
```python
def __init__(self, config, visual_token_num):
    super(LlamaForCausalLM, self).__init__(config)
    self.model = LlavaLlamaModel(config)
    ...
    self.visual_token_num = visual_token_num  # [CDPruner]
```

#### b) `llava_arch.py` (Lines 140-187)
- Implements Conditional DPP algorithm
- Calculates token similarity and query relevance
- Uses Fast MAP-DPP for token selection
- Returns pruned image features with index masks

#### c) `clip_encoder.py` (Lines 38-96)
- Loads CLIP text tower for text embedding
- Returns tuple: `(image_features, image_embeds, text_embeds)`
- Enables relevance calculation between visual tokens and text query

## Usage

### Basic Usage

```bash
# With 64 tokens (default)
python run.py --data MMBench_DEV_EN --model llava_v1.5_7b_cdpruner

# With 32 tokens (more pruning)
python run.py --data MMBench_DEV_EN --model llava_v1.5_7b_cdpruner_32

# With 128 tokens (less pruning)
python run.py --data MMBench_DEV_EN --model llava_v1.5_7b_cdpruner_128
```

### Programmatic Usage

```python
from vlmeval.vlm import LLaVA_CDPruner

# Initialize with custom token number
model = LLaVA_CDPruner(
    model_path="/path/to/llava-v1.5-7b",
    visual_token_num=96
)

# Use with VLMEvalKit evaluation pipeline
# The model will automatically apply token pruning during inference
```

### Specifying Model Path

You can use either:
1. Hugging Face model ID: `"liuhaotian/llava-v1.5-7b"`
2. Local path: `"/gpfs/scrubbed/jxb1st/model/hub/models--liuhaotian--llava-v1.5-7b/llava-v1.5-7b"`

## Requirements

The CDPruner integration requires the same environment as the original CDPruner project:
- Python 3.10
- PyTorch 2.1.2 (or compatible with your CUDA version)
- Transformers 4.37.2
- All LLaVA and VLMEvalKit dependencies

**Note:** You may need to install additional VLMEvalKit dependencies. If you encounter missing module errors, install them via:
```bash
conda activate your_environment
pip install <missing_module>
```

## Architecture

```
User Input (Text + Image)
    ↓
LLaVA_CDPruner (wrapper)
    ↓
Local LLaVA Package (vlmeval/llava_cdpruner/)
    ↓
┌─────────────────────────────────────────────┐
│ 1. CLIP Vision Encoder → image_features     │
│ 2. CLIP Vision Encoder → image_embeds       │
│ 3. CLIP Text Encoder → text_embeds          │
│ 4. Calculate similarity & relevance         │
│ 5. Construct kernel matrix                  │
│ 6. Fast MAP-DPP selection                   │
│ 7. Apply pruning mask                       │
└─────────────────────────────────────────────┘
    ↓
MM Projector (projects to LLM space)
    ↓
LLaMA-2-7B (language model)
    ↓
Generated Text Output
```

## Verification

Run the verification script to check the integration:

```bash
./test_files_only.sh
```

This verifies:
- All required files are in place
- Key CDPruner modifications are present
- Models are registered in config
- Integration is ready to use

## Differences from Standalone CDPruner

1. **Package Location**: Model files are in `vlmeval/llava_cdpruner/` instead of `llava/`
2. **Import Paths**: All imports updated from `llava.*` to `vlmeval.llava_cdpruner.*`
3. **Model Registration**: Integrated into VLMEvalKit's model registry
4. **Evaluation**: Uses VLMEvalKit's evaluation pipeline instead of custom scripts

## Benefits

1. **Unified Evaluation**: Use VLMEvalKit's comprehensive benchmark evaluation
2. **Easy Comparison**: Compare CDPruner against other VLMs in the same framework
3. **Flexible Configuration**: Easily adjust `visual_token_num` parameter
4. **No Conflicts**: Local package avoids conflicts with other LLaVA installations

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError` for VLMEvalKit dependencies:
```bash
conda activate cdpruner  # or your environment
pip install <missing_module>
```

### Model Loading Errors

Ensure your model path is correct and the model files exist:
```bash
ls -la /path/to/your/llava-v1.5-7b/
```

### CUDA Errors

Make sure your PyTorch installation matches your CUDA version:
```bash
python -c "import torch; print(torch.version.cuda)"
```

## Files Modified/Created

### Created Files:
- `vlmeval/llava_cdpruner/` (entire directory)
- `vlmeval/vlm/llava/llava_cdpruner.py`
- `test_files_only.sh`
- `CDPRUNER_INTEGRATION.md` (this file)

### Modified Files:
- `vlmeval/vlm/__init__.py` (added LLaVA_CDPruner import)
- `vlmeval/vlm/llava/__init__.py` (added LLaVA_CDPruner export)
- `vlmeval/config.py` (added CDPruner model registrations)

## Contact

For issues specific to the CDPruner algorithm, refer to the original CDPruner repository.
For issues with the VLMEvalKit integration, check this integration document and the VLMEvalKit documentation.


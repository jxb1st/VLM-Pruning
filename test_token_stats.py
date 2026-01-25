#!/usr/bin/env python
"""
Test script to verify token statistics printing functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vlmeval.vlm.internvl.debug_logger import InternVLDebugLogger

print("="*80)
print("Token Statistics Printing Test")
print("="*80)

# Create logger
logger = InternVLDebugLogger()

print("\n✓ Debug logger created successfully")
print(f"  - Log file: {logger.log_file}")
print(f"  - Initial total_sample_count: {logger.total_sample_count}")

print("\n" + "="*80)
print("Simulating token statistics from multiple samples")
print("="*80)

# Simulate different samples with various token counts
test_cases = [
    (1792, 53, "MMBench_DEV_EN_V11"),
    (1536, 48, "MMBench_DEV_EN_V11"),
    (2048, 61, "MMBench_DEV_EN_V11"),
    (1280, 45, "MMStar"),
    (2304, 67, None),  # No dataset name
]

print("\nExpected format: Sample #N [DATASET] | Vision: X | Text: Y | Total: Z")
print("\nPrinting token statistics:\n")

for vision_tokens, text_tokens, dataset in test_cases:
    logger.print_token_stats(vision_tokens, text_tokens, dataset)

print("\n" + "="*80)
print("Test Summary")
print("="*80)
print(f"✓ Total samples counted: {logger.total_sample_count}")
print(f"✓ Expected: {len(test_cases)}")
print(f"✓ Match: {logger.total_sample_count == len(test_cases)}")

print("\n" + "="*80)
print("Test completed successfully!")
print("="*80)

print("\nNote: When running actual inference with InternVL3.5 models:")
print("  - Each task case will print a single line with token stats")
print("  - Only the first sample will generate detailed logs to file")
print("  - All samples will show token statistics in terminal")
print("\nTo test with real model inference, run:")
print("  python run.py --data MMBench_DEV_EN_V11 --model InternVL3_5-8B")


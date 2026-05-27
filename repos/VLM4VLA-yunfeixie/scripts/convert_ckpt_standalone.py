"""Standalone DeepSpeed stage-2 -> FP32 single-file converter.

Usage:
    python scripts/convert_ckpt_standalone.py <ds_ckpt_dir> <fp32_out_path>
"""
import os
import sys
from vlm4vla.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

src = sys.argv[1]
dst = sys.argv[2]

assert os.path.isdir(src), f"not a directory: {src}"
os.makedirs(os.path.dirname(dst), exist_ok=True)

print(f"converting {src} -> {dst}")
convert_zero_checkpoint_to_fp32_state_dict(src, dst)
print(f"done, size: {os.path.getsize(dst) / 1e9:.2f} GB")

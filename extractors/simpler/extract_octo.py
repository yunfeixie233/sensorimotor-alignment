"""
Phase 2: Extract features (feats_A) from Octo-base for CKNNA.

Octo uses a BlockTransformer with group-based attention. The extraction
point is `transformer_outputs["readout_action"]` -- DETR-style learned
query tokens (initialized to zeros + positional embedding) that aggregate
information from task and observation tokens via causal cross-attention.
These are the DIRECT input to the ContinuousActionHead.

Architecture:
  Image (256x256) -> SmallStem16 -> obs tokens
  Language text -> T5-base (frozen) -> task tokens
  readout_action -> zeros + pos_embed -> readout tokens
  BlockTransformer(task, obs, readout) -> transformer_outputs
  transformer_outputs["readout_action"] -> ContinuousActionHead -> actions

Extraction point:
  transformer_outputs["readout_action"].tokens
  Shape: (1, 1, n_readout_tokens, 768) -> mean-pool over readout tokens -> (768,)

Usage:
    python extract_features_octo.py \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/octo-base-bridge
"""

import argparse
import json
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OCTO_ROOT = os.environ.get("OCTO_ROOT", os.path.join(_PROJECT_ROOT, "repos", "octo"))
if _OCTO_ROOT not in sys.path:
    sys.path.insert(0, _OCTO_ROOT)

import jax
import numpy as np
import tensorflow as tf
import torch
from octo.model.octo_model import OctoModel


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract Octo features for CKNNA.")
    parser.add_argument("--model_type", type=str, default="hf://rail-berkeley/octo-base")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading Octo: {args.model_type}")
    model = OctoModel.load_pretrained(args.model_type)
    print(f"  Model loaded on {jax.devices()}")

    partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, weights_only=True)
        feats_list = list(partial_tensor[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_list)} loaded)")
    else:
        feats_list = []
        args.resume_from = 0

    print(f"  Samples to process: {num_samples}")

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_png(img_raw, channels=3)
        img = tf.image.resize(img, (256, 256), method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()

        task = model.create_tasks(texts=[task_descriptions[i]])

        observations = {
            "image_primary": img[None, None],  # (1, 1, 256, 256, 3)
            "pad_mask": np.ones((1, 1), dtype=np.float64),
        }

        transformer_outputs = model.run_transformer(
            observations, task, pad_mask=observations["pad_mask"], train=False
        )

        readout_tokens = np.array(transformer_outputs["readout_action"].tokens)
        feat_np = readout_tokens[0, -1].mean(axis=0).astype(np.float32)  # (768,)
        feat = torch.from_numpy(feat_np)
        feats_list.append(feat)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            dim = feat_np.shape[0]
            print(f"  [{i+1}/{num_samples}]  shape=({dim},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    feats_A = torch.stack(feats_list)
    feats_A_path = os.path.join(args.output_dir, "feats_A.pt")
    torch.save(feats_A, feats_A_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": "Octo-base",
        "model_type": args.model_type,
        "extraction_point": 'transformer_outputs["readout_action"].tokens (POST-transformer, DETR-style queries)',
        "pooling": "mean-pool over readout tokens at last window position",
        "hidden_size": int(feats_A.shape[1]),
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Octo Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

"""
Phase 2: Extract features (feats_A) from RT-1-X for CKNNA.

RT-1-X is a TF SavedModel loaded via tf_agents. The only accessible
intermediate representation is `context_image_tokens` from the policy state,
which are post-EfficientNet + post-FiLM(USE language embedding),
pre-Transformer features.

Architecture:
  Image (256x320) -> EfficientNet-B3 + 26 FiLM layers(USE-512D)
    -> context_image_tokens [81, 512]
  Transformer (8 layers, 512D, 8 heads) -> action logits

Extraction point:
  policy_state['context_image_tokens'] at position 0 after one step
  from fresh get_initial_state(). This is POST-EfficientNet+FiLM,
  PRE-Transformer. Shape per sample: (81, 512) -> mean-pool -> (512,)

The post-transformer tensor exists in the graph but is locked inside the
TF2 concrete function and cannot be extracted at runtime.

Each sample is extracted with a fresh policy state (get_initial_state()),
matching the first step of each evaluation episode.

Usage:
    python extract_features_rt1x.py \
        --ckpt checkpoints/rt_1_x_tf_trained_for_002272480_step \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/rt1x-bridge
"""

import argparse
import json
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
import torch


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract RT-1-X features for CKNNA.")
    parser.add_argument("--ckpt", type=str, required=True)
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

    print("Loading RT-1-X policy...")
    tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        model_path=args.ckpt,
        load_specs_from_pbtxt=True,
        use_tf_function=True,
    )

    print("Loading Universal Sentence Encoder...")
    lang_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    observation = tf_agents.specs.zero_spec_nest(
        tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation)
    )
    init_state = tfa_policy.get_initial_state(batch_size=1)
    print("Tracing TF graph (one dummy step)...")
    _ = tfa_policy.action(
        ts.transition(observation, reward=np.zeros((), dtype=np.float32)),
        init_state
    )
    print(f"  Policy loaded. Samples to process: {num_samples}")

    partial_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_tensor = torch.load(partial_path, weights_only=True)
        feats_list = list(partial_tensor[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_list)} loaded)")
    else:
        feats_list = []
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_png(img_raw, channels=3)
        img = tf.image.resize_with_pad(img, target_height=256, target_width=320)
        img = tf.cast(img, tf.uint8)

        task = task_descriptions[i]
        lang_emb = lang_model([task])[0]

        observation["image"] = img
        observation["natural_language_embedding"] = lang_emb

        fresh_state = tfa_policy.get_initial_state(batch_size=1)
        policy_step = tfa_policy.action(
            ts.transition(observation, reward=np.zeros((), dtype=np.float32)),
            fresh_state
        )

        ctx_tokens = np.array(policy_step.state['context_image_tokens'])
        current_tokens = ctx_tokens[0, 0, :, 0, :]  # (81, 512)
        feat = torch.from_numpy(current_tokens.mean(axis=0).astype(np.float32))
        feats_list.append(feat)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  shape=(512,)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_list), partial_path)

    feats_A = torch.stack(feats_list)
    feats_A_path = os.path.join(args.output_dir, "feats_A.pt")
    torch.save(feats_A, feats_A_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": "RT-1-X (TF SavedModel)",
        "checkpoint": args.ckpt,
        "extraction_point": "policy_state['context_image_tokens'][0] (POST-EfficientNet+FiLM, PRE-Transformer)",
        "pooling": "mean-pool over 81 spatial image tokens",
        "hidden_size": 512,
        "feats_A_shape": list(feats_A.shape),
        "num_samples": num_samples,
        "initialization": "get_initial_state() per sample (matches first step of eval episode)",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== RT-1-X Phase 2 Complete ===")
    print(f"  feats_A: {feats_A_path}  shape={tuple(feats_A.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

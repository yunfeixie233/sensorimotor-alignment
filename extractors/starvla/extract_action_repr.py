"""
Phase 2b: Extract action representations (feats_action) from StarVLA models for CKNNA.

Loads a StarVLA checkpoint, runs the action decoder for each image from Phase 1
data, and saves the internal action representation as feats_action.pt.

Three model families require different extraction strategies:
  - QwenFAST (autoregressive): generate() with output_hidden_states, gather gen
    token hidden states, mean-pool across generated action tokens.
  - QwenOFT  (MLP regression): forward pass with action tokens in prompt,
    gather action-token hidden states before MLP head, mean-pool.
  - QwenGR00T / QwenPI (flow-matching): run denoising loop, hook the
    action_decoder input to capture DiT output at the final step, mean-pool
    across action horizon.

Usage:
  python extract_action_repr_starvla.py \
      --ckpt_path playground/Pretrained_models/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt \
      --data_dir ./cknna_data \
      --output_dir ./cknna_data/Qwen-GR00T-Bridge \
      --seed 42
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STARVLA_ROOT = os.environ.get("STARVLA_ROOT", os.path.join(_PROJECT_ROOT, "repos", "starVLA"))
if STARVLA_ROOT not in sys.path:
    sys.path.insert(0, STARVLA_ROOT)

def get_state_for_model(model, feats_B_7d):
    """Return the state array in the format expected by the model's state encoder.

    StarVLA models are trained on 7D state (pad already removed), so we pass
    feats_B directly. If the model has no state encoder, returns None.
    """
    if not hasattr(model, "action_model"):
        return None
    se = getattr(model.action_model, "state_encoder", None)
    if se is None:
        return None
    expected_dim = se.layer1.in_features
    n = feats_B_7d.shape[0]
    if expected_dim == feats_B_7d.shape[1]:
        return feats_B_7d
    # Pad or truncate to match expected dim
    result = np.zeros((n, expected_dim), dtype=np.float32)
    copy_len = min(expected_dim, feats_B_7d.shape[1])
    result[:, :copy_len] = feats_B_7d[:, :copy_len]
    return result


def extract_action_repr_fast(model, images_pil, instruction):
    """QwenFAST: autoregressive generation with hidden state capture.

    Calls generate() with output_hidden_states=True and return_dict_in_generate=True.
    Extracts last-layer hidden states at each generated action token position,
    then mean-pools across all generated tokens.

    Returns (D,) float32 CPU tensor.
    """
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil],
        instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface.model.generate(
            **qwen_inputs,
            max_length=2048,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    # outputs.hidden_states is a tuple:
    #   [0] = prefill step (tuple of num_layers+1 tensors)
    #   [1:] = each generated token step (tuple of num_layers+1 tensors, each (B,1,D))
    num_gen_steps = len(outputs.hidden_states) - 1
    if num_gen_steps == 0:
        hidden_size = model.qwen_vl_interface.model.config.hidden_size
        return torch.zeros(hidden_size, dtype=torch.float32)

    action_hidden = []
    for t in range(1, len(outputs.hidden_states)):
        last_layer = outputs.hidden_states[t][-1]  # (B, 1, D)
        action_hidden.append(last_layer)

    action_repr = torch.cat(action_hidden, dim=1)  # (B, num_gen, D)
    feat_action = action_repr.float().mean(dim=1).squeeze(0).cpu()  # (D,)
    return feat_action


def extract_action_repr_oft(model, images_pil, instruction):
    """QwenOFT: single forward pass with action tokens in prompt.

    Appends magnifier-emoji action tokens to the instruction, runs a VLM
    forward pass, and gathers the hidden states at action token positions.
    These are the representations just before the MLP action head.
    Mean-pools across action token positions.

    Returns (D,) float32 CPU tensor.
    """
    action_tokens = model.action_token * model.chunk_len
    prompt_suffix = f" Please predict the next {model.chunk_len} robot actions: <action>{action_tokens}<action>."
    instruction_with_actions = instruction + prompt_suffix

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil],
        instructions=[instruction_with_actions],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]  # (B, L, D)
    input_ids = qwen_inputs["input_ids"]
    action_queries = model._gather_action_token_embeddings(
        last_hidden, input_ids, action_token_id=model.action_token_id
    )  # (B, chunk_len, D)

    feat_action = action_queries.float().mean(dim=1).squeeze(0).cpu()  # (D,)
    return feat_action


def extract_action_repr_groot(model, images_pil, instruction, state_vec):
    """QwenGR00T: flow-matching with hook on action_decoder input.

    Runs a VLM forward pass for conditioning, then the flow-matching denoising
    loop. Hooks the action_decoder MLP (pre-hook) to capture its input (= DiT
    output) at every denoising step. Takes the final step's representation and
    mean-pools across the action horizon.

    Returns (D,) float32 CPU tensor.
    """
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil],
        instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        qwenvl_outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = qwenvl_outputs.hidden_states[-1]  # (B, L, D)

    action_reprs = []

    def pre_hook_fn(module, inputs):
        action_reprs.append(inputs[0].detach())

    handle = model.action_model.action_decoder.register_forward_pre_hook(pre_hook_fn)

    state_tensor = None
    if state_vec is not None and model.action_model.state_encoder is not None:
        state_tensor = torch.from_numpy(state_vec[np.newaxis, np.newaxis, :]).to(
            device=last_hidden.device, dtype=last_hidden.dtype
        )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
        model.action_model.predict_action(last_hidden, state_tensor)

    handle.remove()

    final_repr = action_reprs[-1]
    action_horizon = model.action_model.action_horizon
    action_part = final_repr[:, -action_horizon:, :]
    feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()
    return feat_action


def extract_action_repr_pi(model, images_pil, instruction, state_vec):
    """QwenPI: layerwise flow-matching with hook on action_decoder input.

    Similar to GR00T but uses layerwise cross-attention DiT. The denoising
    loop feeds different VLM layers to different DiT blocks. Hooks the
    action_decoder pre-hook at the final denoising step.

    Returns (D,) float32 CPU tensor.
    """
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[images_pil],
        instructions=[instruction],
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        qwenvl_outputs = model.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden = qwenvl_outputs.hidden_states
        expected_layers = len(model.action_model.model.transformer_blocks)
        vl_embs_list = list(all_hidden[-expected_layers:])
        base_hidden = vl_embs_list[-1]

    action_reprs = []

    def pre_hook_fn(module, inputs):
        action_reprs.append(inputs[0].detach())

    handle = model.action_model.action_decoder.register_forward_pre_hook(pre_hook_fn)

    state_tensor = None
    if state_vec is not None and model.action_model.state_encoder is not None:
        state_tensor = torch.from_numpy(state_vec[np.newaxis, np.newaxis, :]).to(
            device=base_hidden.device, dtype=base_hidden.dtype
        )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
        model.action_model.predict_action(vl_embs_list, state_tensor)

    handle.remove()

    final_repr = action_reprs[-1]
    action_horizon = model.action_model.action_horizon
    action_part = final_repr[:, -action_horizon:, :]
    feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()
    return feat_action


FRAMEWORK_DISPATCH = {
    "Qwenvl_Fast": "fast",
    "Qwenvl_OFT": "oft",
    "Qwen_GR00T": "groot",
    "Qwen_PI": "pi",
}


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Extract StarVLA action representations.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 output directory (contains images/, metadata.json, feats_B.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_action.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this sample index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_path = os.path.join(args.data_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    feats_B = torch.load(os.path.join(args.data_dir, "feats_B.pt"), weights_only=True)

    print(f"Loading checkpoint: {args.ckpt_path}")
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(args.device).eval()

    framework_class = type(model).__name__
    family = FRAMEWORK_DISPATCH[framework_class]
    print(f"  Framework: {framework_class} -> family: {family}")
    print(f"  Samples to process: {num_samples}")

    state_all = get_state_for_model(model, feats_B.numpy())
    if state_all is not None:
        print(f"  State dim for model: {state_all.shape[1]}")

    partial_path = os.path.join(args.output_dir, "feats_action_partial.pt")
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
        img = Image.open(img_path).convert("RGB")
        instruction = task_descriptions[i]
        state_vec = state_all[i] if state_all is not None else None

        # Fix seed per sample for reproducibility in flow-matching models
        torch.manual_seed(args.seed + i)

        if family == "fast":
            feat = extract_action_repr_fast(model, [img], instruction)
        elif family == "oft":
            feat = extract_action_repr_oft(model, [img], instruction)
        elif family == "groot":
            feat = extract_action_repr_groot(model, [img], instruction, state_vec)
        elif family == "pi":
            feat = extract_action_repr_pi(model, [img], instruction, state_vec)

        feats_list.append(feat)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - args.resume_from) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  feat_action shape={tuple(feat.shape)}  "
                  f"rate={rate:.1f} samples/s  ETA={eta/60:.1f}min")

        if (i + 1) % 500 == 0:
            partial = torch.stack(feats_list)
            torch.save(partial, partial_path)

    feats_action = torch.stack(feats_list)
    feats_action_path = os.path.join(args.output_dir, "feats_action.pt")
    torch.save(feats_action, feats_action_path)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "checkpoint": args.ckpt_path,
        "framework": framework_class,
        "family": family,
        "feats_action_shape": list(feats_action.shape),
        "num_samples": num_samples,
        "data_dir": args.data_dir,
        "seed": args.seed,
    }
    meta_path = os.path.join(args.output_dir, "action_repr_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Phase 2b Complete ===")
    print(f"  feats_action: {feats_action_path}  shape={tuple(feats_action.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

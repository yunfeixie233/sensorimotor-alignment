"""
7-feature extraction for Pi0 (PaliGemma + Gemma Expert, VLM hidden=2048, expert hidden=1024).

Extracts all 7 features + action in a SINGLE select_action() call per sample:
  P1-V   vit_raw:      vision_tower.vision_model.post_layernorm hook -> (1, 256, 1152). Mean-pool dim=1.
  P2-V   pre_llm:      multi_modal_projector hook -> (1, 256, 2048). Mean-pool dim=1.
  P2-T   pre_llm_txt:  paligemma.language_model.layers[0] pre-hook, text-only mask
  P2-VT  pre_llm_vt:   paligemma.language_model.layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:           VLM norm hook, image mask
  P3-T   txt:           VLM norm hook, task mask
  P3-VT  imgtext:       VLM norm hook, prefix_pad_masks
  Action: expert norm hook at final denoising step, last n_action_steps tokens

Usage:
  PYTHONNOUSERSITE=1 python extractors/extract_7feat_pi0.py \
      --ckpt_path HaomingSong/lerobot-pi0-bridge \
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge \
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge_7feat/Pi0 \
      [--max_samples 10]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

torch.backends.cudnn.enabled = False


def masked_mean_pool(hidden_states, mask):
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


def build_task_mask_pi0(lang_tokens_1d, tokenizer, task, n_img, prefix_len, device):
    mask = torch.zeros(prefix_len, dtype=torch.long, device=device)
    if not task:
        return mask
    ids_list = lang_tokens_1d.tolist()
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start = find_subsequence(ids_list, task_ids)
        if start >= 0:
            mask[n_img + start : n_img + start + len(task_ids)] = 1
            return mask
    return mask


def resolve_vlm_norm_module(model):
    candidates = [
        "paligemma_with_expert.paligemma.language_model.norm",
        "paligemma_with_expert.paligemma.language_model.model.norm",
    ]
    for path in candidates:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            print(f"  VLM hook path resolved: {path}")
            return obj
    raise AttributeError(
        f"Cannot resolve VLM norm module. Tried: {candidates}. "
        f"Check the checkpoint's model structure."
    )


def resolve_expert_norm_module(model):
    candidates = [
        "paligemma_with_expert.gemma_expert.model.norm",
    ]
    for path in candidates:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            print(f"  Expert hook path resolved: {path}")
            return obj
    raise AttributeError(
        f"Cannot resolve expert norm module. Tried: {candidates}. "
        f"Check the checkpoint's model structure."
    )


def resolve_module_by_path(model, path):
    """Resolve a dotted module path from model."""
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def reconstruct_8d_state(feats_B_7d):
    STATE_DIM_RAW = 8
    PAD_INDEX = 6
    n = feats_B_7d.shape[0]
    state_8d = np.zeros((n, STATE_DIM_RAW), dtype=np.float32)
    state_8d[:, :PAD_INDEX] = feats_B_7d[:, :PAD_INDEX]
    state_8d[:, PAD_INDEX + 1 :] = feats_B_7d[:, PAD_INDEX:]
    return state_8d


def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for Pi0.")
    parser.add_argument("--ckpt_path", type=str, default="HaomingSong/lerobot-pi0-bridge",
                        help="HuggingFace model ID or local path to Pi0 checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CKNNA data dir (images/, metadata.json, feats_B.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save 7 .pt + feats_action.pt files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    feats_B = torch.load(os.path.join(args.data_dir, "feats_B.pt"), weights_only=True)
    state_8d_all = reconstruct_8d_state(feats_B.numpy())

    # --- Load Pi0 ---
    print(f"Loading Pi0 from: {args.ckpt_path}")

    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_helpers")
    sys.path.insert(0, helpers_dir)
    from lerobot_common_shim import install_shim
    install_shim()

    from huggingface_hub import snapshot_download
    local_ckpt = snapshot_download(args.ckpt_path)

    sys.path.insert(0, local_ckpt)
    from modeling_pi0 import PI0Policy

    if "predict_action_chunk" in getattr(PI0Policy, "__abstractmethods__", set()):
        PI0Policy.predict_action_chunk = lambda self, batch, **kw: None
        PI0Policy.__abstractmethods__ = PI0Policy.__abstractmethods__ - {"predict_action_chunk"}

    policy = PI0Policy.from_pretrained(local_ckpt)
    policy.to(args.device)
    policy.eval()
    model = policy.model

    # --- Resolve hook targets ---
    vlm_norm = resolve_vlm_norm_module(model)
    expert_norm = resolve_expert_norm_module(model)

    vlm_hidden_size = vlm_norm.weight.shape[0]
    expert_hidden_size = expert_norm.weight.shape[0]
    print(f"  VLM hidden_size: {vlm_hidden_size}")
    print(f"  Expert hidden_size: {expert_hidden_size}")

    # P1-V: vision_tower post_layernorm
    vit_post_ln = resolve_module_by_path(
        model, "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm"
    )
    if vit_post_ln is None:
        # Fallback: try without .model
        vit_post_ln = resolve_module_by_path(
            model, "paligemma_with_expert.paligemma.vision_tower.vision_model.post_layernorm"
        )
    assert vit_post_ln is not None, "Cannot find vision_tower post_layernorm"
    print(f"  P1-V: vision_tower.post_layernorm found (dim={vit_post_ln.weight.shape[0]})")

    # P2-V: multi_modal_projector
    projector = resolve_module_by_path(
        model, "paligemma_with_expert.paligemma.model.multi_modal_projector"
    )
    if projector is None:
        projector = resolve_module_by_path(
            model, "paligemma_with_expert.paligemma.multi_modal_projector"
        )
    assert projector is not None, "Cannot find multi_modal_projector"
    print(f"  P2-V: multi_modal_projector found")

    # P2-T/VT: language_model.layers[0] (bare GemmaModel, NOT .model.layers)
    layer0 = resolve_module_by_path(
        model, "paligemma_with_expert.paligemma.language_model.layers.0"
    )
    if layer0 is None:
        layer0 = resolve_module_by_path(
            model, "paligemma_with_expert.paligemma.model.language_model.layers.0"
        )
    assert layer0 is not None, "Cannot find language_model.layers[0]"
    print(f"  P2-T/VT: language_model.layers[0] found")

    n_action_steps = policy.config.n_action_steps
    num_denoise_steps = policy.config.num_steps
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  num_denoise_steps: {num_denoise_steps}")
    print(f"  Samples: {num_samples}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_ckpt)
    tokenizer_max_length = policy.config.tokenizer_max_length
    print(f"  tokenizer_max_length: {tokenizer_max_length}")

    img_keys = list(policy.config.image_features.keys())
    img_key = img_keys[0]
    print(f"  Image key from config: {img_key}")

    OBS_ROBOT = "observation.state"
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
    state_dim = policy.config.robot_state_feature.shape[0]
    print(f"  Expected state_dim: {state_dim}")

    weight_dtype = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
    print(f"  Model dtype: {weight_dtype}")

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}
    accum["action"] = []

    t0 = time.time()
    for i in range(num_samples):
        img_np = np.array(Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB"))
        task = task_descriptions[i]

        image_tensor = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(args.device)

        state_8d = state_8d_all[i]
        state_padded = np.zeros(state_dim, dtype=np.float32)
        copy_len = min(len(state_8d), state_dim)
        state_padded[:copy_len] = state_8d[:copy_len]
        state_tensor = torch.from_numpy(state_padded).unsqueeze(0).to(args.device)

        tokenized = tokenizer(
            [task],
            max_length=tokenizer_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        lang_tokens = tokenized["input_ids"].to(args.device)
        lang_masks = tokenized["attention_mask"].to(dtype=torch.bool, device=args.device)

        batch = {
            img_key: image_tensor,
            OBS_ROBOT: state_tensor,
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
            "task": [task],
        }

        # Pre-compute prefix dimensions for masking
        images, img_masks = policy.prepare_images(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        prefix_len = prefix_pad_masks.shape[1]
        n_img = prefix_len - lang_tokens.shape[1]

        # --- Register hooks ---
        captures = {}
        expert_hook_outputs = []

        def hook_p1(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            captures["p1_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

        def hook_p2(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            captures["p2_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

        # P2-T/VT: Use prefix_embs directly (LLM input embeddings).
        # Pi0's custom forward path doesn't trigger standard layers[0] pre-hooks,
        # but prefix_embs IS the LLM input (verified: different from norm output, cosine=0.07).
        prefix_hidden = prefix_embs.detach().float()
        img_mask_2d = torch.zeros(prefix_hidden.shape[:2], device=prefix_hidden.device, dtype=torch.long)
        img_mask_2d[0, :n_img] = prefix_pad_masks[0, :n_img].long()
        attn_2d = prefix_pad_masks.long()
        text_mask = (attn_2d - img_mask_2d).clamp(min=0)
        captures["p2_vt"] = masked_mean_pool(prefix_hidden, attn_2d).squeeze(0).cpu()
        captures["p2_t"] = masked_mean_pool(prefix_hidden, text_mask).squeeze(0).cpu()

        def vlm_hook_fn(module, inp, out):
            if isinstance(out, tuple):
                captures["vlm_hidden"] = out[0].detach()
            else:
                captures["vlm_hidden"] = out.detach()

        def expert_hook_fn(module, inp, out):
            if isinstance(out, tuple):
                expert_hook_outputs.append(out[0].detach())
            else:
                expert_hook_outputs.append(out.detach())

        h1 = vit_post_ln.register_forward_hook(hook_p1)
        h2 = projector.register_forward_hook(hook_p2)
        h_vlm = vlm_norm.register_forward_hook(vlm_hook_fn)
        h_expert = expert_norm.register_forward_hook(expert_hook_fn)

        torch.manual_seed(args.seed + i)

        with torch.inference_mode():
            policy.select_action(batch)

        h1.remove()
        h2.remove()
        h_vlm.remove()
        h_expert.remove()

        # --- P3 from VLM norm hook ---
        hidden = captures["vlm_hidden"]

        feat_imgtext = masked_mean_pool(hidden, prefix_pad_masks).squeeze(0).cpu()

        img_mask = torch.zeros_like(prefix_pad_masks)
        img_mask[0, :n_img] = prefix_pad_masks[0, :n_img]
        feat_img = masked_mean_pool(hidden, img_mask).squeeze(0).cpu()

        task_mask = build_task_mask_pi0(
            lang_tokens[0], tokenizer, task, n_img, prefix_len, lang_tokens.device
        ).unsqueeze(0)
        feat_txt = masked_mean_pool(hidden, task_mask).squeeze(0).cpu()

        captures["p3_vt"] = feat_imgtext
        captures["p3_v"] = feat_img
        captures["p3_t"] = feat_txt

        # --- Action from expert ---
        final_expert_out = expert_hook_outputs[-1]
        action_part = final_expert_out[:, -n_action_steps:, :]
        feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()

        for k in keys:
            if k not in captures or captures[k] is None:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(captures[k])
        accum["action"].append(feat_action)

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            shapes = "  ".join(f"{k}={tuple(captures[k].shape)}" for k in keys)
            print(
                f"  [{i+1}/{num_samples}]  {shapes}  action={tuple(feat_action.shape)}  "
                f"expert_hooks={len(expert_hook_outputs)}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
                flush=True,
            )

    file_map = {
        "p1_v": "feats_A_vit.pt",
        "p2_v": "feats_A_pre_llm.pt",
        "p2_t": "feats_A_pre_llm_txt.pt",
        "p2_vt": "feats_A_pre_llm_vt.pt",
        "p3_v": "feats_A_img.pt",
        "p3_t": "feats_A_txt.pt",
        "p3_vt": "feats_A.pt",
        "action": "feats_action.pt",
    }

    print(f"\nSaving {len(file_map)} feature tensors:")
    for k in list(keys) + ["action"]:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<28} shape={tuple(tensor.shape)}")

    meta = {
        "model": args.ckpt_path,
        "type": "all_7_features_plus_action",
        "architecture": "Pi0 (PaliGemma + Gemma Expert)",
        "num_samples": num_samples,
        "vlm_hidden_size": vlm_hidden_size,
        "expert_hidden_size": expert_hidden_size,
        "n_action_steps": n_action_steps,
        "num_denoise_steps": num_denoise_steps,
        "seed": args.seed,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in list(keys) + ["action"]},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()

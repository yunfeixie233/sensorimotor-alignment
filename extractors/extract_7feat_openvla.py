"""
7-feature + action extraction for OpenVLA models (base openvla-7b and FT openvla-bridge-sft).

Extracts all 7 features + feats_action in a SINGLE forward/generate call per sample:
  P1-V   vit_raw:       vision_backbone forward hook (DINOv2+SigLIP concat, 2176d)
  P2-V   pre_llm:       projector forward hook (PrismaticProjector 2176->4096)
  P2-T   pre_llm_txt:   language_model.model.layers[0] pre-hook, text-only mask
  P2-VT  pre_llm_vt:    language_model.model.layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:            hidden_states[-1], image mask
  P3-T   txt:            hidden_states[-1], task mask
  P3-VT  imgtext:        hidden_states[-1], attention mask
  action:                action token hidden states (generate for base, hook for FT)

Architecture: Prismatic VLM (DINOv2 + SigLIP vision + Llama-2-7B LM, hidden_size=4096)

FT model workaround: openvla-bridge-sft overrides forward() to return plain logits.
We register hooks on model.language_model to force output_hidden_states=True and capture
hidden_states. Action tokens are appended inside forward(), so we trim prefill and extract
action positions from the extended hidden_states.

Usage:
  PYTHONNOUSERSITE=1 python extractors/extract_7feat_openvla.py \
      --ckpt /path/to/openvla-7b \
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/openvla-7b \
      [--max_samples 10] [--device cuda]

Env: openvla_env with PYTHONNOUSERSITE=1
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

torch.backends.cudnn.enabled = False


def masked_mean_pool(hidden_states, mask):
    """Mean-pool hidden_states over positions indicated by mask."""
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    """Find first occurrence of subseq in seq, return start index or -1."""
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


def build_task_mask_prismatic(input_ids_1d, tokenizer, task, num_patches):
    """Build task mask in multimodal space (offset by num_patches)."""
    multimodal_len = len(input_ids_1d) + num_patches
    device = input_ids_1d.device
    mask = torch.zeros(multimodal_len, dtype=torch.long, device=device)
    if not task:
        return mask
    ids_list = input_ids_1d.tolist()
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start_in_text = find_subsequence(ids_list, task_ids)
        if start_in_text >= 0:
            mm_start = start_in_text + num_patches
            mask[mm_start : mm_start + len(task_ids)] = 1
            return mask
    return mask


def main():
    parser = argparse.ArgumentParser(description="7-feature + action extraction for OpenVLA.")
    parser.add_argument("--ckpt", type=str, required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True, help="CKNNA data dir (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save .pt files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load metadata ---
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    # --- Load model ---
    print(f"Loading OpenVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    tokenizer = processor.tokenizer

    # --- Vision backbone info ---
    vb = model.vision_backbone
    num_patches = vb.featurizer.patch_embed.num_patches  # 256
    vit_dim = vb.featurizer.embed_dim + vb.fused_featurizer.embed_dim  # 1024 + 1152 = 2176

    print(f"  hidden_size: {hidden_size}")
    print(f"  num_patches: {num_patches}")
    print(f"  vit_concat_dim: {vit_dim}")
    print(f"  Samples to process: {num_samples}")

    # --- Detect FT model ---
    is_ft_model = "bridge-sft" in args.ckpt or "ft-200k" in args.ckpt
    num_action_tokens = None

    if is_ft_model:
        print("  Detected fine-tuned model -- using hook-based forward (not generate())")
        # Load action constants from checkpoint
        _spec = importlib.util.spec_from_file_location(
            "_openvla_sft_constants",
            os.path.join(args.ckpt, "constants.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        action_dim = _mod.ACTION_DIM
        num_actions_chunk = _mod.NUM_ACTIONS_CHUNK
        num_action_tokens = action_dim * num_actions_chunk
        print(f"  ACTION_DIM={action_dim}  NUM_ACTIONS_CHUNK={num_actions_chunk}  "
              f"action_tokens={num_action_tokens}")

    # --- Set up persistent hooks for P1-V and P2-V ---
    captures = {}

    # P1-V: vision_backbone forward hook -- concat of DINOv2 + SigLIP, (1, 256, 2176)
    def hook_vb(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        # Mean-pool over patch dim
        captures["p1_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    # P2-V: projector forward hook -- (1, 256, 4096)
    def hook_proj(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["p2_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    # P2-T and P2-VT: LLM layer 0 pre-hook
    # We store these in a mutable dict so the hook can access per-sample masks
    hook_ctx = {}

    def hook_layer0_pre(module, args):
        # Guard: only capture on the FIRST call (prefill pass).
        # During generate(), this hook fires once per autoregressive step;
        # without this guard the last (single-token) call overwrites the
        # correct prefill capture.
        if hook_ctx.get("captured"):
            return
        hook_ctx["captured"] = True

        t = args[0]  # (1, seq_len, 4096) — may include appended action tokens for FT models
        hidden = t.detach().float()
        mm_attn = hook_ctx["multimodal_mask"]
        img_mask = hook_ctx["image_mask"]
        mm_len = mm_attn.shape[1]
        # Trim to multimodal_len (FT model appends action tokens, making hidden longer)
        hidden_prefill = hidden[:, :mm_len, :]
        # P2-VT: mean-pool all prefill positions
        captures["p2_vt"] = masked_mean_pool(hidden_prefill, mm_attn).squeeze(0).cpu()
        # P2-T: text-only positions (attention - image)
        text_mask = (mm_attn - img_mask).clamp(min=0)
        captures["p2_t"] = masked_mean_pool(hidden_prefill, text_mask).squeeze(0).cpu()

    h_vb = model.vision_backbone.register_forward_hook(hook_vb)
    h_proj = model.projector.register_forward_hook(hook_proj)
    h_l0 = model.language_model.model.layers[0].register_forward_pre_hook(hook_layer0_pre)

    # --- FT model: hooks to inject output_hidden_states into language_model ---
    _hook_output = {}
    if is_ft_model:
        def _lm_pre_hook(_module, _args, kwargs):
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            return _args, kwargs

        def _lm_post_hook(_module, _input, output):
            _hook_output["hidden_states"] = output.hidden_states

        h_lm_pre = model.language_model.register_forward_pre_hook(_lm_pre_hook, with_kwargs=True)
        h_lm_post = model.language_model.register_forward_hook(_lm_post_hook)

    # --- Feature keys and accumulation ---
    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt", "action"]
    accum = {k: [] for k in keys}

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]
        prompt = f"In: What action should the robot take to {task}?\nOut:"

        inputs = processor(prompt, img)
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16, device=args.device)

        text_seq_len = input_ids.shape[1]
        multimodal_len = 1 + num_patches + (text_seq_len - 1)

        # Build multimodal attention mask: [BOS] [patches] [text tokens after BOS]
        patch_mask = torch.ones(
            (1, num_patches), dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )

        # Image mask: positional, patches at positions 1:1+num_patches
        image_mask = torch.zeros_like(multimodal_mask)
        image_mask[0, 1 : 1 + num_patches] = 1

        # Task mask in multimodal space
        task_mask = build_task_mask_prismatic(
            input_ids[0], tokenizer, task, num_patches
        ).unsqueeze(0)

        # Store masks for layer0 pre-hook
        hook_ctx["multimodal_mask"] = multimodal_mask
        hook_ctx["image_mask"] = image_mask
        hook_ctx["captured"] = False  # reset guard so hook fires on next prefill

        # Clear captures
        captures.clear()
        _hook_output.clear()

        if is_ft_model:
            # --- FT path: single forward(), hooks capture hidden_states ---
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )

            last_hidden = _hook_output["hidden_states"][-1]
            # FT model appends action_tokens + stop_token inside forward()
            # Shape: (1, multimodal_len + num_action_tokens + 1, hidden_size)
            prefill_hidden = last_hidden[:, :multimodal_len, :]

            # P3 features from prefill
            captures["p3_vt"] = masked_mean_pool(prefill_hidden, multimodal_mask).squeeze(0).cpu()
            captures["p3_v"] = masked_mean_pool(prefill_hidden, image_mask).squeeze(0).cpu()
            captures["p3_t"] = masked_mean_pool(prefill_hidden, task_mask).squeeze(0).cpu()

            # Action features
            action_start = multimodal_len
            action_end = action_start + num_action_tokens
            action_hidden = last_hidden[:, action_start:action_end, :]
            if action_hidden.shape[1] == 0:
                captures["action"] = torch.zeros(hidden_size, dtype=torch.float32)
            else:
                captures["action"] = action_hidden.float().mean(dim=1).squeeze(0).cpu()

        else:
            # --- Base path: forward() for P1-V through P3-VT ---
            # Using forward() instead of generate() is simpler, faster, and
            # avoids the hook-overwriting bug (generate() calls layers[0]
            # once per autoregressive step, clobbering the prefill capture).
            # P1-V through P3-VT are identical between forward() and generate()
            # because the prefill pass in generate() IS a forward() call.
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # Last-layer hidden states (prefill only, no autoregressive tokens)
            prefill_hidden = outputs.hidden_states[-1][:, :multimodal_len, :]

            # P3 features
            captures["p3_vt"] = masked_mean_pool(prefill_hidden, multimodal_mask).squeeze(0).cpu()
            captures["p3_v"] = masked_mean_pool(prefill_hidden, image_mask).squeeze(0).cpu()
            captures["p3_t"] = masked_mean_pool(prefill_hidden, task_mask).squeeze(0).cpu()

            # Action features: not available from forward() on base model.
            # Use zero placeholder — action features require generate().
            captures["action"] = torch.zeros(hidden_size, dtype=torch.float32)

        # Accumulate
        for k in keys:
            if k not in captures or captures[k] is None:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(captures[k])

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            shapes = "  ".join(f"{k}={tuple(captures[k].shape)}" for k in keys)
            print(
                f"  [{i+1}/{num_samples}]  {shapes}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
                flush=True,
            )

    # --- Remove hooks ---
    h_vb.remove()
    h_proj.remove()
    h_l0.remove()
    if is_ft_model:
        h_lm_pre.remove()
        h_lm_post.remove()

    # --- Save ---
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

    print(f"\nSaving {len(keys)} feature tensors:")
    for k in keys:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<28} shape={tuple(tensor.shape)}")

    # --- Metadata ---
    meta = {
        "model": args.ckpt,
        "type": "all_7_features_plus_action",
        "is_ft_model": is_ft_model,
        "num_samples": num_samples,
        "hidden_size": hidden_size,
        "vit_concat_dim": vit_dim,
        "num_patches": num_patches,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    if is_ft_model and num_action_tokens is not None:
        meta["num_action_tokens"] = num_action_tokens
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def masked_mean_pool(hidden_states, attention_mask):
    h = hidden_states.float()
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
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
            mask[n_img + start:n_img + start + len(task_ids)] = 1
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


def reconstruct_8d_state(feats_B_7d):
    STATE_DIM_RAW = 8
    PAD_INDEX = 6
    n = feats_B_7d.shape[0]
    state_8d = np.zeros((n, STATE_DIM_RAW), dtype=np.float32)
    state_8d[:, :PAD_INDEX] = feats_B_7d[:, :PAD_INDEX]
    state_8d[:, PAD_INDEX + 1:] = feats_B_7d[:, PAD_INDEX:]
    return state_8d


def main():
    parser = argparse.ArgumentParser(description="Unified extraction: feats_A + feats_action from Pi0.")
    parser.add_argument("--ckpt_path", type=str, default="HaomingSong/lerobot-pi0-bridge",
                        help="HuggingFace model ID or local path to Pi0 checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json, feats_B.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt, feats_A_img.pt, feats_A_txt.pt, feats_action.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    feats_B = torch.load(os.path.join(args.data_dir, "feats_B.pt"), weights_only=True)
    state_8d_all = reconstruct_8d_state(feats_B.numpy())

    print(f"Loading Pi0 from: {args.ckpt_path}")

    helpers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_helpers")
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

    vlm_norm = resolve_vlm_norm_module(model)
    expert_norm = resolve_expert_norm_module(model)

    vlm_hidden_size = vlm_norm.weight.shape[0]
    expert_hidden_size = expert_norm.weight.shape[0]
    print(f"  VLM hidden_size: {vlm_hidden_size}")
    print(f"  Expert hidden_size: {expert_hidden_size}")

    n_action_steps = policy.config.n_action_steps
    num_denoise_steps = policy.config.num_steps
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  num_denoise_steps: {num_denoise_steps}")
    print(f"  Samples to process: {num_samples}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_ckpt)
    tokenizer_max_length = policy.config.tokenizer_max_length
    print(f"  tokenizer_max_length: {tokenizer_max_length}")

    vlm_hook_output = {}
    expert_hook_outputs = []

    def vlm_hook_fn(module, inp, out):
        if isinstance(out, tuple):
            vlm_hook_output["feat"] = out[0].detach()
        else:
            vlm_hook_output["feat"] = out.detach()

    def expert_hook_fn(module, inp, out):
        if isinstance(out, tuple):
            expert_hook_outputs.append(out[0].detach())
        else:
            expert_hook_outputs.append(out.detach())

    vlm_handle = vlm_norm.register_forward_hook(vlm_hook_fn)
    expert_handle = expert_norm.register_forward_hook(expert_hook_fn)

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

    partial_path = os.path.join(args.output_dir, "feats_partial.pt")
    if args.resume_from > 0 and os.path.exists(partial_path):
        partial_data = torch.load(partial_path, weights_only=True)
        feats_imgtext_list = list(partial_data["feats_A"][:args.resume_from])
        feats_img_list = list(partial_data["feats_A_img"][:args.resume_from])
        feats_txt_list = list(partial_data["feats_A_txt"][:args.resume_from])
        feats_action_list = list(partial_data["feats_action"][:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_imgtext_list)} loaded)")
    else:
        feats_imgtext_list = []
        feats_img_list = []
        feats_txt_list = []
        feats_action_list = []
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
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

        images, img_masks = policy.prepare_images(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )

        vlm_hook_output.clear()
        expert_hook_outputs.clear()

        torch.manual_seed(args.seed + i)

        with torch.inference_mode():
            policy.select_action(batch)

        hidden = vlm_hook_output["feat"]
        prefix_len = prefix_pad_masks.shape[1]
        n_img = prefix_len - lang_tokens.shape[1]

        feat_imgtext = masked_mean_pool(hidden, prefix_pad_masks).squeeze(0).cpu()

        img_mask = torch.zeros_like(prefix_pad_masks)
        img_mask[0, :n_img] = prefix_pad_masks[0, :n_img]
        feat_img = masked_mean_pool(hidden, img_mask).squeeze(0).cpu()

        task_mask = build_task_mask_pi0(
            lang_tokens[0], tokenizer, task, n_img, prefix_len, lang_tokens.device
        ).unsqueeze(0)
        feat_txt = masked_mean_pool(hidden, task_mask).squeeze(0).cpu()

        final_expert_out = expert_hook_outputs[-1]
        action_part = final_expert_out[:, -n_action_steps:, :]
        feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()

        feats_imgtext_list.append(feat_imgtext)
        feats_img_list.append(feat_img)
        feats_txt_list.append(feat_txt)
        feats_action_list.append(feat_action)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  vlm_shape=({vlm_hidden_size},)  expert_shape={tuple(feat_action.shape)}  "
                  f"expert_hooks={len(expert_hook_outputs)}  rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

        if (i + 1) % args.save_every == 0:
            partial_data = {
                "feats_A": torch.stack(feats_imgtext_list),
                "feats_A_img": torch.stack(feats_img_list),
                "feats_A_txt": torch.stack(feats_txt_list),
                "feats_action": torch.stack(feats_action_list),
            }
            torch.save(partial_data, partial_path)

    vlm_handle.remove()
    expert_handle.remove()

    feats_A = torch.stack(feats_imgtext_list)
    feats_A_img = torch.stack(feats_img_list)
    feats_A_txt = torch.stack(feats_txt_list)
    feats_action = torch.stack(feats_action_list)

    for suffix, tensor in [("feats_A", feats_A),
                          ("feats_A_img", feats_A_img),
                          ("feats_A_txt", feats_A_txt),
                          ("feats_action", feats_action)]:
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(tensor, p)
        print(f"  Saved {p}  shape={tuple(tensor.shape)}")

    if os.path.exists(partial_path):
        os.remove(partial_path)

    extraction_meta = {
        "model": args.ckpt_path,
        "vlm_extraction_point": "hook on language_model.norm (post-norm GemmaModel)",
        "expert_extraction_point": "hook on gemma_expert.model.norm at final denoising step",
        "vlm_hidden_size": vlm_hidden_size,
        "expert_hidden_size": expert_hidden_size,
        "n_action_steps": n_action_steps,
        "num_denoise_steps": num_denoise_steps,
        "num_samples": num_samples,
        "seed": args.seed,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt", "feats_action.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Unified Pi0 Extraction Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

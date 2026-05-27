"""
7-feature extraction for SpatialVLA (PaliGemma2: SigLIP + Gemma2-4B, hidden=2304).

Extracts all 7 features in a SINGLE forward pass per sample:
  P1-V   vit_raw:      vision_tower hook -> last_hidden_state (1, 256, 1152). Mean-pool dim=1.
  P2-V   pre_llm:      multi_modal_projector hook -> (1, 256, 2304). Mean-pool dim=1.
  P2-T   pre_llm_txt:  language_model.model.layers[0] pre-hook, text-only mask
  P2-VT  pre_llm_vt:   language_model.model.layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:           hidden_states[-1], image mask
  P3-T   txt:           hidden_states[-1], task mask
  P3-VT  imgtext:       hidden_states[-1], attention mask

Bonus: if model.position_embedding_3d exists, also saves feats_A_depth.pt.

Usage:
  python extractors/extract_7feat_spatialvla.py \
      --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge \
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge_7feat/SpatialVLA \
      [--max_samples 10]
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

torch.backends.cudnn.enabled = False

IMAGE_TOKEN_INDEX = 257152


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


def build_task_mask(input_ids_1d, tokenizer, task):
    mask = torch.zeros(len(input_ids_1d), dtype=torch.long, device=input_ids_1d.device)
    if not task:
        return mask
    ids_list = input_ids_1d.tolist()
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start = find_subsequence(ids_list, task_ids)
        if start >= 0:
            mask[start : start + len(task_ids)] = 1
            return mask
    # Fallback: character-level search
    full_text = tokenizer.decode(ids_list, skip_special_tokens=True)
    char_pos = full_text.lower().find(task.lower().strip())
    if char_pos < 0:
        return mask
    task_end_char = char_pos + len(task.strip())
    start_tok, end_tok = None, None
    for k in range(len(ids_list)):
        prefix_len = len(tokenizer.decode(ids_list[: k + 1], skip_special_tokens=True))
        if start_tok is None and prefix_len > char_pos:
            start_tok = k
        if prefix_len >= task_end_char:
            end_tok = k + 1
            break
    if start_tok is not None and end_tok is not None:
        mask[start_tok:end_tok] = 1
    return mask


def extract_all_7(model, inputs_on_device, tokenizer, instruction, device):
    """Extract all 7 features in one forward pass. Returns dict of (D,) tensors."""
    captures = {}

    # --- P1-V: vision_tower output ---
    def hook_p1(module, input, output):
        # output is BaseModelOutput or tuple
        if hasattr(output, "last_hidden_state"):
            t = output.last_hidden_state
        elif isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        # (1, 256, 1152) -> mean-pool dim=1 -> (1152,)
        captures["p1_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    # --- P2-V: multi_modal_projector output ---
    def hook_p2(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        # (1, 256, 2304) -> mean-pool dim=1 -> (2304,)
        captures["p2_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    # --- P2-T and P2-VT: LLM layers[0] pre-hook ---
    def hook_layer0_pre(module, args):
        t = args[0]  # (1, seq_len, 2304)
        hidden = t.detach().float()
        # P2-VT: mean-pool all positions
        captures["p2_vt"] = hidden.mean(dim=1).squeeze(0).cpu()
        # P2-T: text-only positions
        input_ids = inputs_on_device["input_ids"]
        image_mask = (input_ids == IMAGE_TOKEN_INDEX).long()
        attn_mask = inputs_on_device.get(
            "attention_mask",
            torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long),
        )
        text_mask = (attn_mask - image_mask).clamp(min=0)
        captures["p2_t"] = masked_mean_pool(hidden, text_mask).squeeze(0).cpu()

    # --- Bonus: depth embedding ---
    has_depth = hasattr(model, "position_embedding_3d") and model.position_embedding_3d is not None

    def hook_depth(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["depth"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    h1 = model.vision_tower.register_forward_hook(hook_p1)
    h2 = model.multi_modal_projector.register_forward_hook(hook_p2)
    h3 = model.language_model.model.layers[0].register_forward_pre_hook(hook_layer0_pre)
    h_depth = model.position_embedding_3d.register_forward_hook(hook_depth) if has_depth else None

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            **inputs_on_device,
            output_hidden_states=True,
            return_dict=True,
        )

    h1.remove()
    h2.remove()
    h3.remove()
    if h_depth is not None:
        h_depth.remove()

    # --- P3 features from final hidden state ---
    last_hidden = outputs.hidden_states[-1]
    input_ids = inputs_on_device["input_ids"]
    attn_mask = inputs_on_device.get(
        "attention_mask",
        torch.ones(last_hidden.shape[:2], device=last_hidden.device, dtype=torch.long),
    )
    image_mask = (input_ids == IMAGE_TOKEN_INDEX).long()

    captures["p3_vt"] = masked_mean_pool(last_hidden, attn_mask).squeeze(0).cpu()
    captures["p3_v"] = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    task_mask = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
    captures["p3_t"] = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

    return captures


def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for SpatialVLA.")
    parser.add_argument("--ckpt", type=str,
                        default="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CKNNA data dir (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save 7 .pt files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading SpatialVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(args.device)

    tokenizer = processor.tokenizer
    hidden_size = model.config.text_config.hidden_size
    vit_dim = model.vision_tower.config.hidden_size if hasattr(model.vision_tower, "config") else 1152
    has_depth = hasattr(model, "position_embedding_3d") and model.position_embedding_3d is not None

    print(f"  LLM hidden: {hidden_size}, ViT dim: {vit_dim}")
    print(f"  position_embedding_3d present: {has_depth}")
    print(f"  Samples: {num_samples}")

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}
    if has_depth:
        accum["depth"] = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]

        inputs = processor(
            text=instruction,
            images=img,
            return_tensors="pt",
        )
        inputs_on_device = {k: v.to(args.device) for k, v in inputs.items()}

        feats = extract_all_7(model, inputs_on_device, tokenizer, instruction, args.device)

        for k in keys:
            if k not in feats or feats[k] is None:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(feats[k])

        if has_depth and "depth" in feats:
            accum["depth"].append(feats["depth"])

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            shapes = "  ".join(f"{k}={tuple(feats[k].shape)}" for k in keys)
            print(
                f"  [{i+1}/{num_samples}]  {shapes}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
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
    }

    print(f"\nSaving {len(keys)} feature tensors:")
    for k in keys:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<28} shape={tuple(tensor.shape)}")

    if has_depth and len(accum["depth"]) > 0:
        depth_tensor = torch.stack(accum["depth"])
        depth_path = os.path.join(args.output_dir, "feats_A_depth.pt")
        torch.save(depth_tensor, depth_path)
        print(f"  {'feats_A_depth.pt':<28} shape={tuple(depth_tensor.shape)}")
        file_map["depth"] = "feats_A_depth.pt"

    meta = {
        "model": args.ckpt,
        "type": "all_7_features",
        "architecture": "SpatialVLA (PaliGemma2: SigLIP + Gemma2)",
        "num_samples": num_samples,
        "hidden_size": hidden_size,
        "vit_dim": vit_dim,
        "has_depth_embedding": has_depth,
        "image_token_index": IMAGE_TOKEN_INDEX,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()

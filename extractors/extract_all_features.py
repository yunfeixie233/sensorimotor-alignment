"""
Unified 7-feature extraction for raw VLMs (Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, PaliGemma, KosMos-2).

Extracts all 7 features in a SINGLE forward pass per sample:

  Position × Modality matrix:
  ─────────────────────────────────────────────────────────────────────
  ID     Position                Modality     File saved
  ─────────────────────────────────────────────────────────────────────
  P1-V   Pre-projection          Vision       feats_A_vit.pt
  P2-V   Post-projection pre-LLM Vision       feats_A_pre_llm.pt
  P2-T   Post-projection pre-LLM Text         feats_A_pre_llm_txt.pt
  P2-VT  Post-projection pre-LLM Vision+Text  feats_A_pre_llm_vt.pt
  P3-V   Post-LLM                Vision       feats_A_img.pt
  P3-T   Post-LLM                Text         feats_A_txt.pt
  P3-VT  Post-LLM                Vision+Text  feats_A.pt
  ─────────────────────────────────────────────────────────────────────

Hook strategy:
  P1-V:  forward_hook on merger.ln_q/norm (Qwen), post_layernorm (PG/K2)
  P2-V:  forward_hook on merger (Qwen), projector (PG), resampler (K2)
  P2-T:  forward_pre_hook on layers[0] → mean-pool over the task-instruction tokens
         (chat-template scaffolding excluded via build_task_mask)
  P2-VT: forward_pre_hook on layers[0] → mean-pool over (image ∪ instruction) positions
  P3-*:  output_hidden_states[-1] + masks (same instruction-only convention as P2-T / P2-VT)

Usage:
  python extractors/extract_all_features.py \\
      --model_path Qwen/Qwen2.5-VL-3B-Instruct \\
      --model_family qwen2.5 \\
      --data_dir /path/to/cknna_data \\
      --output_dir /path/to/output \\
      [--max_samples 10]  # for smoke testing
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Compat patch: transformers 5.x PaliGemma causal mask needs torch>=2.6.
# We strip the or_mask_function kwarg so extraction works on torch 2.5.
# This drops PaliGemma's bidirectional image-token attention, but for
# feature extraction (not generation) the effect is negligible.
# ---------------------------------------------------------------------------
if torch.__version__ < "2.6":
    try:
        import transformers.masking_utils as _mu
        _orig_causal = _mu.create_causal_mask

        def _patched_causal(*args, **kwargs):
            kwargs.pop("or_mask_function", None)
            kwargs.pop("and_mask_function", None)
            return _orig_causal(*args, **kwargs)

        _mu.create_causal_mask = _patched_causal
    except Exception:
        pass  # older transformers without masking_utils


# Disable cuDNN — not needed for transformer inference (matmul/softmax only),
# and avoids CUDNN_STATUS_NOT_INITIALIZED on some driver/CUDA combos.
torch.backends.cudnn.enabled = False

IMAGE_TOKEN_INDEX = 151655  # Qwen VL image pad token


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def masked_mean_pool(hidden_states, mask):
    """Mean-pool hidden_states over positions where mask==1."""
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def find_subsequence(seq, subseq):
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


def build_task_mask(input_ids_1d, tokenizer, task):
    """Build mask for task-instruction tokens (same logic as extract_raw_qwen.py)."""
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


# ---------------------------------------------------------------------------
# Model loading (per family)
# ---------------------------------------------------------------------------
def load_qwen25(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def load_qwen3(model_path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def load_qwen3_moe(model_path):
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # Constrain GPU to 36GB so some layers spill to CPU
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "36GiB", "cpu": "80GiB"},
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        offload_folder="/tmp/offload_qwen3_moe",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def load_paligemma(model_path):
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
        attn_implementation="eager",  # sdpa/default needs torch>=2.6 for PaliGemma causal mask
    )
    processor = AutoProcessor.from_pretrained(
        model_path, token=os.environ.get("HF_TOKEN")
    )
    return model, processor


def load_kosmos2(model_path):
    from transformers import Kosmos2ForConditionalGeneration, Kosmos2Processor
    model = Kosmos2ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    processor = Kosmos2Processor.from_pretrained(model_path)
    return model, processor


# ---------------------------------------------------------------------------
# Input building (per family)
# ---------------------------------------------------------------------------
def build_inputs_qwen25(processor, img, instruction):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Qwen3-VL/qwen-vl-utils/src/"))
    from qwen_vl_utils import process_vision_info
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": instruction},
    ]}]]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=texts, images=image_inputs, videos=video_inputs,
                     padding=True, return_tensors="pt")


def build_inputs_qwen3(processor, img, instruction):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": instruction},
    ]}]
    return processor.apply_chat_template(
        [messages], tokenize=True, padding=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )


def build_inputs_paligemma(processor, img, instruction):
    return processor(text=instruction, images=img, return_tensors="pt", padding=True)


def build_inputs_kosmos2(processor, img, instruction):
    return processor(text=f"<grounding>{instruction}", images=img, return_tensors="pt")


# ---------------------------------------------------------------------------
# Hook setup (per family)
# ---------------------------------------------------------------------------
def get_hook_targets(model, family):
    """Return (p1_module, p2_module, layer0_module, image_token_id, get_tokenizer_fn, get_image_mask_fn)."""

    if family in ("qwen2.5", "qwen3", "qwen3_moe"):
        # P1: merger LN (final ViT norm before spatial merge)
        if hasattr(model.model.visual.merger, "ln_q"):
            p1_mod = model.model.visual.merger.ln_q       # Qwen2.5
        else:
            p1_mod = model.model.visual.merger.norm        # Qwen3 / Qwen3-MoE
        p2_mod = model.model.visual.merger                 # P2: full merger output
        layer0 = model.model.language_model.layers[0]      # LLM layer 0
        img_token_id = IMAGE_TOKEN_INDEX

        def get_image_mask(batch):
            return (batch["input_ids"] == img_token_id).long()

        return p1_mod, p2_mod, layer0, img_token_id, get_image_mask

    elif family == "paligemma":
        p1_mod = model.model.vision_tower.vision_model.post_layernorm
        p2_mod = model.model.multi_modal_projector
        layer0 = model.model.language_model.layers[0]       # PaliGemmaModel → GemmaModel → layers
        img_token_id = model.config.image_token_index        # 256000

        def get_image_mask(batch):
            return (batch["input_ids"] == img_token_id).long()

        return p1_mod, p2_mod, layer0, img_token_id, get_image_mask

    elif family == "kosmos2":
        p1_mod = model.vision_model.model.post_layernorm
        p2_mod = model.image_to_text_projection
        layer0 = model.text_model.model.layers[0]          # Kosmos2TextTransformer → layers
        img_token_id = None  # KosMos-2 uses image_embeds_position_mask

        def get_image_mask(batch):
            return batch["image_embeds_position_mask"].long()

        return p1_mod, p2_mod, layer0, img_token_id, get_image_mask

    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Core extraction: all 7 features in one forward pass
# ---------------------------------------------------------------------------
def extract_all_7(model, batch, tokenizer, instruction, device, family,
                  p1_mod, p2_mod, layer0, get_image_mask):
    """Extract all 7 features from a single forward pass.

    Returns dict with keys: p1_v, p2_v, p2_t, p2_vt, p3_v, p3_t, p3_vt.

    Text-bearing positions (p2_t / p2_vt / p3_t / p3_vt) mean-pool over the
    task-instruction tokens (P2-T / P3-T) or over image ∪ instruction positions
    (P2-VT / P3-VT), with chat-template scaffolding excluded via build_task_mask.
    """
    captures = {}

    # --- P1-V hook: raw ViT output ---
    def hook_p1(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        if family == "kosmos2":
            # KosMos-2 post_layernorm fires twice: skip 2D CLS-pooled call
            if t.dim() != 3:
                return
            # Skip CLS at position 0
            patches = t[:, 1:, :].detach().float()
            captures["p1_v"] = patches.mean(dim=1).squeeze(0).cpu()
        elif family == "paligemma":
            # (1, 256, 1152) — mean over patches
            captures["p1_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()
        else:
            # Qwen: (N_patches, D_vit) packed NaViT, no batch dim
            captures["p1_v"] = t.detach().float().mean(dim=0).cpu()

    # --- P2-V hook: merger/projector output ---
    def hook_p2(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        if family in ("qwen2.5", "qwen3", "qwen3_moe"):
            # (N_merged, D_llm) — no batch dim
            captures["p2_v"] = t.detach().float().mean(dim=0).cpu()
        else:
            # PaliGemma/KosMos-2: (1, N_tokens, D_llm)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            captures["p2_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

    # --- P2-VT and P2-T: pre-hook on LLM layer 0 ---
    def hook_layer0_input(module, args):
        t = args[0]  # (1, seq_len, D_llm) — full LLM input with vision tokens substituted
        hidden = t.detach().float()

        image_mask = get_image_mask(batch_on_device)
        input_ids = batch_on_device["input_ids"]

        # P2-T: instruction-only mean-pool. build_task_mask locates the task-instruction
        # tokens (chat-template scaffolding excluded). Works for kosmos2 too — its tokenizer
        # encodes the raw instruction cleanly inside `<grounding>{instruction}<eos>`.
        task_mask = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
        captures["p2_t"] = masked_mean_pool(hidden, task_mask).squeeze(0).cpu()

        # P2-VT: image positions ∪ instruction positions, scaffolding excluded.
        # OR of image_mask and task_mask (both are 0/1 masks; .clamp(max=1) prevents
        # double-counting if they ever overlap, which they don't on real prompts).
        vt_mask = (image_mask + task_mask).clamp(max=1)
        captures["p2_vt"] = masked_mean_pool(hidden, vt_mask).squeeze(0).cpu()

        captures["_n_p2_t"] = int(task_mask.sum().item())
        captures["_n_p2_vt"] = int(vt_mask.sum().item())

    # Register hooks
    h1 = p1_mod.register_forward_hook(hook_p1)
    h2 = p2_mod.register_forward_hook(hook_p2)
    h3 = layer0.register_forward_pre_hook(hook_layer0_input)

    # Forward pass
    batch_on_device = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if family == "kosmos2":
            outputs = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                pixel_values=batch_on_device["pixel_values"],
                image_embeds_position_mask=batch_on_device["image_embeds_position_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = model(
                **batch_on_device,
                output_hidden_states=True,
                return_dict=True,
            )

    # Remove hooks
    h1.remove()
    h2.remove()
    h3.remove()

    # --- P3 features from LLM output ---
    last_hidden = outputs.hidden_states[-1]
    image_mask = get_image_mask(batch_on_device)
    input_ids = batch_on_device["input_ids"]

    # P3-V: image tokens only (no scaffolding to exclude — the image-token positions
    # never contain scaffolding by construction).
    captures["p3_v"] = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    # P3-T: instruction-only mean-pool via build_task_mask (works for all families,
    # including kosmos2 — excludes <s><grounding> framing markers).
    task_mask_p3 = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
    captures["p3_t"] = masked_mean_pool(last_hidden, task_mask_p3).squeeze(0).cpu()

    # P3-VT: image positions ∪ instruction positions, scaffolding excluded.
    vt_mask_p3 = (image_mask + task_mask_p3).clamp(max=1)
    captures["p3_vt"] = masked_mean_pool(last_hidden, vt_mask_p3).squeeze(0).cpu()

    captures["_n_p3_t"] = int(task_mask_p3.sum().item())
    captures["_n_p3_vt"] = int(vt_mask_p3.sum().item())

    return captures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract all 7 CKNNA features in one forward pass.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_family", required=True,
                        choices=["qwen2.5", "qwen3", "qwen3_moe", "paligemma", "kosmos2"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples (for smoke testing); interpreted as a count starting "
                             "from --sample_start, NOT as a global end index")
    parser.add_argument("--sample_start", type=int, default=0,
                        help="Starting sample index (for sample-range sharding across machines). "
                             "The output tensors will have shape (count, D) where count = "
                             "min(total, sample_start + max_samples) - sample_start.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    total_samples = metadata["num_samples"]
    sample_start = args.sample_start
    sample_end = total_samples
    if args.max_samples:
        sample_end = min(sample_end, sample_start + args.max_samples)
    if sample_start < 0 or sample_start >= total_samples:
        raise ValueError(f"--sample_start={sample_start} out of range [0, {total_samples})")
    if sample_end <= sample_start:
        raise ValueError(f"sample_end={sample_end} <= sample_start={sample_start}")
    num_samples = sample_end - sample_start
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    # Load model
    print(f"Loading {args.model_family}: {args.model_path}")
    loaders = {
        "qwen2.5": (load_qwen25, build_inputs_qwen25),
        "qwen3": (load_qwen3, build_inputs_qwen3),
        "qwen3_moe": (load_qwen3_moe, build_inputs_qwen3),  # same input format as dense qwen3
        "paligemma": (load_paligemma, build_inputs_paligemma),
        "kosmos2": (load_kosmos2, build_inputs_kosmos2),
    }
    load_fn, build_fn = loaders[args.model_family]
    model, processor = load_fn(args.model_path)
    # Quantized models (qwen3_moe) use device_map="auto", skip .to(device)
    if args.model_family == "qwen3_moe":
        model = model.eval()
    else:
        model = model.to(args.device).eval()

    # Get tokenizer (needed for build_task_mask in P2-T / P3-T / P2-VT / P3-VT extraction
    # across all families, including KosMos-2).
    tokenizer = processor.tokenizer

    # Get hook targets
    p1_mod, p2_mod, layer0, img_token_id, get_image_mask = get_hook_targets(
        model, args.model_family
    )

    # Probe dims
    print(f"  Family: {args.model_family}")
    print(f"  Samples: {num_samples} (global indices {sample_start}..{sample_end-1} of {total_samples})")

    # Accumulators for the 7 features. Rows of accum[k] correspond to GLOBAL sample
    # indices in order [sample_start, sample_start+1, ..., sample_end-1]. Merging
    # shards is just torch.cat(...) along dim 0 in shard order.
    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}
    mask_size_log = {"p2_t": [], "p2_vt": [], "p3_t": [], "p3_vt": []}

    t0 = time.time()
    for i_global in range(sample_start, sample_end):
        img = Image.open(os.path.join(images_dir, f"{i_global:06d}.png")).convert("RGB")
        instruction = task_descriptions[i_global]
        batch = build_fn(processor, img, instruction)

        feats = extract_all_7(
            model, batch, tokenizer, instruction, args.device, args.model_family,
            p1_mod, p2_mod, layer0, get_image_mask,
        )

        for k in keys:
            if k not in feats or feats[k] is None:
                raise RuntimeError(f"Feature {k} missing at sample {i_global}")
            accum[k].append(feats[k])

        # Track per-sample mask sizes for diagnostics. Map "_n_<name>" → "<name>".
        for tag in ("_n_p2_t", "_n_p2_vt", "_n_p3_t", "_n_p3_vt"):
            if tag in feats:
                key = tag[3:]  # strip leading "_n_"
                mask_size_log[key].append(feats[tag])

        local_i = i_global - sample_start
        if (local_i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (local_i + 1) % 100 == 0 or local_i == 0:
            elapsed = time.time() - t0
            rate = (local_i + 1) / elapsed
            eta = (num_samples - local_i - 1) / rate
            shapes = "  ".join(f"{k}={tuple(feats[k].shape)}" for k in keys)
            n_p2_t = mask_size_log["p2_t"][-1] if mask_size_log["p2_t"] else -1
            print(f"  [{local_i+1}/{num_samples} g={i_global}]  {shapes}  "
                  f"n_p2_t={n_p2_t}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
                  flush=True)

    # Save the 7 feature tensors (canonical filenames matching the VLA-SR extractors).
    file_map = {
        "p1_v":  "feats_A_vit.pt",
        "p2_v":  "feats_A_pre_llm.pt",
        "p2_t":  "feats_A_pre_llm_txt.pt",
        "p2_vt": "feats_A_pre_llm_vt.pt",
        "p3_v":  "feats_A_img.pt",
        "p3_t":  "feats_A_txt.pt",
        "p3_vt": "feats_A.pt",
    }

    print(f"\nSaving {len(keys)} feature tensors:")
    for k in keys:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<28} shape={tuple(tensor.shape)}")

    # Save metadata with mask-size stats for every "instr" variant
    sample_feats = {k: accum[k][0] for k in keys}
    def _stats(name):
        xs = mask_size_log.get(name, [])
        if not xs: return None
        return {
            "mean_tokens_pooled": sum(xs)/len(xs),
            "min": min(xs),
            "max": max(xs),
            "n_empty_masks": sum(1 for x in xs if x == 0),
        }
    meta = {
        "model": args.model_path,
        "model_family": args.model_family,
        "type": "all_7_features",
        "num_samples": num_samples,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "total_samples_in_dataset": total_samples,
        "feature_dims": {k: sample_feats[k].shape[-1] for k in keys},
        "file_map": file_map,
        "mask_stats": {
            "p2_t":  _stats("p2_t"),
            "p2_vt": _stats("p2_vt"),
            "p3_t":  _stats("p3_t"),
            "p3_vt": _stats("p3_vt"),
        },
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()

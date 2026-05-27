"""
7-feature extraction for ABot-M0 (Qwen3-VL-4B-Instruct-Action backbone).

Follows Mode A convention — identical forward path as
extract_7feat_starvla.py / extract_7feat_xvla.py: direct call to
`vlm_interface(**qwen_inputs, output_hidden_states=True)` with hooks
attached to the VLM internals. Skips downstream action decoder and VGGT
fusion (both happen AFTER our hooks fire, so the extracted tensors are
bit-exact equivalent to running full predict_action()).

Extracts in a SINGLE forward pass per sample:
  P1-V   vit_raw:      merger.norm hook output
  P2-V   pre_llm:      merger hook output (post linear_fc2)
  P2-T   pre_llm_txt:  layers[0] pre-hook, text-only mask
  P2-VT  pre_llm_vt:   layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:          hidden_states[-1], image mask
  P3-T   txt:          hidden_states[-1], task mask
  P3-VT  imgtext:      hidden_states[-1], attention mask

Compliance notes (2026-04-11):
 - Qwen3-VL-4B base weights are NOT downloaded locally; we init the
   VLM with accelerate.init_empty_weights() then load weights from the
   ABot checkpoint via load_state_dict(strict=False, assign=True).
 - Vocab extended to 153984 to match ABot's -Action variant.
 - cfg.framework.use_vggt = False (VGGT fuses hidden_states[-1] AFTER
   all 7 hooks fire -> feature-neutral).
 - torch.backends.cudnn.enabled = False (avoids CUDNN_STATUS_NOT_INITIALIZED
   on visual conv3d in Qwen3-VL; matches existing extractor convention).

Usage:
  ABOT_ROOT=/lambda/nfs/vla/cknna_project/repos/ABot-Manipulation \
  VGGT_ROOT=/lambda/nfs/vla/cknna_project/repos/vggt \
  python extractors/extract_7feat_abot_m0.py \
      --ckpt_path /lambda/nfs/vla/cknna_project/checkpoints/abot-m0-robotwin2/checkpoints/steps_125000_pytorch_model.pt \
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/abot-m0-robotwin2 \
      [--max_samples 10]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

torch.backends.cudnn.enabled = False

# ---------------------------------------------------------------------------
# Path setup: ABot + vggt (vggt only needed because ABot_M0.py does a
# top-level `from vggt.models.vggt import VGGT`; we set use_vggt=False in
# the config so the actual VGGT module is never instantiated).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ABOT_ROOT = os.environ.get(
    "ABOT_ROOT", os.path.join(_PROJECT_ROOT, "repos", "ABot-Manipulation")
)
VGGT_ROOT = os.environ.get(
    "VGGT_ROOT", os.path.join(_PROJECT_ROOT, "repos", "vggt")
)
for p in (ABOT_ROOT, VGGT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Local copy of Qwen3-VL-4B-Instruct (JSONs only — weights come from ABot ckpt)
QWEN3VL_4B_LOCAL = "/lambda/nfs/vla/pretrained_models/Qwen3-VL-4B-Instruct"

# ---------------------------------------------------------------------------
# Monkey-patch Qwen3VLForConditionalGeneration.from_pretrained (called by
# ABot's QWen3.py line 58) to init an empty model from config instead of
# trying to download/load base weights from the author's original path.
# Without init_empty_weights() this hangs for minutes doing CPU kaiming init
# on ~4B fp32 parameters.
# ---------------------------------------------------------------------------
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
    AutoProcessor,
)
from accelerate import init_empty_weights


@classmethod
def _patched_from_pretrained(cls, model_id, **kwargs):
    print(f"[patched] init_empty_weights() from local Qwen3-VL-4B config", flush=True)
    cfg = AutoConfig.from_pretrained(QWEN3VL_4B_LOCAL)
    # ABot's -Action variant appends 2048 action tokens to the vocab:
    # 151936 -> 153984. We must match this BEFORE load_state_dict.
    cfg.text_config.vocab_size = 153984
    cfg.vocab_size = 153984
    with init_empty_weights():
        model = cls._from_config(cfg)
    return model


Qwen3VLForConditionalGeneration.from_pretrained = _patched_from_pretrained

_orig_auto_proc = AutoProcessor.from_pretrained


def _patched_auto_proc(model_id, **kwargs):
    return _orig_auto_proc(QWEN3VL_4B_LOCAL, **kwargs)


AutoProcessor.from_pretrained = _patched_auto_proc

# ---------------------------------------------------------------------------
# Reuse the hook/pool logic verbatim from extract_7feat_starvla.py.
# ---------------------------------------------------------------------------
IMAGE_TOKEN_INDEX = 151655  # verified from Qwen3-VL-4B-Instruct config.json


def masked_mean_pool(hidden_states, mask):
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


def extract_all_7(vlm_model, qwen_inputs, tokenizer, instruction, device, get_image_mask):
    """Extract all 7 features in one forward pass. Returns dict of (D,) tensors.

    vlm_model here is the underlying Qwen3VLForConditionalGeneration
    (i.e. model.qwen_vl_interface.model). Identical hook positions to
    extract_7feat_starvla.py — only difference is ABot's merger uses the
    `norm` naming (vs older `ln_q`).
    """
    captures = {}

    merger = vlm_model.model.visual.merger
    if hasattr(merger, "ln_q"):
        ln_module = merger.ln_q
    else:
        ln_module = merger.norm  # Qwen3-VL-4B: merger children = [norm, linear_fc1, act_fn, linear_fc2]

    layer0 = vlm_model.model.language_model.layers[0]

    def hook_p1(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["p1_v"] = t.detach().float().mean(dim=0).cpu()

    def hook_p2(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        captures["p2_v"] = t.detach().float().mean(dim=0).cpu()

    def hook_layer0_pre(module, args):
        t = args[0]
        hidden = t.detach().float()
        captures["p2_vt"] = hidden.mean(dim=1).squeeze(0).cpu()
        image_mask = get_image_mask(batch_on_device)
        attn_mask = batch_on_device.get(
            "attention_mask",
            torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long),
        )
        text_mask = (attn_mask - image_mask).clamp(min=0)
        captures["p2_t"] = masked_mean_pool(hidden, text_mask).squeeze(0).cpu()

    h1 = ln_module.register_forward_hook(hook_p1)
    h2 = merger.register_forward_hook(hook_p2)
    h3 = layer0.register_forward_pre_hook(hook_layer0_pre)

    batch_on_device = {k: v.to(device) for k, v in qwen_inputs.items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = vlm_model(
            **batch_on_device,
            output_hidden_states=True,
            return_dict=True,
        )

    h1.remove()
    h2.remove()
    h3.remove()

    last_hidden = outputs.hidden_states[-1]
    attn_mask = batch_on_device.get(
        "attention_mask",
        torch.ones(last_hidden.shape[:2], device=last_hidden.device, dtype=torch.long),
    )
    image_mask = get_image_mask(batch_on_device)

    captures["p3_vt"] = masked_mean_pool(last_hidden, attn_mask).squeeze(0).cpu()
    captures["p3_v"] = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    input_ids = batch_on_device["input_ids"]
    task_mask = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
    captures["p3_t"] = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

    return captures


def load_abot_model(ckpt_path, device):
    """Build ABot_M0 framework from ckpt, load weights with assign=True.

    Replicates baseframework.from_pretrained() but (a) sets use_vggt=False
    BEFORE instantiation and (b) uses strict=False + assign=True to materialize
    meta tensors from the ckpt and drop the 1809 unused VGGT keys.
    """
    from ABot.model.framework.share_tools import read_mode_config, dict_to_namespace
    from ABot.model.framework import build_framework

    print(f"Reading ABot config from: {ckpt_path}")
    model_cfg, norm_stats = read_mode_config(Path(ckpt_path))
    cfg = dict_to_namespace(model_cfg)
    cfg.framework.use_vggt = False  # feature-neutral; VGGT fires AFTER hooks
    cfg.trainer.pretrained_checkpoint = None

    print("Building framework with init_empty_weights...")
    t0 = time.time()
    m = build_framework(cfg=cfg)
    print(f"  built in {time.time()-t0:.1f}s")

    print("Loading state_dict (assign=True, strict=False)...")
    t0 = time.time()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    del sd
    print(
        f"  loaded in {time.time()-t0:.1f}s  missing={len(missing)}  unexpected={len(unexpected)}"
    )
    # Sanity: we expect 0 missing (everything materialized) and all unexpected
    # keys to be spatial_model.* / spatial_projector.* / fuser.* (VGGT branch).
    for name, p in m.named_parameters():
        if p.is_meta:
            raise RuntimeError(f"Parameter {name} is still on meta device after load")

    m.norm_stats = norm_stats
    m = m.to(device).eval()
    return m


def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for ABot-M0.")
    parser.add_argument("--ckpt_path", required=True, help="ABot .pt checkpoint")
    parser.add_argument("--data_dir", required=True, help="CKNNA data dir")
    parser.add_argument("--output_dir", required=True, help="Where to save 7 .pt files")
    parser.add_argument("--device", default="cuda")
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

    print(f"Loading ABot-M0 from: {args.ckpt_path}")
    model = load_abot_model(args.ckpt_path, args.device)

    vlm_model = model.qwen_vl_interface.model
    processor = model.qwen_vl_interface.processor
    tokenizer = processor.tokenizer
    framework_class = type(model).__name__

    vlm_hidden_size = vlm_model.config.text_config.hidden_size
    merger = vlm_model.model.visual.merger
    if hasattr(merger, "ln_q"):
        vit_dim = merger.ln_q.weight.shape[0]
    else:
        vit_dim = merger.norm.weight.shape[0]
    n_llm_layers = len(vlm_model.model.language_model.layers)

    print(f"  Framework: {framework_class}")
    print(
        f"  VLM hidden: {vlm_hidden_size}, ViT dim: {vit_dim}, "
        f"LLM layers: {n_llm_layers}, IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}"
    )
    print(f"  Samples to extract: {num_samples}")

    def get_image_mask(batch):
        return (batch["input_ids"] == IMAGE_TOKEN_INDEX).long()

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]

        # Exact same input builder as official ABot predict_action()
        qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=[[img]],
            instructions=[instruction],
        )

        feats = extract_all_7(
            vlm_model, qwen_inputs, tokenizer, instruction, args.device, get_image_mask
        )

        for k in keys:
            if k not in feats or feats[k] is None:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(feats[k])

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

    meta = {
        "model": args.ckpt_path,
        "family": "ABot-M0 (Qwen3-VL-4B-Instruct-Action + DiT-B action decoder)",
        "framework": framework_class,
        "type": "all_7_features",
        "num_samples": num_samples,
        "vlm_hidden": vlm_hidden_size,
        "vit_dim": vit_dim,
        "n_llm_layers": n_llm_layers,
        "image_token_index": IMAGE_TOKEN_INDEX,
        "mode": "A (direct vlm_interface call, use_vggt=False)",
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()

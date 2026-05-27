"""
7-feature extraction for finetuned StarVLA models (Qwen2.5-VL / Qwen3-VL backbone).

Extracts all 7 features in a SINGLE forward pass per sample:
  P1-V   vit_raw:      merger.ln_q/norm hook (packed NaViT, no batch dim)
  P2-V   pre_llm:      merger forward hook (packed NaViT)
  P2-T   pre_llm_txt:  layers[0] pre-hook, text-only mask
  P2-VT  pre_llm_vt:   layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:           hidden_states[-1], image mask
  P3-T   txt:           hidden_states[-1], task mask
  P3-VT  imgtext:       hidden_states[-1], attention mask

Usage:
  STARVLA_ROOT=/home/ubuntu/vla/cknna_project/repos/starVLA \
  python extractors/extract_7feat_starvla.py \
      --ckpt_path /path/to/steps_XXXXX_pytorch_model.pt \
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/Qwen-GR00T-Bridge \
      [--max_samples 10]
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

torch.backends.cudnn.enabled = False

# ---------------------------------------------------------------------------
# Compat patch: transformers 5.x removed top-level hidden_size from
# Qwen2_5_VLConfig. StarVLA's QwenOFT.__init__ accesses config.hidden_size
# which crashes. Add a property that delegates to text_config.hidden_size.
# ---------------------------------------------------------------------------
try:
    from transformers import Qwen2_5_VLConfig as _Q25Cfg
    if not hasattr(_Q25Cfg, "hidden_size"):
        _Q25Cfg.hidden_size = property(lambda self: self.text_config.hidden_size)
    # transformers >=5.3: use_return_dict is a read-only property on
    # PreTrainedConfig, but from_pretrained tries to set it from config.json
    # kwargs via __init__ -> setattr -> object.__setattr__ which hits the
    # descriptor. Patch __setattr__ to silently skip read-only properties.
    from transformers.configuration_utils import PreTrainedConfig as _PC
    _original_setattr = _PC.__setattr__
    def _safe_setattr(self, key, value):
        try:
            _original_setattr(self, key, value)
        except AttributeError:
            # Silently skip read-only properties (e.g. use_return_dict)
            pass
    _PC.__setattr__ = _safe_setattr
except ImportError:
    pass

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STARVLA_ROOT = os.environ.get("STARVLA_ROOT", os.path.join(_PROJECT_ROOT, "repos", "starVLA"))
if STARVLA_ROOT not in sys.path:
    sys.path.insert(0, STARVLA_ROOT)

IMAGE_TOKEN_INDEX = 151655


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
    """Extract all 7 features in one forward pass. Returns dict of (D,) tensors."""
    captures = {}

    merger = vlm_model.model.visual.merger
    if hasattr(merger, "ln_q"):
        ln_module = merger.ln_q
    else:
        ln_module = merger.norm

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


def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for StarVLA.")
    parser.add_argument("--ckpt_path", required=True, help="StarVLA .pt checkpoint")
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

    print(f"Loading StarVLA: {args.ckpt_path}")
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(args.device).eval()

    vlm_model = model.qwen_vl_interface.model
    processor = model.qwen_vl_interface.processor
    tokenizer = processor.tokenizer
    framework_class = type(model).__name__

    vlm_hidden_size = vlm_model.config.text_config.hidden_size
    merger = vlm_model.model.visual.merger
    vit_dim = merger.ln_q.weight.shape[0] if hasattr(merger, "ln_q") else merger.norm.weight.shape[0]

    print(f"  Framework: {framework_class}")
    print(f"  VLM hidden: {vlm_hidden_size}, ViT dim: {vit_dim}")
    print(f"  Samples: {num_samples}")

    def get_image_mask(batch):
        return (batch["input_ids"] == IMAGE_TOKEN_INDEX).long()

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]

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
        "framework": framework_class,
        "type": "all_7_features",
        "num_samples": num_samples,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()

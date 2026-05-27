"""
7-feature extraction for VLM4VLA-format Qwen2.5-VL-3B + FCDecoder action-head
checkpoints (https://github.com/yunfeixie233/VLM4VLA fork).

The VLM4VLA training stack saves a Lightning + DeepSpeed Stage-2 model whose
state-dict keys are prefixed `model.backbone.<qwen-key>` (the inner
Qwen2_5_VLForConditionalGeneration) plus `model.act_head.*` and
`model.action_token` (FCDecoder + per-stream action token). The action head
operates *after* the LLM forward and cannot influence the 7 SiMA feature
positions, so we strip the action-head weights, load the upstream
Qwen2_5_VLForConditionalGeneration directly, and run the same hook layout used
by `extract_7feat_starvla.py` (which extracts byte-identical features for the
existing Qwen2.5-VL VLA cohort: Qwen-{GR00T,OFT}-Bridge*).

Image preprocessing uses `qwen_vl_utils.process_vision_info` smart-resize
(min_pixels=28*28*256, max_pixels=1280*28*28) for cohort consistency with the
12 existing finetuned VLA rows on the SiMA-vs-SR scatter — NOT VLM4VLA's
training-time 224x224 BICUBIC path.

Extracts all 7 features in a SINGLE forward pass per sample:
  P1-V   vit_raw:      merger.ln_q hook
  P2-V   pre_llm:      merger forward hook
  P2-T   pre_llm_txt:  layers[0] pre-hook, text-only mask (legacy: includes scaffolding)
  P2-VT  pre_llm_vt:   layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:          hidden_states[-1], image mask
  P3-T   txt:          hidden_states[-1], task mask (build_task_mask, instruction-only)
  P3-VT  imgtext:      hidden_states[-1], attention mask

Usage:
  /lambda/nfs/vla/conda/envs/starVLA/bin/python \
    extractors/extract_7feat_vlm4vla.py \
        --ckpt_path /lambda/nfs/vla/cache/vlm4vla_ckpts/run_a_step10k/stepstep=0010000.fp32.pt \
        --base_vlm_path /lambda/nfs/vla/cache/vlm4vla_ckpts/run_a_step10k/qwen_base \
        --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \
        --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/vlm4vla-qwen25vl3b-bridge-step10k \
        [--max_samples 10]
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

torch.backends.cudnn.enabled = False

# Compat patch for transformers 5.x: same as extract_7feat_starvla.py
try:
    from transformers import Qwen2_5_VLConfig as _Q25Cfg
    if not hasattr(_Q25Cfg, "hidden_size"):
        _Q25Cfg.hidden_size = property(lambda self: self.text_config.hidden_size)
    from transformers.configuration_utils import PreTrainedConfig as _PC
    _original_setattr = _PC.__setattr__
    def _safe_setattr(self, key, value):
        try:
            _original_setattr(self, key, value)
        except AttributeError:
            pass
    _PC.__setattr__ = _safe_setattr
except ImportError:
    pass


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


def load_vlm4vla_qwen(ckpt_path, base_vlm_path, device):
    """Load Qwen2_5_VLForConditionalGeneration from VLM4VLA-format FP32 .pt.

    The FP32 .pt file (from `vlm4vla.utils.zero_to_fp32`) flattens DeepSpeed
    Lightning checkpoint state into a dict whose state-dict tensor keys are
    prefixed `model.backbone.<qwen-key>` (Qwen2.5-VL params) plus
    `model.act_head.*` and `model.action_token`. We strip the action-head
    keys, drop `model.backbone.` prefix, and call `load_state_dict` on a
    fresh Qwen2_5_VLForConditionalGeneration.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    print(f"  Loading base Qwen2.5-VL from: {base_vlm_path}")
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_vlm_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    print(f"  Loading FP32 ckpt: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)

    DROP_PREFIXES = (
        "model.act_head.",
        "model.action_token",
        "model.clip_norm_head.",
        "model.fwd_head.",
        "model.embed_arm_state.",
        "model.embed_gripper_state.",
        "model.embed_state.",
        "model.static_image_tokens",
        "model.hand_image_tokens",
    )

    qwen_sd = {}
    skipped = 0
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if not k.startswith("model."):
            continue
        if any(k.startswith(p) for p in DROP_PREFIXES):
            skipped += 1
            continue
        new_k = k[len("model."):]
        if new_k.startswith("backbone."):
            new_k = new_k[len("backbone."):]
        qwen_sd[new_k] = v

    print(f"  state_dict: {len(qwen_sd)} qwen keys (skipped {skipped} action-head/etc)")

    msg = vlm_model.load_state_dict(qwen_sd, strict=False)
    print(f"  load_state_dict: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.unexpected_keys:
        print(f"    unexpected (first 10): {msg.unexpected_keys[:10]}")
    if msg.missing_keys:
        # tied lm_head is OK; flag anything else
        unusual = [k for k in msg.missing_keys if "lm_head" not in k]
        if unusual:
            print(f"    missing (first 10): {unusual[:10]}")
    return vlm_model


def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for VLM4VLA Qwen2.5-VL.")
    parser.add_argument("--ckpt_path", required=True, help="VLM4VLA FP32 .pt checkpoint")
    parser.add_argument("--base_vlm_path", required=True,
                        help="Qwen2.5-VL-3B-Instruct base dir (for architecture skeleton + processor)")
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

    vlm_model = load_vlm4vla_qwen(args.ckpt_path, args.base_vlm_path, args.device)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Use base AutoProcessor defaults (min_pixels=4*28*28=3136, max_pixels=16384*28*28),
    # matching StarVLA's `AutoProcessor.from_pretrained(model_id)` exactly. This is
    # required for cohort consistency: every existing Qwen2.5-VL row in the scatter was
    # extracted under those defaults, which leave small DROID images (320x180) at their
    # native pixel count rather than upscaling. Forcing min_pixels=28*28*256 would
    # quadruple the image-token count and shift all 7 SiMA features.
    processor = AutoProcessor.from_pretrained(args.base_vlm_path)
    tokenizer = processor.tokenizer

    vlm_hidden_size = vlm_model.config.text_config.hidden_size
    merger = vlm_model.model.visual.merger
    vit_dim = merger.ln_q.weight.shape[0] if hasattr(merger, "ln_q") else merger.norm.weight.shape[0]

    print(f"  VLM hidden: {vlm_hidden_size}, ViT dim: {vit_dim}")
    print(f"  Samples: {num_samples}")

    def build_inputs(img, instruction):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(msgs)
        return processor(
            text=[text], images=image_inputs, videos=None,
            padding=True, return_tensors="pt",
        )

    def get_image_mask(batch):
        return (batch["input_ids"] == IMAGE_TOKEN_INDEX).long()

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]

        qwen_inputs = build_inputs(img, instruction)

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
        "framework": "VLM4VLA_RoboQwen25VL",
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

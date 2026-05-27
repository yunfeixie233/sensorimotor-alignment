"""
Phase 2: Extract VLM features (feats_A) from CogACT for CKNNA.

CogACT shares the same Prismatic VLM backbone as OpenVLA:
  DINOv2 + SigLIP fused vision + Llama-2-7b LM, hidden_size=4096

Extraction point: hidden_states[-1] (POST-norm, after final Llama-2 RMSNorm)
Pooling: masked mean-pool over the full multimodal sequence
         (vision patches + text tokens, including the cognition token)

Prompt format: same as CogACT predict_action():
  prompt_builder.add_turn("human", "What action should the robot take to {task}?")
  + appended tokens [29871, 2] (empty-space token + cognition/EOS token)

The multimodal sequence is: [BOS, vision_patches, text_tokens_after_BOS]
where vision patches are always valid (mask=1). The multimodal_attention_mask
is reconstructed identically to extract_features_openvla.py.

State: NOT in feats_A. CogACT's proprioceptive state is never fed into
the VLM; the DiT action model receives state implicitly via the cognition
feature only. This matches all other models in the CKNNA comparison.

Requires: cogact conda env

Usage:
    python extract_features_cogact.py \
        --ckpt CogACT/CogACT-Base \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/cogact-base-bridge
"""

import argparse
import json
import os
import time

import sys
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_COGACT_ROOT = os.environ.get("COGACT_ROOT", os.path.join(_PROJECT_ROOT, "repos", "CogACT"))
if _COGACT_ROOT not in sys.path:
    sys.path.insert(0, _COGACT_ROOT)

from vla import load_vla


def _redirect_llama_to_local_cache():
    """Redirect meta-llama/Llama-2-7b-hf to a pre-populated local cache directory.

    Prismatic's load() calls AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    which fails without gated-model access. This patches from_pretrained to use
    a local cache created by scripts/setup_prismatic_cache.sh.
    """
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    llama_cache = os.environ.get(
        "PRISMATIC_LLAMA_CACHE",
        os.path.expanduser("~/.cache/huggingface/llama2-7b-config"),
    )
    if not os.path.isdir(llama_cache):
        return
    if not os.path.isfile(os.path.join(llama_cache, "config.json")):
        return

    _LLAMA_HF_PATHS = {"meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"}

    _orig_config_fp = AutoConfig.from_pretrained.__func__
    @classmethod
    def _patched_config_fp(cls, pretrained_model_name_or_path, *a, **kw):
        if pretrained_model_name_or_path in _LLAMA_HF_PATHS:
            pretrained_model_name_or_path = llama_cache
        return _orig_config_fp(cls, pretrained_model_name_or_path, *a, **kw)
    AutoConfig.from_pretrained = _patched_config_fp

    _orig_tok_fp = AutoTokenizer.from_pretrained.__func__
    @classmethod
    def _patched_tok_fp(cls, pretrained_model_name_or_path, *a, **kw):
        if pretrained_model_name_or_path in _LLAMA_HF_PATHS:
            pretrained_model_name_or_path = llama_cache
        return _orig_tok_fp(cls, pretrained_model_name_or_path, *a, **kw)
    AutoTokenizer.from_pretrained = _patched_tok_fp

    print(f"[cogact] Redirected meta-llama/Llama-2-7b-hf -> {llama_cache}")


_redirect_llama_to_local_cache()


def masked_mean_pool(hidden_states, attention_mask):
    """Mean-pool hidden states over valid (non-padding) tokens."""
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


def build_task_mask_prismatic(input_ids_1d, tokenizer, task, num_patches):
    """Build task-instruction-only mask in multimodal sequence space.

    Tries bare encoding first (handles SentencePiece/BPE after non-space chars),
    then falls back to space-prefixed encoding.
    """
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
            mask[mm_start:mm_start + len(task_ids)] = 1
            return mask
    return mask


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract CogACT features for CKNNA.")
    parser.add_argument("--ckpt", type=str, default="CogACT/CogACT-Base",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--action_model_type", type=str, default="DiT-B",
                        help="DiT variant: DiT-Small, DiT-B (Base), DiT-L (Large)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading CogACT: {args.ckpt} (action_model_type={args.action_model_type})")
    vla = load_vla(
        args.ckpt,
        load_for_training=False,
        action_model_type=args.action_model_type,
    )
    vla.vlm = vla.vlm.to(torch.bfloat16)
    vla = vla.to(args.device).eval()

    vlm = vla.vlm
    tokenizer = vlm.llm_backbone.tokenizer
    image_transform = vlm.vision_backbone.image_transform
    hidden_size = vlm.llm_backbone.llm.config.hidden_size

    print(f"  hidden_size: {hidden_size}")
    print(f"  tokenizer type: {type(tokenizer).__name__}")
    print(f"  Samples to process: {num_samples}")

    autocast_dtype = vlm.llm_backbone.half_precision_dtype

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {task.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(args.device)
        assert isinstance(tokenizer, LlamaTokenizerFast), f"Expected LlamaTokenizerFast, got {type(tokenizer)}"
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871, 2]], dtype=torch.long, device=args.device)), dim=1
        )
        attention_mask = torch.ones_like(input_ids)

        pixel_values = image_transform(img)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(args.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(args.device) for k, v in pixel_values.items()}

        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            output = vlm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden = output.hidden_states[-1]

        text_seq_len = input_ids.shape[1]
        multimodal_seq_len = last_hidden.shape[1]
        num_patches = multimodal_seq_len - text_seq_len

        patch_mask = torch.ones(
            (1, num_patches), dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )

        feat_imgtext = masked_mean_pool(last_hidden, multimodal_mask).squeeze(0).cpu()

        image_mask = torch.zeros_like(multimodal_mask)
        image_mask[0, 1:1 + num_patches] = 1
        feat_img = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

        task_mask = build_task_mask_prismatic(input_ids[0], tokenizer, task.lower(), num_patches).unsqueeze(0)
        feat_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

        feats_imgtext_list.append(feat_imgtext)
        feats_img_list.append(feat_img)
        feats_txt_list.append(feat_txt)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  shape=({hidden_size},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

    for suffix, flist in [("feats_A", feats_imgtext_list),
                          ("feats_A_img", feats_img_list),
                          ("feats_A_txt", feats_txt_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    extraction_meta = {
        "model": args.ckpt,
        "action_model_type": args.action_model_type,
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== CogACT Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

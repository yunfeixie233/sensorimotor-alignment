"""
Extract VLM features from the raw (pretrained, no robot fine-tuning) Prismatic VLM:
  prism-dinosiglip-224px+7b  (DINOv2 + SigLIP fused vision + Llama-2-7B LM)

This is the base VLM that both OpenVLA and CogACT were fine-tuned from.
Training data: LLaVA instruction tuning only (llava-lvis4v-lrv). Zero robot data.

Extraction point: hidden_states[-1] (POST-norm, after final Llama-2 RMSNorm)
Pooling: masked mean-pool over the full multimodal sequence (same as CogACT/OpenVLA)

Prompt format: matches CogACT extraction:
  "What action should the robot take to {task}?" + appended tokens [29871, 2]

Requires: cogact conda env (has prismatic installed from /home/ubuntu/verl/CogACT)

Usage:
    python extract_features_raw_prismatic.py \
        --model_path /home/ubuntu/verl/prismatic-vlms-hf/prism-dinosiglip-224px+7b \
        --data_dir /home/ubuntu/verl/starVLA/cknna/cknna_data_exp2 \
        --output_dir /home/ubuntu/verl/starVLA/cknna/cknna_data_exp2/prismatic-raw
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic import load


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


def build_task_mask_prismatic(input_ids_1d, tokenizer, task, num_patches):
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


def _redirect_llama_to_local_cache():
    """Redirect meta-llama/Llama-2-7b-hf lookups to the pre-populated local cache.

    prismatic.load() calls AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    and AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf"), which fail without
    a HuggingFace token (gated model) unless the files are already in the HF cache.

    Run scripts/setup_prismatic_cache.sh once to populate the cache directory.
    This function then patches transformers to use that local directory instead.
    """
    llama_cache = os.environ.get(
        "PRISMATIC_LLAMA_CACHE",
        os.path.expanduser("~/.cache/huggingface/llama2-7b-config"),
    )
    if not os.path.isdir(llama_cache):
        raise FileNotFoundError(
            f"Llama-2 local cache not found at: {llama_cache}\n"
            "Run scripts/setup_prismatic_cache.sh first, or set PRISMATIC_LLAMA_CACHE "
            "to the directory containing config.json and tokenizer files."
        )
    if not os.path.isfile(os.path.join(llama_cache, "config.json")):
        raise FileNotFoundError(
            f"config.json missing in {llama_cache}\n"
            "Run scripts/setup_prismatic_cache.sh first."
        )
    # Monkey-patch: intercept from_pretrained("meta-llama/Llama-2-7b-hf", ...)
    # and redirect to the local cache directory.
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    _orig_auto_config = AutoConfig.from_pretrained.__func__ if hasattr(AutoConfig.from_pretrained, "__func__") else None
    _orig_auto_tok = AutoTokenizer.from_pretrained.__func__ if hasattr(AutoTokenizer.from_pretrained, "__func__") else None
    _orig_llama = LlamaForCausalLM.from_pretrained.__func__ if hasattr(LlamaForCausalLM.from_pretrained, "__func__") else None

    _LLAMA_HF_PATHS = {"meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"}

    import transformers
    _orig_config_fp = transformers.AutoConfig.from_pretrained

    @classmethod  # type: ignore[misc]
    def _patched_config(cls, pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path in _LLAMA_HF_PATHS:
            pretrained_model_name_or_path = llama_cache
            kwargs.pop("token", None)
        return _orig_config_fp(pretrained_model_name_or_path, *args, **kwargs)

    _orig_tok_fp = transformers.AutoTokenizer.from_pretrained

    @classmethod  # type: ignore[misc]
    def _patched_tok(cls, pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path in _LLAMA_HF_PATHS:
            pretrained_model_name_or_path = llama_cache
            kwargs.pop("token", None)
        return _orig_tok_fp(pretrained_model_name_or_path, *args, **kwargs)

    transformers.AutoConfig.from_pretrained = _patched_config
    transformers.AutoTokenizer.from_pretrained = _patched_tok

    print(f"[prismatic] Redirected meta-llama/Llama-2-7b-hf -> {llama_cache}")
    return llama_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local path to prism-dinosiglip-224px+7b directory")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    # Redirect Llama-2 HF hub calls to local cache (avoids gated model download)
    _redirect_llama_to_local_cache()

    print(f"Loading raw Prismatic VLM from: {args.model_path}")
    vlm = load(args.model_path, hf_token=None)
    vlm = vlm.to(torch.bfloat16).to(args.device).eval()

    tokenizer = vlm.llm_backbone.tokenizer
    image_transform = vlm.vision_backbone.image_transform
    hidden_size = vlm.llm_backbone.llm.config.hidden_size

    print(f"  hidden_size: {hidden_size}")
    print(f"  tokenizer type: {type(tokenizer).__name__}")
    print(f"  vision_backbone: {type(vlm.vision_backbone).__name__}")
    print(f"  llm_backbone: {type(vlm.llm_backbone).__name__}")
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
        prompt_builder.add_turn(role="human",
                                message=f"What action should the robot take to {task.lower()}?")
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

        task_mask = build_task_mask_prismatic(
            input_ids[0], tokenizer, task.lower(), num_patches
        ).unsqueeze(0)
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
        "model": "prism-dinosiglip-224px+7b (raw, no robot fine-tuning)",
        "extraction_point": "hidden_states[-1] (post-norm Llama-2 RMSNorm)",
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Raw Prismatic Extraction Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

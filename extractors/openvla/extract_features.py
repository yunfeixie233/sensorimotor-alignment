"""
Phase 2: Extract VLM features (feats_A) from OpenVLA-7B for CKNNA.

Loads the OpenVLA checkpoint, runs a forward pass with
output_hidden_states=True on each (image, task) pair from Phase 1 data,
and saves the masked-mean-pooled last hidden state as feats_A.

Architecture: Prismatic VLM (DINOv2 + SigLIP fused vision + Llama-2-7b LM, hidden_size=4096)
Extraction point: hidden_states[-1] (POST-norm, after final Llama-2 RMSNorm)
Pooling: masked mean-pool over non-padding tokens in the multimodal sequence
         (includes both projected vision patches and text tokens)

The multimodal sequence is: [BOS, vision_patches, text_tokens_after_BOS]
where vision patches are always valid (mask=1). The multimodal_attention_mask
is reconstructed from the original attention_mask since it is built internally
by PrismaticForConditionalGeneration.forward().

Prompt format: "In: What action should the robot take to {instruction}?\\nOut:"

Requires: conda env with transformers==4.40.1, timm==0.9.16, torch>=2.1

Usage:
    python extract_features_openvla.py \
        --ckpt $WORK/SimplerEnv-OpenVLA/checkpoints/openvla-7b \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/openvla-7b-bridge
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


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

    Prismatic multimodal sequence: [BOS, vision_patches, text_after_BOS].
    Task tokens appear within text_after_BOS. We find them in input_ids
    (text-only space) then offset by num_patches to map to multimodal positions.

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
    parser = argparse.ArgumentParser(description="Phase 2: Extract OpenVLA features for CKNNA.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(script_dir, "..", "checkpoints", "openvla-7b")
    parser.add_argument("--ckpt", type=str,
                        default=default_ckpt,
                        help="HuggingFace model ID or local path")
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

    print(f"Loading OpenVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  vision_backbone_id: {model.config.vision_backbone_id}")
    print(f"  llm_backbone_id: {model.config.llm_backbone_id}")
    print(f"  use_fused_vision_backbone: {model.config.use_fused_vision_backbone}")
    print(f"  Samples to process: {num_samples}")

    tokenizer = processor.tokenizer

    _hook_output = {}

    def _lm_pre_hook(_module, args, kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return args, kwargs

    def _lm_post_hook(_module, _input, output):
        _hook_output["hidden_states"] = output.hidden_states

    h1 = model.language_model.register_forward_pre_hook(_lm_pre_hook, with_kwargs=True)
    h2 = model.language_model.register_forward_hook(_lm_post_hook)

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        prompt = f"In: What action should the robot take to {task}?\nOut:"

        inputs = processor(prompt, img)
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16, device=args.device)

        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

        last_hidden = _hook_output["hidden_states"][-1]

        text_seq_len = input_ids.shape[1]
        vb = model.vision_backbone
        if hasattr(vb, 'get_num_patches'):
            num_patches = vb.get_num_patches() * vb.get_num_images_in_input()
        else:
            num_patches = vb.featurizer.patch_embed.num_patches

        multimodal_len = 1 + num_patches + (text_seq_len - 1)
        last_hidden_trimmed = last_hidden[:, :multimodal_len, :]

        patch_mask = torch.ones(
            (1, num_patches), dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )

        feat_imgtext = masked_mean_pool(last_hidden_trimmed, multimodal_mask).squeeze(0).cpu()

        image_mask = torch.zeros_like(multimodal_mask)
        image_mask[0, 1:1 + num_patches] = 1
        feat_img = masked_mean_pool(last_hidden_trimmed, image_mask).squeeze(0).cpu()

        task_mask = build_task_mask_prismatic(input_ids[0], tokenizer, task, num_patches).unsqueeze(0)
        feat_txt = masked_mean_pool(last_hidden_trimmed, task_mask).squeeze(0).cpu()

        feats_imgtext_list.append(feat_imgtext)
        feats_img_list.append(feat_img)
        feats_txt_list.append(feat_txt)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  shape=({hidden_size},)  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min")

    h1.remove()
    h2.remove()

    for suffix, flist in [("feats_A", feats_imgtext_list),
                          ("feats_A_img", feats_img_list),
                          ("feats_A_txt", feats_txt_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    extraction_meta = {
        "model": args.ckpt,
        "extraction_point": "hidden_states[-1] (post-norm Llama-2 RMSNorm, via hook)",
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== OpenVLA Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

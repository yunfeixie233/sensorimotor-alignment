"""
Extract VLM features (feats_A) from raw PaliGemma / PaliGemma-2 models for CKNNA.

These are the ORIGINAL pretrained models, NOT fine-tuned on robot actions.
Used to test the frozen-VLM hypothesis: do pretrained VLM features show
different CKNNA-vs-horizon trends compared to end-to-end fine-tuned VLAs?

Matches the prompt format used by StarVLA's extract_features_starvla.py
to ensure a fair comparison.

Supported checkpoints (both handled by same script):
  - PaliGemma-1: google/paligemma-3b-pt-224  (Gemma-2B backbone, hidden=2048)
  - PaliGemma-2: google/paligemma2-3b-pt-224 (Gemma-2-2B backbone, hidden=2304)

Usage:
  python extract_raw_paligemma.py \\
      --model_path google/paligemma-3b-pt-224 \\
      --data_dir ./cknna_data_exp1v2_full_L \\
      --output_dir ./cknna_data_exp1v2_full_L/paligemma1-raw
"""

import argparse
import json
import os
import time

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
        if seq[i : i + m] == subseq:
            return i
    return -1


def build_task_mask(input_ids_1d, tokenizer, task, image_token_id):
    """Build mask for task-instruction tokens in the input sequence.

    Skips image token positions. Fast path: exact BPE subsequence match.
    Fallback: character-level alignment via progressive prefix decoding.
    """
    mask = torch.zeros(len(input_ids_1d), dtype=torch.long, device=input_ids_1d.device)
    if not task:
        return mask
    ids_list = input_ids_1d.tolist()
    # Exclude image token positions from text decoding
    text_ids = [t for t in ids_list if t != image_token_id]
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start_in_text = find_subsequence(text_ids, task_ids)
        if start_in_text >= 0:
            # Map back to original indices (skip image token positions)
            text_pos = 0
            tok_start = None
            for k, tid in enumerate(ids_list):
                if tid == image_token_id:
                    continue
                if text_pos == start_in_text:
                    tok_start = k
                if text_pos >= start_in_text and text_pos < start_in_text + len(task_ids):
                    mask[k] = 1
                text_pos += 1
            if tok_start is not None:
                return mask
    # Fallback: character-level alignment on non-image tokens
    text_only = tokenizer.decode(text_ids, skip_special_tokens=True)
    char_pos = text_only.lower().find(task.lower().strip())
    if char_pos < 0:
        return mask
    task_end_char = char_pos + len(task.strip())
    text_ids_running = []
    text_indices = [k for k, t in enumerate(ids_list) if t != image_token_id]
    start_tok = None
    end_tok = None
    for j, k in enumerate(text_indices):
        text_ids_running.append(ids_list[k])
        prefix_len = len(tokenizer.decode(text_ids_running, skip_special_tokens=True))
        if start_tok is None and prefix_len > char_pos:
            start_tok = k
        if prefix_len >= task_end_char:
            end_tok = k + 1
            break
    if start_tok is not None and end_tok is not None:
        for k in range(start_tok, end_tok):
            if ids_list[k] != image_token_id:
                mask[k] = 1
    return mask


def load_paligemma(model_path):
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def build_inputs_paligemma(processor, img, instruction):
    """Build model inputs for PaliGemma feature extraction.

    PaliGemma uses a prefix-LM: image tokens + text are the prefix (bidirectional
    attention via token_type_ids=0). We pass the instruction as text input so all
    tokens are in prefix mode — correct for feature extraction.
    """
    batch = processor(
        text=instruction,
        images=img,
        return_tensors="pt",
    )
    return batch


def extract_feat_a(model, batch, tokenizer, instruction, device, image_token_id):
    batch = batch.to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]
    attention_mask = batch["attention_mask"]
    input_ids = batch["input_ids"]

    feat_imgtext = masked_mean_pool(last_hidden, attention_mask).squeeze(0).cpu()

    image_mask = (input_ids == image_token_id).to(attention_mask.dtype)
    feat_img = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    task_mask = build_task_mask(
        input_ids[0], tokenizer, instruction, image_token_id
    ).unsqueeze(0)
    feat_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt


def main():
    parser = argparse.ArgumentParser(description="Extract raw PaliGemma VLM features for CKNNA.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID or local path. "
                             "E.g. google/paligemma-3b-pt-224")
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

    print(f"Loading raw PaliGemma model: {args.model_path}")
    model, processor = load_paligemma(args.model_path)
    model = model.to(args.device).eval()

    tokenizer = processor.tokenizer
    image_token_id = model.config.image_token_index  # 256000
    hidden_size = model.config.text_config.hidden_size
    print(f"  image_token_id: {image_token_id}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  Samples: {num_samples}")

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []

    t0 = time.time()
    for i in range(num_samples):
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img = Image.open(img_path).convert("RGB")
        instruction = task_descriptions[i]

        batch = build_inputs_paligemma(processor, img, instruction)
        f_imgtext, f_img, f_txt = extract_feat_a(
            model, batch, tokenizer, instruction, args.device, image_token_id
        )
        feats_imgtext_list.append(f_imgtext)
        feats_img_list.append(f_img)
        feats_txt_list.append(f_txt)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{num_samples}]  shape=({hidden_size},)  "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f}min"
            )

    for suffix, flist in [
        ("feats_A", feats_imgtext_list),
        ("feats_A_img", feats_img_list),
        ("feats_A_txt", feats_txt_list),
    ]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    extraction_meta = {
        "model": args.model_path,
        "model_family": "paligemma",
        "type": "raw_pretrained_vlm",
        "hidden_size": hidden_size,
        "image_token_id": image_token_id,
        "num_samples": num_samples,
        "data_dir": args.data_dir,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. Time: {elapsed / 60:.1f} min ({elapsed / num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

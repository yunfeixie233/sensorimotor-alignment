"""
Extract VLM features (feats_A) from raw KosMos-2 model for CKNNA.

These are the ORIGINAL pretrained models, NOT fine-tuned on robot actions.
Used to test the frozen-VLM hypothesis: do pretrained VLM features show
different CKNNA-vs-horizon trends compared to end-to-end fine-tuned VLAs?

Matches the prompt format used by StarVLA's extract_features_starvla.py
to ensure a fair comparison.

Checkpoint: microsoft/kosmos-2-patch14-224
  - Architecture: MAGNETO transformer, hidden_size=2048
  - Image: ViT-L patch14 @ 224px → 64 image tokens
  - Uses image_embeds_position_mask (not input_ids == img_id) to locate image tokens

Usage:
  python extract_raw_kosmos2.py \\
      --model_path microsoft/kosmos-2-patch14-224 \\
      --data_dir ./cknna_data_exp1v2_full_L \\
      --output_dir ./cknna_data_exp1v2_full_L/kosmos2-raw
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


def build_task_mask_kosmos2(input_ids_1d, tokenizer, task, image_embeds_position_mask_1d):
    """Build mask for task-instruction tokens in the Kosmos-2 input sequence.

    In Kosmos-2, image embeddings are inserted at positions marked by
    image_embeds_position_mask. The input_ids sequence has the same length
    as the full sequence (image placeholder positions + text positions).
    We skip image positions when searching for the task instruction.
    """
    mask = torch.zeros(len(input_ids_1d), dtype=torch.long, device=input_ids_1d.device)
    if not task:
        return mask

    ids_list = input_ids_1d.tolist()
    img_mask = image_embeds_position_mask_1d.tolist()

    # Get text token ids (positions not occupied by image embeddings)
    text_ids = [ids_list[k] for k in range(len(ids_list)) if not img_mask[k]]
    text_indices = [k for k in range(len(ids_list)) if not img_mask[k]]

    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start_in_text = find_subsequence(text_ids, task_ids)
        if start_in_text >= 0:
            for j in range(start_in_text, start_in_text + len(task_ids)):
                mask[text_indices[j]] = 1
            return mask

    # Fallback: character-level alignment on text-only tokens
    text_only = tokenizer.decode(text_ids, skip_special_tokens=True)
    char_pos = text_only.lower().find(task.lower().strip())
    if char_pos < 0:
        return mask
    task_end_char = char_pos + len(task.strip())
    running_ids = []
    start_tok = None
    end_tok_idx = None
    for j, k in enumerate(text_indices):
        running_ids.append(ids_list[k])
        prefix_len = len(tokenizer.decode(running_ids, skip_special_tokens=True))
        if start_tok is None and prefix_len > char_pos:
            start_tok = k
        if prefix_len >= task_end_char:
            end_tok_idx = j + 1
            break
    if start_tok is not None and end_tok_idx is not None:
        for k in text_indices[: end_tok_idx]:
            if k >= start_tok:
                mask[k] = 1
    return mask


def load_kosmos2(model_path):
    from transformers import Kosmos2ForConditionalGeneration, Kosmos2Processor

    model = Kosmos2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    processor = Kosmos2Processor.from_pretrained(model_path)
    return model, processor


def build_inputs_kosmos2(processor, img, instruction):
    """Build model inputs for Kosmos-2 feature extraction.

    Kosmos-2 uses a grounding-aware prompt format. We wrap the instruction in
    the standard grounding prefix so the image embeddings are inserted at the
    correct position by the processor.

    The processor produces: input_ids, attention_mask, pixel_values,
    image_embeds_position_mask (marks the 64 image token positions).
    """
    prompt = f"<grounding>{instruction}"
    batch = processor(
        text=prompt,
        images=img,
        return_tensors="pt",
    )
    return batch


def extract_feat_a(model, batch, tokenizer, instruction, device):
    batch = batch.to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_embeds_position_mask=batch["image_embeds_position_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]
    attention_mask = batch["attention_mask"]
    input_ids = batch["input_ids"]
    image_embeds_position_mask = batch["image_embeds_position_mask"]

    feat_imgtext = masked_mean_pool(last_hidden, attention_mask).squeeze(0).cpu()

    # Image tokens: positions where image_embeds_position_mask == 1
    img_mask = image_embeds_position_mask.to(attention_mask.dtype)
    feat_img = masked_mean_pool(last_hidden, img_mask).squeeze(0).cpu()

    # Text tokens: build mask for instruction tokens
    task_mask = build_task_mask_kosmos2(
        input_ids[0], tokenizer, instruction, image_embeds_position_mask[0]
    ).unsqueeze(0)
    feat_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt


def main():
    parser = argparse.ArgumentParser(description="Extract raw KosMos-2 VLM features for CKNNA.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID or local path. "
                             "E.g. microsoft/kosmos-2-patch14-224")
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

    print(f"Loading raw KosMos-2 model: {args.model_path}")
    model, processor = load_kosmos2(args.model_path)
    model = model.to(args.device).eval()

    tokenizer = processor.tokenizer
    hidden_size = model.config.text_config.hidden_size
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

        batch = build_inputs_kosmos2(processor, img, instruction)
        f_imgtext, f_img, f_txt = extract_feat_a(
            model, batch, tokenizer, instruction, args.device
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
        "model_family": "kosmos2",
        "type": "raw_pretrained_vlm",
        "hidden_size": hidden_size,
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

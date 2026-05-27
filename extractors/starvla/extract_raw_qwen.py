"""
Extract VLM features (feats_A) from raw Qwen2.5-VL / Qwen3-VL models for CKNNA.

These are the ORIGINAL pretrained models, NOT fine-tuned on robot actions.
Used to test the frozen-VLM hypothesis: do pretrained VLM features show
different CKNNA-vs-horizon trends compared to end-to-end fine-tuned VLAs?

Matches the prompt format used by StarVLA's extract_features_starvla.py
to ensure a fair comparison.

Usage:
  python extract_features_raw_qwen.py \
      --model_path playground/Pretrained_models/Qwen2.5-VL-3B-Instruct \
      --data_dir ./cknna_data_exp2 \
      --output_dir ./cknna_data_exp2/qwen25vl-3b-raw \
      --model_family qwen2.5
"""

import argparse
import json
import os
import time

import torch
from PIL import Image


IMAGE_TOKEN_INDEX = 151655


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


def build_task_mask(input_ids_1d, tokenizer, task):
    """Build mask for task-instruction tokens in the input sequence.

    Fast path: exact BPE subsequence match with "" and " " prefixes.
    Fallback: character-level alignment via progressive prefix decoding.
    """
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
    start_tok = None
    end_tok = None
    for k in range(len(ids_list)):
        prefix_len = len(tokenizer.decode(ids_list[:k + 1], skip_special_tokens=True))
        if start_tok is None and prefix_len > char_pos:
            start_tok = k
        if prefix_len >= task_end_char:
            end_tok = k + 1
            break
    if start_tok is not None and end_tok is not None:
        mask[start_tok:end_tok] = 1
    return mask


def load_qwen25(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def load_qwen3(model_path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    return model, processor


def build_inputs_qwen25(processor, img, instruction):
    from qwen_vl_utils import process_vision_info

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
    ]
    texts = [
        processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return batch


def build_inputs_qwen3(processor, img, instruction):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        [messages],
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return batch


def extract_feat_a(model, batch, tokenizer, instruction, device):
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

    image_mask = (input_ids == IMAGE_TOKEN_INDEX).to(attention_mask.dtype)
    feat_img = masked_mean_pool(last_hidden, image_mask).squeeze(0).cpu()

    task_mask = build_task_mask(input_ids[0], tokenizer, instruction).unsqueeze(0)
    feat_txt = masked_mean_pool(last_hidden, task_mask).squeeze(0).cpu()

    return feat_imgtext, feat_img, feat_txt


def main():
    parser = argparse.ArgumentParser(description="Extract raw Qwen VLM features for CKNNA.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_family", type=str, required=True,
                        choices=["qwen2.5", "qwen3"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print(f"Loading raw Qwen model: {args.model_path}")
    if args.model_family == "qwen2.5":
        model, processor = load_qwen25(args.model_path)
        build_inputs = build_inputs_qwen25
    else:
        model, processor = load_qwen3(args.model_path)
        build_inputs = build_inputs_qwen3

    model = model.to(args.device).eval()
    tokenizer = processor.tokenizer
    if hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
    else:
        hidden_size = model.config.text_config.hidden_size
    print(f"  Model family: {args.model_family}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Samples: {num_samples}")

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []

    t0 = time.time()
    for i in range(num_samples):
        img_path = os.path.join(images_dir, f"{i:06d}.png")
        img = Image.open(img_path).convert("RGB")
        instruction = task_descriptions[i]

        batch = build_inputs(processor, img, instruction)
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
        "model_family": args.model_family,
        "type": "raw_pretrained_vlm",
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "data_dir": args.data_dir,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
        "image_token_index": IMAGE_TOKEN_INDEX,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. Time: {elapsed / 60:.1f} min ({elapsed / num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


IMAGE_TOKEN_INDEX = 257152


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


def build_task_mask(input_ids_1d, tokenizer, task):
    mask = torch.zeros(len(input_ids_1d), dtype=torch.long, device=input_ids_1d.device)
    if not task:
        return mask
    ids_list = input_ids_1d.tolist()
    for prefix in ["", " "]:
        task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
        start = find_subsequence(ids_list, task_ids)
        if start >= 0:
            mask[start:start + len(task_ids)] = 1
            return mask
    return mask


def main():
    parser = argparse.ArgumentParser(description="Unified single-pass extraction: feats_A + feats_action")
    parser.add_argument("--ckpt", type=str, default="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt, feats_A_img.pt, feats_A_txt.pt, feats_action.pt")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig/1.0.0",
                        help="Dataset key for intrinsic camera params (ZoeDepth)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--forward_pass_only", action="store_true",
                        help="Use single forward pass instead of model.generate(). "
                             "Faster but feats_action = feats_A (no action generation). "
                             "Auto-enabled for PT (pre-trained) checkpoints.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    # Auto-detect PT (pre-trained, not fine-tuned) checkpoint: use forward pass only
    is_pt = args.forward_pass_only or "-pt" in args.ckpt.lower()
    if is_pt:
        print(f"  NOTE: PT checkpoint detected -- using single forward pass (no autoregressive generation)")

    print(f"Loading SpatialVLA: {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(args.device)

    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  forward_pass_only: {is_pt}")
    print(f"  Samples to process: {num_samples}")

    tokenizer = processor.tokenizer

    partial_paths = {
        "feats_A": os.path.join(args.output_dir, "feats_A_partial.pt"),
        "feats_A_img": os.path.join(args.output_dir, "feats_A_img_partial.pt"),
        "feats_A_txt": os.path.join(args.output_dir, "feats_A_txt_partial.pt"),
        "feats_action": os.path.join(args.output_dir, "feats_action_partial.pt"),
    }

    feats_A_list = []
    feats_A_img_list = []
    feats_A_txt_list = []
    feats_action_list = []

    if args.resume_from > 0:
        if os.path.exists(partial_paths["feats_A"]):
            feats_A_list = list(torch.load(partial_paths["feats_A"], weights_only=True)[:args.resume_from])
        if os.path.exists(partial_paths["feats_A_img"]):
            feats_A_img_list = list(torch.load(partial_paths["feats_A_img"], weights_only=True)[:args.resume_from])
        if os.path.exists(partial_paths["feats_A_txt"]):
            feats_A_txt_list = list(torch.load(partial_paths["feats_A_txt"], weights_only=True)[:args.resume_from])
        if os.path.exists(partial_paths["feats_action"]):
            feats_action_list = list(torch.load(partial_paths["feats_action"], weights_only=True)[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from} ({len(feats_A_list)} loaded)")
    else:
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        inputs = processor(
            images=[img],
            text=task,
            unnorm_key=args.unnorm_key,
            return_tensors="pt",
            do_normalize=False,
        )
        inputs = inputs.to(dtype=torch.bfloat16, device=args.device)

        mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]

        if is_pt:
            # Single forward pass: no autoregressive generation.
            # feats_action = feats_A (PT model has no trained action head).
            with torch.inference_mode():
                out = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            prefill_hidden = out.hidden_states[-1]
            num_gen_steps = 0
        else:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            prefill_hidden = outputs.hidden_states[0][-1]
            num_gen_steps = len(outputs.hidden_states) - 1

        feat_A = masked_mean_pool(prefill_hidden, mask).squeeze(0).cpu()

        image_mask = (input_ids == IMAGE_TOKEN_INDEX).to(mask.dtype)
        feat_A_img = masked_mean_pool(prefill_hidden, image_mask).squeeze(0).cpu()

        task_mask = build_task_mask(input_ids[0], tokenizer, task).unsqueeze(0)
        feat_A_txt = masked_mean_pool(prefill_hidden, task_mask).squeeze(0).cpu()

        if is_pt:
            feat_action = feat_A.clone()
        elif num_gen_steps == 0:
            feat_action = torch.zeros(hidden_size, dtype=torch.float32)
        else:
            action_hidden = [outputs.hidden_states[t][-1] for t in range(1, len(outputs.hidden_states))]
            action_repr = torch.cat(action_hidden, dim=1)
            feat_action = action_repr.float().mean(dim=1).squeeze(0).cpu()

        feats_A_list.append(feat_A)
        feats_A_img_list.append(feat_A_img)
        feats_A_txt_list.append(feat_A_txt)
        feats_action_list.append(feat_action)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            mode_str = "fwd" if is_pt else f"gen={num_gen_steps}"
            print(f"  [{i+1}/{num_samples}]  {mode_str}  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_A_list), partial_paths["feats_A"])
            torch.save(torch.stack(feats_A_img_list), partial_paths["feats_A_img"])
            torch.save(torch.stack(feats_A_txt_list), partial_paths["feats_A_txt"])
            torch.save(torch.stack(feats_action_list), partial_paths["feats_action"])

    feats_A = torch.stack(feats_A_list)
    feats_A_img = torch.stack(feats_A_img_list)
    feats_A_txt = torch.stack(feats_A_txt_list)
    feats_action = torch.stack(feats_action_list)

    output_paths = {
        "feats_A": os.path.join(args.output_dir, "feats_A.pt"),
        "feats_A_img": os.path.join(args.output_dir, "feats_A_img.pt"),
        "feats_A_txt": os.path.join(args.output_dir, "feats_A_txt.pt"),
        "feats_action": os.path.join(args.output_dir, "feats_action.pt"),
    }

    torch.save(feats_A, output_paths["feats_A"])
    torch.save(feats_A_img, output_paths["feats_A_img"])
    torch.save(feats_A_txt, output_paths["feats_A_txt"])
    torch.save(feats_action, output_paths["feats_action"])

    for path in partial_paths.values():
        if os.path.exists(path):
            os.remove(path)

    extraction_meta = {
        "model": args.ckpt,
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt", "feats_action.pt"],
        "image_token_index": IMAGE_TOKEN_INDEX,
        "seed": args.seed,
        "extraction_method": "single-pass generate() with output_hidden_states=True",
        "feats_A_source": "hidden_states[0][-1] (prefill step)",
        "feats_action_source": "hidden_states[1:][-1] (generated tokens)",
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Unified Extraction Complete ===")
    print(f"  feats_A: {output_paths['feats_A']}  shape={tuple(feats_A.shape)}")
    print(f"  feats_A_img: {output_paths['feats_A_img']}  shape={tuple(feats_A_img.shape)}")
    print(f"  feats_A_txt: {output_paths['feats_A_txt']}  shape={tuple(feats_A_txt.shape)}")
    print(f"  feats_action: {output_paths['feats_action']}  shape={tuple(feats_action.shape)}")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

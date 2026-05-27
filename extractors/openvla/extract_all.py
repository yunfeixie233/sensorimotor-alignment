"""
Unified extraction script: Extract both feats_A (VLM hidden states) and feats_action
(action representations) from OpenVLA-7B in a SINGLE generate() call per sample.

Key insight: generate() with output_hidden_states=True returns:
- hidden_states[0][-1] = prefill step last-layer hidden states (feats_A)
- hidden_states[1:][-1] = each generated token's last-layer hidden states (feats_action)

Architecture: Prismatic VLM (CLIP + SigLIP vision + Llama-2-7B LM, hidden_size=4096)

Usage:
    python extract_all_openvla.py \
        --ckpt $WORK/SimplerEnv-OpenVLA/checkpoints/openvla-7b \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/openvla-7b-bridge \
        --seed 42
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


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


def main():
    parser = argparse.ArgumentParser(description="Unified extraction: feats_A + feats_action")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(script_dir, "..", "checkpoints", "openvla-7b")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Phase 1 data directory (images/, metadata.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save feats_A.pt and feats_action.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
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
    print(f"  Samples to process: {num_samples}")

    tokenizer = processor.tokenizer

    vb = model.vision_backbone
    if hasattr(vb, 'get_num_patches'):
        num_patches = vb.get_num_patches() * vb.get_num_images_in_input()
    else:
        num_patches = vb.featurizer.patch_embed.num_patches

    # The fine-tuned model (openvla-bridge-sft / ft-200k) overrides forward() to return
    # plain logits (a Tensor), which breaks generate(). Use a hook-based single forward
    # pass instead: capture hidden_states from the language model directly.
    is_ft_model = "bridge-sft" in args.ckpt or "ft-200k" in args.ckpt

    if is_ft_model:
        print("  Detected fine-tuned model -- using hook-based forward (not generate())")
        _hook_output = {}

        def _lm_pre_hook(_module, _args, kwargs):
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            return _args, kwargs

        def _lm_post_hook(_module, _input, output):
            _hook_output["hidden_states"] = output.hidden_states

        h1 = model.language_model.register_forward_pre_hook(_lm_pre_hook, with_kwargs=True)
        h2 = model.language_model.register_forward_hook(_lm_post_hook)

        # ACTION_DIM * NUM_ACTIONS_CHUNK from constants.py of the checkpoint
        # (7 * 5 = 35 for bridge-sft)
        import importlib.util, sys
        _spec = importlib.util.spec_from_file_location(
            "_openvla_sft_constants",
            os.path.join(args.ckpt, "constants.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        action_dim = _mod.ACTION_DIM
        num_actions_chunk = _mod.NUM_ACTIONS_CHUNK
        num_action_tokens = action_dim * num_actions_chunk
        print(f"  ACTION_DIM={action_dim}  NUM_ACTIONS_CHUNK={num_actions_chunk}  "
              f"action_tokens={num_action_tokens}")

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []
    feats_action_list = []

    partial_feats_A_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    partial_feats_action_path = os.path.join(args.output_dir, "feats_action_partial.pt")
    
    if args.resume_from > 0:
        if os.path.exists(partial_feats_A_path):
            partial_feats_A = torch.load(partial_feats_A_path, weights_only=True)
            feats_imgtext_list = list(partial_feats_A[:args.resume_from, 0])
            feats_img_list = list(partial_feats_A[:args.resume_from, 1])
            feats_txt_list = list(partial_feats_A[:args.resume_from, 2])
        if os.path.exists(partial_feats_action_path):
            partial_feats_action = torch.load(partial_feats_action_path, weights_only=True)
            feats_action_list = list(partial_feats_action[:args.resume_from])
        print(f"  Resuming from sample {args.resume_from}")
    else:
        args.resume_from = 0

    torch.manual_seed(args.seed)

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = task_descriptions[i]

        prompt = f"In: What action should the robot take to {task}?\nOut:"

        inputs = processor(prompt, img)
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16, device=args.device)

        text_seq_len = input_ids.shape[1]
        multimodal_len = 1 + num_patches + (text_seq_len - 1)

        patch_mask = torch.ones(
            (1, num_patches), dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )
        image_mask = torch.zeros_like(multimodal_mask)
        image_mask[0, 1:1 + num_patches] = 1
        task_mask = build_task_mask_prismatic(input_ids[0], tokenizer, task, num_patches).unsqueeze(0)

        if is_ft_model:
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )
            last_hidden = _hook_output["hidden_states"][-1]
            # last_hidden shape: (1, multimodal_len + num_action_tokens + 1, hidden_size)
            # The ft model appends action_tokens + stop_token inside forward(),
            # so action positions start right after multimodal_len.
            prefill_hidden_trimmed = last_hidden[:, :multimodal_len, :]
            feat_imgtext = masked_mean_pool(prefill_hidden_trimmed, multimodal_mask).squeeze(0).cpu()
            feat_img = masked_mean_pool(prefill_hidden_trimmed, image_mask).squeeze(0).cpu()
            feat_txt = masked_mean_pool(prefill_hidden_trimmed, task_mask).squeeze(0).cpu()

            action_start = multimodal_len
            action_end = action_start + num_action_tokens
            action_hidden = last_hidden[:, action_start:action_end, :]
            if action_hidden.shape[1] == 0:
                feat_action = torch.zeros(hidden_size, dtype=torch.float32)
            else:
                feat_action = action_hidden.float().mean(dim=1).squeeze(0).cpu()

            num_gen_steps = num_action_tokens
        else:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=7,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            prefill_hidden = outputs.hidden_states[0][-1]
            prefill_hidden_trimmed = prefill_hidden[:, :multimodal_len, :]
            feat_imgtext = masked_mean_pool(prefill_hidden_trimmed, multimodal_mask).squeeze(0).cpu()
            feat_img = masked_mean_pool(prefill_hidden_trimmed, image_mask).squeeze(0).cpu()
            feat_txt = masked_mean_pool(prefill_hidden_trimmed, task_mask).squeeze(0).cpu()

            num_gen_steps = len(outputs.hidden_states) - 1
            if num_gen_steps == 0:
                feat_action = torch.zeros(hidden_size, dtype=torch.float32)
            else:
                action_hidden_tensors = []
                for t in range(1, len(outputs.hidden_states)):
                    action_hidden_tensors.append(outputs.hidden_states[t][-1])
                action_repr = torch.cat(action_hidden_tensors, dim=1)
                feat_action = action_repr.float().mean(dim=1).squeeze(0).cpu()

        feats_imgtext_list.append(feat_imgtext)
        feats_img_list.append(feat_img)
        feats_txt_list.append(feat_txt)
        feats_action_list.append(feat_action)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{num_samples}]  gen_tokens={num_gen_steps}  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

        if (i + 1) % args.save_every == 0:
            feats_A_partial = torch.stack([
                torch.stack([feats_imgtext_list[j], feats_img_list[j], feats_txt_list[j]])
                for j in range(len(feats_imgtext_list))
            ])
            torch.save(feats_A_partial, partial_feats_A_path)
            torch.save(torch.stack(feats_action_list), partial_feats_action_path)

    feats_A = torch.stack([
        torch.stack([feats_imgtext_list[j], feats_img_list[j], feats_txt_list[j]])
        for j in range(len(feats_imgtext_list))
    ])
    feats_action = torch.stack(feats_action_list)

    for suffix, tensor in [("feats_A", feats_A[:, 0]), ("feats_A_img", feats_A[:, 1]), 
                          ("feats_A_txt", feats_A[:, 2]), ("feats_action", feats_action)]:
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(tensor, p)
        print(f"  Saved {p}  shape={tuple(tensor.shape)}")

    if os.path.exists(partial_feats_A_path):
        os.remove(partial_feats_A_path)
    if os.path.exists(partial_feats_action_path):
        os.remove(partial_feats_action_path)

    extraction_meta = {
        "model": args.ckpt,
        "extraction_point": "generate() hidden_states[0][-1] (prefill) + hidden_states[1:][-1] (gen tokens)",
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "seed": args.seed,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt", "feats_action.pt"],
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Unified Extraction Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

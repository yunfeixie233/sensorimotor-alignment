import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GROOT_N15 = os.environ.get("ISAAC_GROOT_DIR", os.path.join(_PROJECT_ROOT, "repos", "Isaac-GR00T-N15"))
if _GROOT_N15 not in sys.path:
    sys.path.insert(0, _GROOT_N15)

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values

from transformers.image_processing_utils_fast import BaseImageProcessorFast
if not hasattr(BaseImageProcessorFast, "_prepare_input_images"):
    from functools import partial as _partial

    def _prepare_input_images(self, images, do_convert_rgb, input_data_format, device):
        images = self._prepare_images_structure(images)
        fn = _partial(self._process_image, do_convert_rgb=do_convert_rgb,
                      input_data_format=input_data_format, device=device)
        return [fn(img) for img in images]

    BaseImageProcessorFast._prepare_input_images = _prepare_input_images


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


def build_bridge_batch(image_np, task_description, state_8d):
    img = image_np[None]
    return {
        "video.image_0": img,
        "state.x": state_8d[0:1][None],
        "state.y": state_8d[1:2][None],
        "state.z": state_8d[2:3][None],
        "state.roll": state_8d[3:4][None],
        "state.pitch": state_8d[4:5][None],
        "state.yaw": state_8d[5:6][None],
        "state.pad": state_8d[6:7][None],
        "state.gripper": state_8d[7:8][None],
        "annotation.human.action.task_description": [task_description],
    }


def feats_b_7d_to_state_8d(feats_B):
    n = feats_B.shape[0]
    state_8d = np.zeros((n, 8), dtype=np.float32)
    state_8d[:, :6] = feats_B[:, :6]
    state_8d[:, 6] = 0.0
    state_8d[:, 7] = feats_B[:, 6]
    return state_8d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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

    feats_B = torch.load(
        os.path.join(args.data_dir, "feats_B.pt"), weights_only=True
    ).numpy()
    state_all_8d = feats_b_7d_to_state_8d(feats_B)

    print(f"Loading GR00T N1.5: {args.ckpt}")

    data_config = DATA_CONFIG_MAP["bridge"]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.ckpt,
        modality_config=modality_config,
        modality_transform=transforms,
        embodiment_tag="oxe",
        device=args.device,
    )
    model = policy.model

    backbone_cfg = model.config.backbone_cfg
    hidden_size = 2048
    action_hidden_size = model.action_head.hidden_size
    action_horizon = model.action_head.action_horizon
    print(f"  select_layer: {backbone_cfg['select_layer']}")
    print(f"  project_to_dim: {backbone_cfg['project_to_dim']}")
    print(f"  backbone hidden_size: {hidden_size}")
    print(f"  action_hidden_size: {action_hidden_size}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  Samples to process: {num_samples}")

    eagle_image_token_index = model.backbone.eagle_model.config.image_token_index
    from transformers import AutoTokenizer
    eagle_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(__import__('gr00t').__file__),
                     "model", "backbone", "eagle2_hg_model"),
        trust_remote_code=True, use_fast=False,
    )
    print(f"  eagle image_token_index: {eagle_image_token_index}")

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []
    feats_action_list = []

    partial_feats_A_path = os.path.join(args.output_dir, "feats_A_partial.pt")
    partial_feats_A_img_path = os.path.join(args.output_dir, "feats_A_img_partial.pt")
    partial_feats_A_txt_path = os.path.join(args.output_dir, "feats_A_txt_partial.pt")
    partial_feats_action_path = os.path.join(args.output_dir, "feats_action_partial.pt")

    if args.resume_from > 0:
        if os.path.exists(partial_feats_A_path):
            feats_imgtext_list = list(torch.load(partial_feats_A_path, weights_only=True)[:args.resume_from])
        if os.path.exists(partial_feats_A_img_path):
            feats_img_list = list(torch.load(partial_feats_A_img_path, weights_only=True)[:args.resume_from])
        if os.path.exists(partial_feats_A_txt_path):
            feats_txt_list = list(torch.load(partial_feats_A_txt_path, weights_only=True)[:args.resume_from])
        if os.path.exists(partial_feats_action_path):
            feats_action_list = list(torch.load(partial_feats_action_path, weights_only=True)[:args.resume_from])
        print(f"  Resuming from {args.resume_from}")
    else:
        args.resume_from = 0

    t0 = time.time()
    for i in range(args.resume_from, num_samples):
        img = np.array(
            Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB").resize((256, 256))
        )
        task = task_descriptions[i]
        state_8d = state_all_8d[i]

        batch = build_bridge_batch(img, task, state_8d)
        batch_unsqueezed = unsqueeze_dict_values(batch)
        normalized_input = policy.apply_transforms(batch_unsqueezed)

        torch.manual_seed(args.seed + i)

        backbone_outputs_capture = {}
        action_reprs = []

        def backbone_post_hook(module, inputs, outputs):
            backbone_outputs_capture["backbone_features"] = outputs["backbone_features"].detach()
            backbone_outputs_capture["backbone_attention_mask"] = outputs["backbone_attention_mask"].detach()
            backbone_outputs_capture["eagle_input_ids"] = inputs[0]["eagle_input_ids"].detach()

        def action_decoder_pre_hook(module, inputs):
            action_reprs.append(inputs[0].detach())

        backbone_handle = model.backbone.register_forward_hook(backbone_post_hook)
        action_handle = model.action_head.action_decoder.register_forward_pre_hook(action_decoder_pre_hook)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            model.get_action(normalized_input)

        backbone_handle.remove()
        action_handle.remove()

        backbone_features = backbone_outputs_capture["backbone_features"]
        backbone_mask = backbone_outputs_capture["backbone_attention_mask"]
        eagle_input_ids = backbone_outputs_capture["eagle_input_ids"]

        feat_imgtext = masked_mean_pool(backbone_features, backbone_mask).squeeze(0).cpu()

        image_mask = (eagle_input_ids == eagle_image_token_index).to(backbone_mask.dtype)
        feat_img = masked_mean_pool(backbone_features, image_mask).squeeze(0).cpu()

        task_mask = build_task_mask(eagle_input_ids[0], eagle_tokenizer, task).unsqueeze(0)
        feat_txt = masked_mean_pool(backbone_features, task_mask).squeeze(0).cpu()

        final_repr = action_reprs[-1]
        action_part = final_repr[:, -action_horizon:, :]
        feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()

        feats_imgtext_list.append(feat_imgtext)
        feats_img_list.append(feat_img)
        feats_txt_list.append(feat_txt)
        feats_action_list.append(feat_action)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            done = i + 1 - args.resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{num_samples}]  feats_A=({hidden_size},)  feats_action=({action_hidden_size},)  "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True
            )

        if (i + 1) % args.save_every == 0:
            torch.save(torch.stack(feats_imgtext_list), partial_feats_A_path)
            torch.save(torch.stack(feats_img_list), partial_feats_A_img_path)
            torch.save(torch.stack(feats_txt_list), partial_feats_A_txt_path)
            torch.save(torch.stack(feats_action_list), partial_feats_action_path)

    feats_A = torch.stack(feats_imgtext_list)
    feats_A_img = torch.stack(feats_img_list)
    feats_A_txt = torch.stack(feats_txt_list)
    feats_action = torch.stack(feats_action_list)

    feats_A_path = os.path.join(args.output_dir, "feats_A.pt")
    feats_A_img_path = os.path.join(args.output_dir, "feats_A_img.pt")
    feats_A_txt_path = os.path.join(args.output_dir, "feats_A_txt.pt")
    feats_action_path = os.path.join(args.output_dir, "feats_action.pt")

    torch.save(feats_A, feats_A_path)
    torch.save(feats_A_img, feats_A_img_path)
    torch.save(feats_A_txt, feats_A_txt_path)
    torch.save(feats_action, feats_action_path)

    for p in [partial_feats_A_path, partial_feats_A_img_path, partial_feats_A_txt_path, partial_feats_action_path]:
        if os.path.exists(p):
            os.remove(p)

    for suffix, t in [("feats_A", feats_A),
                      ("feats_A_img", feats_A_img),
                      ("feats_A_txt", feats_A_txt),
                      ("feats_action", feats_action)]:
        print(f"  Saved {suffix}.pt  shape={tuple(t.shape)}")

    extraction_meta = {
        "model": "GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2",
        "checkpoint_path": args.ckpt,
        "hidden_size": hidden_size,
        "action_hidden_size": action_hidden_size,
        "action_horizon": action_horizon,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt", "feats_action.pt"],
        "eagle_image_token_index": eagle_image_token_index,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== GR00T N1.5 Unified Extraction Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

"""
7-feature extraction for GR00T N1.5 and N1.6 (combined script).

Extracts all 7 features + action in a SINGLE get_action() call per sample:
  P1-V   vit_raw:      vision post_layernorm hook -> (1, N, 1152). Mean-pool dim=1.
  P2-V   pre_llm:      mlp1 hook -> (1, N, 2048). Mean-pool dim=1.
  P2-T   pre_llm_txt:  language_model.model.layers[0] pre-hook, text-only mask
  P2-VT  pre_llm_vt:   language_model.model.layers[0] pre-hook, all tokens mean-pooled
  P3-V   img:           backbone_features, image mask
  P3-T   txt:           backbone_features, text mask
  P3-VT  imgtext:       backbone_features, backbone_attention_mask
  Action: action_decoder pre-hook

MUST run with PYTHONNOUSERSITE=1 to avoid transformers 5.4.0 dataclass breakage.

Usage:
  PYTHONNOUSERSITE=1 python extractors/extract_7feat_groot.py \
      --version n15 \
      --ckpt /path/to/groot-n15-checkpoint \
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge \
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_bridge_7feat/GR00T-N15 \
      [--max_samples 10]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

torch.backends.cudnn.enabled = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def masked_mean_pool(hidden_states, mask):
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
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
    return mask


def feats_b_7d_to_state_8d(feats_B):
    n = feats_B.shape[0]
    state_8d = np.zeros((n, 8), dtype=np.float32)
    state_8d[:, :6] = feats_B[:, :6]
    state_8d[:, 6] = 0.0
    state_8d[:, 7] = feats_B[:, 6]
    return state_8d


def resolve_module_by_path(model, path):
    """Resolve a dotted module path from model."""
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


# =========================================================================
# N1.5 specifics
# =========================================================================

def build_bridge_batch_n15(image_np, task_description, state_8d):
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


def load_n15(ckpt, device):
    groot_dir = os.environ.get("ISAAC_GROOT_DIR", os.path.join(_PROJECT_ROOT, "repos", "Isaac-GR00T-N15"))
    if groot_dir not in sys.path:
        sys.path.insert(0, groot_dir)

    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values

    # BaseImageProcessorFast patch
    from transformers.image_processing_utils_fast import BaseImageProcessorFast
    if not hasattr(BaseImageProcessorFast, "_prepare_input_images"):
        from functools import partial as _partial

        def _prepare_input_images(self, images, do_convert_rgb, input_data_format, device):
            images = self._prepare_images_structure(images)
            fn = _partial(self._process_image, do_convert_rgb=do_convert_rgb,
                          input_data_format=input_data_format, device=device)
            return [fn(img) for img in images]

        BaseImageProcessorFast._prepare_input_images = _prepare_input_images

    data_config = DATA_CONFIG_MAP["bridge"]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    policy = Gr00tPolicy(
        model_path=ckpt,
        modality_config=modality_config,
        modality_transform=transforms,
        embodiment_tag="oxe",
        device=device,
    )
    return policy, unsqueeze_dict_values


def load_n16(ckpt, device):
    groot_dir = os.environ.get("ISAAC_GROOT_N16_DIR", os.path.join(_PROJECT_ROOT, "repos", "Isaac-GR00T-N16"))
    if groot_dir not in sys.path:
        sys.path.insert(0, groot_dir)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    # BaseImageProcessorFast patch
    from transformers.image_processing_utils_fast import BaseImageProcessorFast
    if not hasattr(BaseImageProcessorFast, "_prepare_input_images"):
        from functools import partial as _partial

        def _prepare_input_images(self, images, do_convert_rgb, input_data_format, device):
            images = self._prepare_images_structure(images)
            fn = _partial(self._process_image, do_convert_rgb=do_convert_rgb,
                          input_data_format=input_data_format, device=device)
            return [fn(img) for img in images]

        BaseImageProcessorFast._prepare_input_images = _prepare_input_images

    policy = Gr00tPolicy(
        model_path=ckpt,
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        device=device,
    )
    return policy, _rec_to_dtype


# =========================================================================
# N1.6 batch building
# =========================================================================

def build_bridge_obs_n16(image_np, task_description, state_8d):
    return {
        "video": {
            "image_0": image_np[np.newaxis, np.newaxis],
        },
        "state": {
            "x": state_8d[0:1].reshape(1, 1, 1).astype(np.float32),
            "y": state_8d[1:2].reshape(1, 1, 1).astype(np.float32),
            "z": state_8d[2:3].reshape(1, 1, 1).astype(np.float32),
            "roll": state_8d[3:4].reshape(1, 1, 1).astype(np.float32),
            "pitch": state_8d[4:5].reshape(1, 1, 1).astype(np.float32),
            "yaw": state_8d[5:6].reshape(1, 1, 1).astype(np.float32),
            "pad": state_8d[6:7].reshape(1, 1, 1).astype(np.float32),
            "gripper": state_8d[7:8].reshape(1, 1, 1).astype(np.float32),
        },
        "language": {
            "annotation.human.action.task_description": [[task_description]],
        },
    }


def process_single_obs_n16(policy, obs, _rec_to_dtype):
    unbatched = policy._unbatch_observation(obs)
    vla_step = policy._to_vla_step_data(unbatched[0])
    from gr00t.data.types import MessageType
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
    processed = policy.processor(messages)
    collated = policy.collate_fn([processed])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    return collated["inputs"]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for GR00T N1.5/N1.6.")
    parser.add_argument("--version", type=str, required=True, choices=["n15", "n16"],
                        help="GR00T version: n15 or n16")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CKNNA data dir (images/, metadata.json, feats_B.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save 7 .pt + feats_action.pt files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    is_n15 = args.version == "n15"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    feats_B = torch.load(
        os.path.join(args.data_dir, "feats_B.pt"), weights_only=True
    ).numpy()
    state_all_8d = feats_b_7d_to_state_8d(feats_B)

    # --- Load model ---
    if is_n15:
        print(f"Loading GR00T N1.5: {args.ckpt}")
        policy, unsqueeze_dict_values = load_n15(args.ckpt, args.device)
    else:
        print(f"Loading GR00T N1.6: {args.ckpt}")
        policy, _rec_to_dtype = load_n16(args.ckpt, args.device)

    model = policy.model
    hidden_size = 2048
    action_hidden_size = model.action_head.hidden_size
    action_horizon = model.action_head.action_horizon
    print(f"  action_hidden_size: {action_hidden_size}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  Samples: {num_samples}")

    # --- Resolve module paths (version-specific) ---
    if is_n15:
        # N1.5: Eagle model paths (double-nested vision_model)
        vit_post_ln = resolve_module_by_path(
            model, "backbone.eagle_model.vision_model.vision_model.post_layernorm"
        )
        mlp1 = resolve_module_by_path(model, "backbone.eagle_model.mlp1")
        layer0 = resolve_module_by_path(model, "backbone.eagle_model.language_model.model.layers.0")
        eagle_image_token_index = model.backbone.eagle_model.config.image_token_index

        from transformers import AutoTokenizer
        eagle_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(os.path.dirname(__import__('gr00t').__file__),
                         "model", "backbone", "eagle2_hg_model"),
            trust_remote_code=True, use_fast=False,
        )
        print(f"  eagle image_token_index: {eagle_image_token_index}")
    else:
        # N1.6: Siglip2 paths (double-nested vision_model)
        vit_post_ln = resolve_module_by_path(
            model, "backbone.model.vision_model.vision_model.post_layernorm"
        )
        mlp1 = resolve_module_by_path(model, "backbone.model.mlp1")
        layer0 = resolve_module_by_path(model, "backbone.model.language_model.model.layers.0")
        image_token_index = model.backbone.model.config.image_token_index

        backbone_tokenizer = policy.processor.processor.tokenizer
        print(f"  image_token_index: {image_token_index}")

    assert vit_post_ln is not None, "Cannot find vision post_layernorm"
    assert mlp1 is not None, "Cannot find mlp1 (projector)"
    assert layer0 is not None, "Cannot find language_model.layers[0]"

    vit_dim = vit_post_ln.weight.shape[0]
    print(f"  P1-V: post_layernorm found (dim={vit_dim})")
    print(f"  P2-V: mlp1 found")
    print(f"  P2-T/VT: layers[0] found")

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}
    accum["action"] = []

    t0 = time.time()
    for i in range(num_samples):
        task = task_descriptions[i]
        state_8d = state_all_8d[i]

        if is_n15:
            img = np.array(
                Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB").resize((256, 256))
            )
            batch_raw = build_bridge_batch_n15(img, task, state_8d)
            batch_unsqueezed = unsqueeze_dict_values(batch_raw)
            inputs = policy.apply_transforms(batch_unsqueezed)
        else:
            img = np.array(
                Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
            )
            obs = build_bridge_obs_n16(img, task, state_8d)
            inputs = process_single_obs_n16(policy, obs, _rec_to_dtype)

        torch.manual_seed(42 + i)

        # --- Register hooks ---
        captures = {}
        action_reprs = []

        def hook_p1(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            captures["p1_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

        def hook_p2(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            # mlp1 may output (seq, D) without batch; handle both
            if t.dim() == 2:
                captures["p2_v"] = t.detach().float().mean(dim=0).cpu()
            else:
                captures["p2_v"] = t.detach().float().mean(dim=1).squeeze(0).cpu()

        def hook_layer0_pre(module, args_tuple):
            t = args_tuple[0]  # (1, seq_len, 2048)
            hidden = t.detach().float()
            captures["p2_vt"] = hidden.mean(dim=1).squeeze(0).cpu()
            # Text mask will be computed after backbone hook provides masks
            captures["layer0_hidden"] = hidden

        def backbone_post_hook(module, hook_inputs, outputs):
            captures["backbone_features"] = outputs["backbone_features"].detach()
            captures["backbone_attention_mask"] = outputs["backbone_attention_mask"].detach()
            if is_n15:
                captures["eagle_input_ids"] = hook_inputs[0]["eagle_input_ids"].detach()
            else:
                captures["image_mask"] = outputs["image_mask"].detach()
                captures["input_ids"] = hook_inputs[0]["input_ids"].detach()

        def action_decoder_pre_hook(module, hook_inputs):
            action_reprs.append(hook_inputs[0].detach())

        h1 = vit_post_ln.register_forward_hook(hook_p1)
        h2 = mlp1.register_forward_hook(hook_p2)
        h3 = layer0.register_forward_pre_hook(hook_layer0_pre)
        h_backbone = model.backbone.register_forward_hook(backbone_post_hook)
        h_action = model.action_head.action_decoder.register_forward_pre_hook(action_decoder_pre_hook)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.get_action(inputs)

        h1.remove()
        h2.remove()
        h3.remove()
        h_backbone.remove()
        h_action.remove()

        # --- P2-T: text mask from layer0 hidden ---
        if "layer0_hidden" in captures:
            layer0_hidden = captures["layer0_hidden"]
            backbone_mask = captures["backbone_attention_mask"]
            if is_n15:
                eagle_ids = captures["eagle_input_ids"]
                img_mask_l0 = (eagle_ids == eagle_image_token_index).long()
            else:
                img_mask_l0 = captures["image_mask"].to(torch.long)
            # backbone_mask may be shorter/longer than layer0_hidden; use min
            seq_len_l0 = layer0_hidden.shape[1]
            if backbone_mask.shape[1] >= seq_len_l0:
                attn_l0 = backbone_mask[:, :seq_len_l0].long()
                img_l0 = img_mask_l0[:, :seq_len_l0] if img_mask_l0.shape[1] >= seq_len_l0 else torch.nn.functional.pad(img_mask_l0, (0, seq_len_l0 - img_mask_l0.shape[1]))
            else:
                attn_l0 = torch.nn.functional.pad(backbone_mask.long(), (0, seq_len_l0 - backbone_mask.shape[1]))
                img_l0 = torch.nn.functional.pad(img_mask_l0.long(), (0, seq_len_l0 - img_mask_l0.shape[1]))
            text_mask_l0 = (attn_l0 - img_l0).clamp(min=0)
            captures["p2_t"] = masked_mean_pool(layer0_hidden, text_mask_l0).squeeze(0).cpu()
            del captures["layer0_hidden"]

        # --- P3 from backbone_features ---
        backbone_features = captures["backbone_features"]
        backbone_mask = captures["backbone_attention_mask"]

        captures["p3_vt"] = masked_mean_pool(backbone_features, backbone_mask).squeeze(0).cpu()

        if is_n15:
            eagle_ids = captures["eagle_input_ids"]
            image_mask = (eagle_ids == eagle_image_token_index).to(backbone_mask.dtype)
            captures["p3_v"] = masked_mean_pool(backbone_features, image_mask).squeeze(0).cpu()

            task_mask = build_task_mask(eagle_ids[0], eagle_tokenizer, task).unsqueeze(0)
            captures["p3_t"] = masked_mean_pool(backbone_features, task_mask).squeeze(0).cpu()
        else:
            image_mask = captures["image_mask"].to(backbone_mask.dtype)
            captures["p3_v"] = masked_mean_pool(backbone_features, image_mask).squeeze(0).cpu()

            # N1.6: task mask always fails -> use non-image, non-padding as text
            image_mask_bool = image_mask.to(torch.bool)
            backbone_mask_bool = backbone_mask.to(torch.bool)
            text_mask = (backbone_mask_bool & ~image_mask_bool).to(backbone_mask.dtype)
            captures["p3_t"] = masked_mean_pool(backbone_features, text_mask).squeeze(0).cpu()

        # --- Action ---
        final_repr = action_reprs[-1]
        action_part = final_repr[:, -action_horizon:, :]
        feat_action = action_part.float().mean(dim=1).squeeze(0).cpu()

        for k in keys:
            if k not in captures or captures[k] is None:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(captures[k])
        accum["action"].append(feat_action)

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            shapes = "  ".join(f"{k}={tuple(captures[k].shape)}" for k in keys)
            print(
                f"  [{i+1}/{num_samples}]  {shapes}  action={tuple(feat_action.shape)}  "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f}min",
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
        "action": "feats_action.pt",
    }

    print(f"\nSaving {len(file_map)} feature tensors:")
    for k in list(keys) + ["action"]:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<28} shape={tuple(tensor.shape)}")

    version_label = "GR00T-N1.5" if is_n15 else "GR00T-N1.6"
    meta = {
        "model": f"{version_label}",
        "checkpoint_path": args.ckpt,
        "type": "all_7_features_plus_action",
        "architecture": f"{version_label} (Eagle/Siglip2 + Qwen3)",
        "num_samples": num_samples,
        "hidden_size": hidden_size,
        "vit_dim": vit_dim,
        "action_hidden_size": action_hidden_size,
        "action_horizon": action_horizon,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in list(keys) + ["action"]},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()

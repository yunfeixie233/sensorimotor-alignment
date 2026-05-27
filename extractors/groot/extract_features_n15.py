"""
Phase 2: Extract VLM backbone features (feats_A) from GR00T N1.5 for CKNNA.

Architecture:
    GR00T_N1_5 = EagleBackbone (NVEagle: SigLIP2 + Qwen3-1.7B, truncated to 12 layers)
               + FlowmatchingActionHead (DiT)

Extraction point:
    backbone_features from EagleBackbone.forward()
    = eagle_model.hidden_states[12] (POST-norm, after Qwen3 final RMSNorm of truncated model)
    = (B, seq_len, 2048)
    project_to_dim=null means no linear projection is applied (Identity).

    State enters ONLY via the action head (state_proj), NOT the backbone.
    So backbone_features contains pure vision+text features.

Pooling: masked mean-pool over non-padding tokens (using backbone_attention_mask)

Alignment with other models:
    - Same semantic target as StarVLA/SpatialVLA/Pi0/OpenVLA:
      post-VLM condition embedding, before action decoder.
    - POST-norm consistent across all models.
    - Masked mean-pool consistent across all models.
    - No state leakage in feats_A.

Requires:
    transformers==4.51.3, flash-attn, Isaac-GR00T on PYTHONPATH

Usage:
    PYTHONPATH=$WORK/Isaac-GR00T python extract_features_groot_n15.py \
        --ckpt $WORK/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2 \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/groot-n15-bridge
"""

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
from gr00t.model.policy import Gr00tPolicy

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


def build_task_mask(input_ids_1d, tokenizer, task):
    """Build mask for task-instruction tokens in the input sequence.

    Tries bare encoding first (handles SentencePiece/BPE after non-space chars
    like <bos> or newline), then falls back to space-prefixed encoding.
    """
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


def build_bridge_batch(image_np, task_description):
    """Construct an observation dict matching GR00T's Bridge format.

    Uses dummy state values since state enters only the action head, not
    the backbone. The backbone output is identical regardless of state.

    Args:
        image_np: (H, W, 3) uint8 numpy array
        task_description: str

    Returns:
        batch dict ready for Gr00tPolicy.apply_transforms()
    """
    img = image_np[None]  # (1, H, W, C) -- T=1
    zeros = np.zeros((1, 1))  # (T=1, D=1) for each scalar state key
    return {
        "video.image_0": img,
        "state.x": zeros.copy(),
        "state.y": zeros.copy(),
        "state.z": zeros.copy(),
        "state.roll": zeros.copy(),
        "state.pitch": zeros.copy(),
        "state.yaw": zeros.copy(),
        "state.pad": zeros.copy(),
        "state.gripper": zeros.copy(),
        "annotation.human.action.task_description": [task_description],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract GR00T N1.5 backbone features for CKNNA."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to GR00T N1.5 checkpoint directory (e.g. ShuaiYang03/GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2)",
    )
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
    print(f"  select_layer: {backbone_cfg['select_layer']}")
    print(f"  project_to_dim: {backbone_cfg['project_to_dim']}")
    print(f"  backbone hidden_size: {hidden_size}")
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

    t0 = time.time()
    for i in range(num_samples):
        img = np.array(
            Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB").resize((256, 256))
        )
        task = task_descriptions[i]

        batch = build_bridge_batch(img, task)

        from gr00t.model.policy import unsqueeze_dict_values
        batch_unsqueezed = unsqueeze_dict_values(batch)

        normalized_input = policy.apply_transforms(batch_unsqueezed)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            backbone_inputs, _ = model.prepare_input(normalized_input)
            backbone_outputs = model.backbone(backbone_inputs)

        backbone_features = backbone_outputs["backbone_features"]
        backbone_mask = backbone_outputs["backbone_attention_mask"]

        feat_imgtext = masked_mean_pool(backbone_features, backbone_mask).squeeze(0).cpu()

        eagle_input_ids = backbone_inputs["eagle_input_ids"]
        image_mask = (eagle_input_ids == eagle_image_token_index).to(backbone_mask.dtype)
        feat_img = masked_mean_pool(backbone_features, image_mask).squeeze(0).cpu()

        task_mask = build_task_mask(eagle_input_ids[0], eagle_tokenizer, task).unsqueeze(0)
        feat_txt = masked_mean_pool(backbone_features, task_mask).squeeze(0).cpu()

        feats_imgtext_list.append(feat_imgtext)
        feats_img_list.append(feat_img)
        feats_txt_list.append(feat_txt)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{num_samples}]  shape=({hidden_size},)  "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f}min"
            )

    for suffix, flist in [("feats_A", feats_imgtext_list),
                          ("feats_A_img", feats_img_list),
                          ("feats_A_txt", feats_txt_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, f"{suffix}.pt")
        torch.save(t, p)
        print(f"  Saved {p}  shape={tuple(t.shape)}")

    extraction_meta = {
        "model": "GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2",
        "checkpoint_path": args.ckpt,
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
        "eagle_image_token_index": eagle_image_token_index,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== GR00T N1.5 Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

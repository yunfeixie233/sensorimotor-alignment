"""
Extract VLM backbone features (feats_A) from GR00T N1.6 for CKNNA.

Architecture:
    Gr00tN1d6 = EagleBackbone (Eagle-Block2A-2B-v2: SigLIP2 + Qwen3-1.7B, truncated to 16 layers)
              + Gr00tN1d6ActionHead (AlternateVLDiT, 32 layers)

Extraction point:
    backbone_features from EagleBackbone.forward()
    = eagle_model.hidden_states[-1] after truncation to 16 layers
    = POST-norm (Qwen3 final RMSNorm applied)
    = (B, seq_len, 2048)

    State enters ONLY via the action head (state_encoder), NOT the backbone.
    So backbone_features contains pure vision+text features.
    NOTE: top 4 LLM layers (12-15) are fine-tuned during training, but inference is the same.

Pooling: masked mean-pool over non-padding tokens (using backbone_attention_mask)

Requires:
    PYTHONPATH must point to gr00t_1p6/Isaac-GR00T

Usage:
    PYTHONPATH=$WORK/gr00t_1p6/Isaac-GR00T python extract_features_groot_n16.py \
        --ckpt $WORK/GR00T-N1.6-bridge \
        --data_dir ./cknna_data \
        --output_dir ./cknna_data/groot-n16-bridge
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
_GROOT_N16 = os.environ.get("ISAAC_GROOT_N16_DIR", os.path.join(_PROJECT_ROOT, "repos", "Isaac-GR00T-N16"))
if _GROOT_N16 not in sys.path:
    sys.path.insert(0, _GROOT_N16)

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

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
            mask[start:start + len(task_ids)] = 1
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


def build_bridge_obs(image_np, task_description):
    """Build N1.6-format observation dict for a single sample (B=1, T=1).

    State is dummy zeros since it only enters the action head, not the backbone.
    """
    return {
        "video": {
            "image_0": image_np[np.newaxis, np.newaxis],
        },
        "state": {
            "x": np.zeros((1, 1, 1), dtype=np.float32),
            "y": np.zeros((1, 1, 1), dtype=np.float32),
            "z": np.zeros((1, 1, 1), dtype=np.float32),
            "roll": np.zeros((1, 1, 1), dtype=np.float32),
            "pitch": np.zeros((1, 1, 1), dtype=np.float32),
            "yaw": np.zeros((1, 1, 1), dtype=np.float32),
            "pad": np.zeros((1, 1, 1), dtype=np.float32),
            "gripper": np.zeros((1, 1, 1), dtype=np.float32),
        },
        "language": {
            "annotation.human.action.task_description": [[task_description]],
        },
    }


def process_single_obs(policy, obs):
    """Run one observation through the N1.6 processor pipeline.

    Replicates what Gr00tPolicy._get_action does internally up to collation.
    """
    unbatched = policy._unbatch_observation(obs)
    vla_step = policy._to_vla_step_data(unbatched[0])
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
    processed = policy.processor(messages)
    collated = policy.collate_fn([processed])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    return collated["inputs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to GR00T N1.6 checkpoint directory (e.g. nvidia/GR00T-N1.6-bridge)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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

    print(f"Loading GR00T N1.6: {args.ckpt}")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        model_path=args.ckpt,
        device=args.device,
    )
    model = policy.model
    hidden_size = 2048

    print(f"  select_layer: {model.config.select_layer}")
    print(f"  backbone_embedding_dim: {model.config.backbone_embedding_dim}")
    print(f"  use_alternate_vl_dit: {model.config.use_alternate_vl_dit}")
    print(f"  Samples to process: {num_samples}")

    from transformers import AutoTokenizer
    eagle_dir = os.path.join(
        os.path.dirname(__import__('gr00t').__file__),
        "model", "modules", "nvidia", "Eagle-Block2A-2B-v2",
    )
    backbone_tokenizer = AutoTokenizer.from_pretrained(
        eagle_dir, trust_remote_code=True, use_fast=False
    )
    image_token_index = model.backbone.model.config.image_token_index
    print(f"  image_token_index: {image_token_index}")

    feats_imgtext_list = []
    feats_img_list = []
    feats_txt_list = []

    t0 = time.time()
    for i in range(num_samples):
        img = np.array(
            Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        )
        task = task_descriptions[i]

        obs = build_bridge_obs(img, task)
        inputs = process_single_obs(policy, obs)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            backbone_inputs, _ = model.prepare_input(inputs)
            backbone_outputs = model.backbone(backbone_inputs)

        backbone_features = backbone_outputs["backbone_features"]
        backbone_mask = backbone_outputs["backbone_attention_mask"]
        image_mask = backbone_outputs["image_mask"]

        feat_imgtext = masked_mean_pool(backbone_features, backbone_mask).squeeze(0).cpu()

        feat_img = masked_mean_pool(backbone_features, image_mask.to(backbone_mask.dtype)).squeeze(0).cpu()

        input_ids = backbone_inputs["input_ids"]
        img_positions = image_mask.bool()[0]
        text_positions = (~img_positions).nonzero(as_tuple=True)[0]
        text_only_ids = input_ids[0][text_positions]
        text_task_mask = build_task_mask(text_only_ids, backbone_tokenizer, task)
        task_mask = torch.zeros(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        matched = text_positions[text_task_mask.bool()]
        if matched.numel() > 0:
            task_mask[matched] = 1
        task_mask = task_mask.unsqueeze(0)
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
        "model": "GR00T-N1.6-bridge",
        "checkpoint_path": args.ckpt,
        "hidden_size": hidden_size,
        "num_samples": num_samples,
        "outputs": ["feats_A.pt", "feats_A_img.pt", "feats_A_txt.pt"],
        "image_token_index": image_token_index,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata.json"), "w") as f:
        json.dump(extraction_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== GR00T N1.6 Phase 2 Complete ===")
    print(f"  Time: {elapsed/60:.1f} min  ({elapsed/num_samples:.2f} s/sample)")


if __name__ == "__main__":
    main()

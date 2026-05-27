"""
Extract decomposed SpatialVLA features for CKNNA ablation.

SpatialVLA's image tokens are: O_3d = X + P'
  X  = SigLIP vision encoder output (2D semantic features)
  P' = Ego3DPositionEmbeddingMLP(ZoeDepth xyz) (3D position embeddings)

This script hooks model.vision_tower and model.position_embedding_3d to
capture X and P' separately, then mean-pools each over image patches.

Produces three feature files (all pre-LM, in SigLIP hidden dimension):
  feats_A_X.pt    -- mean-pooled SigLIP features (semantic only)
  feats_A_P.pt    -- mean-pooled 3D position embeddings (depth only)
  feats_A_XP.pt   -- mean-pooled X + P' (combined, pre-LM)

Usage:
  python extract_features_spatialvla_decomposed.py \
      --ckpt IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \
      --data_dir ./cknna_data \
      --output_dir ./cknna_data/spatialvla-sft-bridge
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig/1.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    print("Loading SpatialVLA: %s" % args.ckpt)
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.ckpt,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(args.device)

    print("  use_vision_zoe: %s" % getattr(model.config, "use_vision_zoe", "N/A"))
    print("  Samples: %d" % num_samples)

    _buf = {}

    def _hook_vision_tower(module, inp, out):
        _buf["X"] = out.last_hidden_state.detach()

    def _hook_position_embedding_3d(module, inp, out):
        _buf["P"] = out.detach()

    h1 = model.vision_tower.register_forward_hook(_hook_vision_tower)
    h2 = model.position_embedding_3d.register_forward_hook(_hook_position_embedding_3d)

    feats_X_list = []
    feats_P_list = []
    feats_XP_list = []

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, "%06d.png" % i)).convert("RGB")
        task = task_descriptions[i]

        inputs = processor(
            images=[img],
            text=task,
            unnorm_key=args.unnorm_key,
            return_tensors="pt",
            do_normalize=False,
        )
        inputs = inputs.to(dtype=torch.bfloat16, device=args.device)

        _buf.clear()
        with torch.inference_mode():
            model(**inputs)

        X = _buf["X"]    # (1, num_patches, d_siglip)
        P = _buf["P"]    # (1, num_patches, d_siglip)

        feat_X = X.float().mean(dim=1).squeeze(0).cpu()
        feat_P = P.float().mean(dim=1).squeeze(0).cpu()
        feat_XP = (X + P).float().mean(dim=1).squeeze(0).cpu()

        feats_X_list.append(feat_X)
        feats_P_list.append(feat_P)
        feats_XP_list.append(feat_XP)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print("  [%d/%d]  X=%s  P=%s  rate=%.1f/s  ETA=%.1fmin"
                  % (i + 1, num_samples, tuple(feat_X.shape), tuple(feat_P.shape),
                     rate, eta / 60))

    h1.remove()
    h2.remove()

    for name, flist in [("feats_A_X", feats_X_list),
                        ("feats_A_P", feats_P_list),
                        ("feats_A_XP", feats_XP_list)]:
        t = torch.stack(flist)
        p = os.path.join(args.output_dir, "%s.pt" % name)
        torch.save(t, p)
        print("  Saved %s  shape=%s" % (p, tuple(t.shape)))

    elapsed = time.time() - t0
    print("\nDone in %.1f min  (%.2f s/sample)" % (elapsed / 60, elapsed / num_samples))


if __name__ == "__main__":
    main()

"""
7-feature extraction for X-VLA (Florence2 / BART encoder).

Extracts all 7 features in a SINGLE forward pass per sample:
  P1-V   vit_raw:      DaViT output before projection (before image_proj_norm)
                       Hooked via custom forward hook on `vlm.image_proj_norm`'s INPUT.
  P2-V   pre_llm:      Post-projection image embeds (after image_proj_norm).
                       Hooked via forward hook on `vlm.image_proj_norm` OUTPUT.
  P2-T   pre_llm_txt:  BART encoder layers[0] forward_pre_hook, text-only mask.
  P2-VT  pre_llm_vt:   BART encoder layers[0] forward_pre_hook, all valid tokens.
  P3-V   img:          BART encoder final hidden state, image positions only.
  P3-T   txt:          BART encoder final hidden state, text positions only (non-padding).
  P3-VT  imgtext:      BART encoder final hidden state, all valid tokens mean-pooled.

Token layout in the merged BART encoder input (one sample, 1 image view):
  positions [0 : N_img]  → image tokens (always valid)
  positions [N_img : N_img + L]  → text tokens (mask out padding via input_ids != pad)

Usage:
  XVLA_ROOT=/lambda/nfs/vla/cknna_project/repos/X-VLA \\
  python extractors/extract_7feat_xvla.py \\
      --ckpt /lambda/nfs/vla/cknna_project/checkpoints/xvla-pt \\
      --data_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid \\
      --output_dir /lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_7feat/xvla-pt \\
      [--max_samples 5]
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
from PIL import Image

torch.backends.cudnn.enabled = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XVLA_ROOT = os.environ.get("XVLA_ROOT", os.path.join(_PROJECT_ROOT, "repos", "X-VLA"))
if XVLA_ROOT not in sys.path:
    sys.path.insert(0, XVLA_ROOT)

from models.modeling_xvla import XVLA
from models.configuration_xvla import XVLAConfig
from models.processing_xvla import XVLAProcessor
from safetensors.torch import load_file


def masked_mean_pool(hidden_states, mask):
    h = hidden_states.float()
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def load_xvla(ckpt_dir, device):
    """Load X-VLA from a local checkpoint directory."""
    print(f"Loading X-VLA from {ckpt_dir}")
    config = XVLAConfig.from_pretrained(ckpt_dir)
    with torch.device("cpu"):
        model = XVLA(config)
    st_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    state_dict = {}
    for f in st_files:
        state_dict.update(load_file(f))
    res = model.load_state_dict(state_dict, strict=False)
    # encoder.embed_tokens shares weights with model.shared and is filled by tying
    real_missing = [k for k in res.missing_keys if "embed_tokens" not in k]
    if real_missing:
        print(f"  WARNING: {len(real_missing)} unexpected missing keys: {real_missing[:5]}")
    if res.unexpected_keys:
        print(f"  WARNING: {len(res.unexpected_keys)} unexpected keys: {res.unexpected_keys[:5]}")
    model = model.eval().to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def load_processor(ckpt_dir):
    try:
        processor = XVLAProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
    except Exception:
        from transformers import AutoImageProcessor, AutoTokenizer
        image_processor = AutoImageProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        processor = XVLAProcessor(image_processor=image_processor, tokenizer=tokenizer)

    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token_id = 1
        tok.pad_token = tok.convert_ids_to_tokens(1)
    if tok.eos_token_id is None:
        tok.eos_token_id = 2
        tok.eos_token = tok.convert_ids_to_tokens(2)
    if tok.bos_token_id is None:
        tok.bos_token_id = 0
        tok.bos_token = tok.convert_ids_to_tokens(0)
    return processor


def extract_all_7(model, processor, img, instruction, device,
                  N_img, L, pad_token_id):
    """Extract all 7 features in one forward pass.

    Strategy: register 3 hooks, then call forward_vlm() once.
      - hook A on `vlm.image_proj_norm`: input gives P1-V (DaViT output),
        output gives P2-V (post-projection image features).
      - hook B on `vlm.language_model.model.encoder.layers[0]` forward_pre_hook:
        captures merged inputs_embeds → P2-VT (all) and P2-T (text mask).
    The output of forward_vlm() (= encoder final hidden state) yields P3-V/T/VT.
    """
    captures = {}

    inputs = processor(images=[[img]], language_instruction=[instruction])
    input_ids = inputs["input_ids"].to(device)        # [1, L]
    image_input = inputs["image_input"].to(device)
    image_mask = inputs["image_mask"].to(device)

    text_ids_1d = input_ids[0]
    text_mask_1d = (text_ids_1d != pad_token_id).float()  # [L]

    vm = model.vlm
    image_proj_norm = vm.image_proj_norm
    encoder_layer0 = vm.language_model.model.encoder.layers[0]

    # ── Hook A: image_proj_norm input/output ────────────────────────────────
    def hook_image_proj_norm(module, inputs_tup, output):
        # input is the pre-norm value (DaViT output × image_projection)
        x_in = inputs_tup[0]
        # x_in shape: (batch_total, T*N_img_per_view, dim_projection)
        # For 1 image view, batch_total=1
        captures["p1_v"] = x_in.detach().float().mean(dim=1).squeeze(0).cpu()
        # output is the post-norm value
        captures["p2_v"] = output.detach().float().mean(dim=1).squeeze(0).cpu()

    h_proj = image_proj_norm.register_forward_hook(hook_image_proj_norm)

    # ── Hook B: BART encoder layers[0] pre-hook ─────────────────────────────
    def hook_layer0_pre(module, args, kwargs):
        # BART layer signature: (hidden_states, attention_mask, ...)
        if "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        else:
            hidden = args[0]
        h = hidden.detach().float()  # [1, T_enc, D]
        # P2-VT: mean over all valid tokens
        # The encoder receives [image (N_img) | text (L)], so build full mask
        full_mask = torch.cat([
            torch.ones(1, N_img, device=h.device),
            text_mask_1d.unsqueeze(0).to(h.device),
        ], dim=1)  # [1, N_img+L]
        captures["p2_vt"] = masked_mean_pool(h, full_mask).squeeze(0).cpu()
        # P2-T: mean over text tokens only (skip image positions)
        text_only_mask = torch.cat([
            torch.zeros(1, N_img, device=h.device),
            text_mask_1d.unsqueeze(0).to(h.device),
        ], dim=1)  # [1, N_img+L]
        captures["p2_t"] = masked_mean_pool(h, text_only_mask).squeeze(0).cpu()

    h_layer0 = encoder_layer0.register_forward_pre_hook(
        hook_layer0_pre, with_kwargs=True
    )

    try:
        with torch.inference_mode():
            enc = model.forward_vlm(input_ids, image_input, image_mask)
    finally:
        h_proj.remove()
        h_layer0.remove()

    # P3-* from final encoder output
    vlm_feat = enc["vlm_features"]  # [1, N_img + L, D]
    img_hidden = vlm_feat[:, :N_img, :].float()
    captures["p3_v"] = img_hidden.mean(dim=1).squeeze(0).cpu()

    text_hidden = vlm_feat[:, N_img:, :].float()  # [1, L, D]
    captures["p3_t"] = masked_mean_pool(
        text_hidden, text_mask_1d.unsqueeze(0)
    ).squeeze(0).cpu()

    full_mask = torch.cat([
        torch.ones(1, N_img, device=vlm_feat.device),
        text_mask_1d.unsqueeze(0),
    ], dim=1)
    captures["p3_vt"] = masked_mean_pool(vlm_feat.float(), full_mask).squeeze(0).cpu()

    return captures


def main():
    parser = argparse.ArgumentParser(description="7-feature extraction for X-VLA-Pt.")
    parser.add_argument("--ckpt", required=True, help="Local X-VLA checkpoint dir")
    parser.add_argument("--data_dir", required=True, help="CKNNA data dir")
    parser.add_argument("--output_dir", required=True, help="Where to save 7 .pt files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_samples = metadata["num_samples"]
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    task_descriptions = metadata["task_descriptions"]
    images_dir = os.path.join(args.data_dir, "images")

    model = load_xvla(args.ckpt, args.device)
    processor = load_processor(args.ckpt)
    pad_token_id = processor.tokenizer.pad_token_id

    # Probe N_img by feeding a dummy image
    dummy_img = Image.new("RGB", (224, 224))
    dummy_inputs = processor(images=[[dummy_img]], language_instruction=["test"])
    dummy_inputs = {k: v.to(args.device) for k, v in dummy_inputs.items()}
    with torch.inference_mode():
        dummy_enc = model.forward_vlm(
            dummy_inputs["input_ids"],
            dummy_inputs["image_input"],
            dummy_inputs["image_mask"],
        )
    T_enc = dummy_enc["vlm_features"].shape[1]
    L = dummy_inputs["input_ids"].shape[1]
    N_img = T_enc - L
    D_text = dummy_enc["vlm_features"].shape[2]
    print(f"  N_img={N_img}, L_text={L}, T_enc={T_enc}, D_text={D_text}")
    del dummy_inputs, dummy_enc

    keys = ["p1_v", "p2_v", "p2_t", "p2_vt", "p3_v", "p3_t", "p3_vt"]
    accum = {k: [] for k in keys}

    t0 = time.time()
    for i in range(num_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        instruction = task_descriptions[i]

        feats = extract_all_7(
            model, processor, img, instruction, args.device,
            N_img, L, pad_token_id,
        )

        for k in keys:
            if k not in feats:
                raise RuntimeError(f"Feature {k} missing at sample {i}")
            accum[k].append(feats[k])

        if (i + 1) % 500 == 0:
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            shapes = "  ".join(f"{k}={tuple(feats[k].shape)}" for k in keys)
            print(
                f"  [{i+1}/{num_samples}]  {shapes}  rate={rate:.1f}/s  ETA={eta/60:.1f}min",
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
    }

    print(f"\nSaving {len(keys)} feature tensors:")
    for k in keys:
        tensor = torch.stack(accum[k])
        out_path = os.path.join(args.output_dir, file_map[k])
        torch.save(tensor, out_path)
        print(f"  {file_map[k]:<28} shape={tuple(tensor.shape)}")

    meta = {
        "model": args.ckpt,
        "family": "X-VLA (Florence2/BART)",
        "type": "all_7_features",
        "num_samples": num_samples,
        "N_img": N_img,
        "L_text": L,
        "feature_dims": {k: int(accum[k][0].shape[-1]) for k in keys},
        "file_map": file_map,
    }
    with open(os.path.join(args.output_dir, "extraction_metadata_all7.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed/60:.1f} min total, {num_samples/elapsed:.2f} samples/s")


if __name__ == "__main__":
    main()

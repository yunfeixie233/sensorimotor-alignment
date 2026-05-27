# Vision Encoder Checkpoints for CKNNA

Standalone vision encoder checkpoints corresponding to the **pre-VLM-training** versions
of the ViTs used inside the 8 raw VLMs tested in the CKNNA 7-feature experiment.

## Inventory

| Dir | HF Model ID | Used In | Layers | Dim | Heads | Patch | Img | Size |
|-----|------------|---------|--------|-----|-------|-------|-----|------|
| `clip_vit_l14/` | `openai/clip-vit-large-patch14` | KosMos-2 | 24 | 1024 | 16 | 14 | 224 | 6.4G |
| `siglip_so400m_patch14/` | `google/siglip-so400m-patch14-224` | PaliGemma-1/2 | 27 | 1152 | 16 | 14 | 224 | 3.3G |
| `siglip2_large_patch16/` | `google/siglip2-large-patch16-384` | Qwen3-VL-2B/4B | 24 | 1024 | 16 | 16 | 384 | 3.4G |
| `siglip2_so400m_patch16/` | `google/siglip2-so400m-patch16-384` | Qwen3-VL-8B | 27 | 1152 | 16 | 16 | 384 | 4.3G |
| `qwen25vl_vit/` | N/A (extract from VLM) | Qwen2.5-VL-3B/7B | 32 | 1280 | 16 | 14 | dyn | — |

**Note**: Each HF dir includes both vision and text encoder weights.
To load vision-only, use `CLIPVisionModel` / `SiglipVisionModel` from transformers.

## Verification (all configs confirmed 2026-04-09)

### CLIP ViT-L/14 → KosMos-2
- KosMos-2 uses **frozen original OpenAI CLIP ViT-L/14** (not fine-tuned)
- Confirmed: KosMos-2 training script uses `--visual-model-name ViT-L-14`
- All 8 vision_config parameters match `openai/clip-vit-large-patch14`
- Source: `unilm/kosmos-2/open_clip/src/open_clip/model_configs/ViT-L-14.json`

### SigLIP-SO400M/14 → PaliGemma-1/2
- PaliGemma uses **frozen SigLIP-So400m** (not fine-tuned during PaliGemma pretraining)
- Confirmed: PaliGemma paper (arXiv 2407.07726) states vision component is frozen pretrained SigLIP
- This checkpoint IS the exact pre-VLM-training baseline
- Architecture: Shape-Optimized ViT from SoViT (arXiv 2305.13035)

### SigLIP-2 Large/SO400M patch16 → Qwen3-VL
- Qwen3-VL paper (arXiv 2511.21631): "initialized from official pretrained checkpoints"
- Qwen3-VL then does **continued pretraining** of the ViT (adding 2D-RoPE, dynamic resolution)
- So these checkpoints are the **pre-continued-pretraining** baseline (before Qwen modifications)
- Match: Qwen3-VL-2B/4B config (24L/1024D/patch16) = `siglip2-large-patch16-384`
- Match: Qwen3-VL-8B config (27L/1152D/patch16) = `siglip2-so400m-patch16-384`

### Qwen2.5-VL ViT → No standalone checkpoint
- Qwen2.5-VL paper (arXiv 2502.13923): "trained from scratch using DataComp"
- The ViT was CLIP-pretrained by the Qwen team, then integrated into the VLM
- No standalone checkpoint was ever released
- Use `qwen25vl_vit/extract_vit_weights.py` to extract from the full VLM

## Quick start: loading vision-only models

```python
# CLIP ViT-L/14
from transformers import CLIPVisionModel, CLIPImageProcessor
model = CLIPVisionModel.from_pretrained("vision_encoder/clip_vit_l14")
proc = CLIPImageProcessor.from_pretrained("vision_encoder/clip_vit_l14")
out = model(**proc(images=img, return_tensors="pt"), output_hidden_states=True)
# out.last_hidden_state: (B, 257, 1024)  [256 patches + CLS]
# out.pooler_output: (B, 1024)

# SigLIP-SO400M/14
from transformers import SiglipVisionModel, AutoProcessor
model = SiglipVisionModel.from_pretrained("vision_encoder/siglip_so400m_patch14")
proc = AutoProcessor.from_pretrained("vision_encoder/siglip_so400m_patch14")
out = model(**proc(images=img, return_tensors="pt"), output_hidden_states=True)
# out.last_hidden_state: (B, 256, 1152)  [no CLS token]
# out.pooler_output: (B, 1152)  [learned attention pooling]

# SigLIP-2 Large patch16 (for Qwen3-VL-2B/4B comparison)
model = SiglipVisionModel.from_pretrained("vision_encoder/siglip2_large_patch16")
proc = AutoProcessor.from_pretrained("vision_encoder/siglip2_large_patch16")
out = model(**proc(images=img, return_tensors="pt"), output_hidden_states=True)
# out.last_hidden_state: (B, 576, 1024)  [24x24 patches at 384px]

# SigLIP-2 SO400M patch16 (for Qwen3-VL-8B comparison)
model = SiglipVisionModel.from_pretrained("vision_encoder/siglip2_so400m_patch16")
proc = AutoProcessor.from_pretrained("vision_encoder/siglip2_so400m_patch16")
out = model(**proc(images=img, return_tensors="pt"), output_hidden_states=True)
# out.last_hidden_state: (B, 576, 1152)  [24x24 patches at 384px]

# Qwen2.5-VL ViT (requires full VLM download)
# Run: python qwen25vl_vit/extract_vit_weights.py --model_path Qwen/Qwen2.5-VL-3B-Instruct
```

## Repos

| Dir | Repo | Description |
|-----|------|-------------|
| `repos/open_clip/` | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | OpenCLIP: CLIP training + inference code |
| `repos/big_vision/` | [google-research/big_vision](https://github.com/google-research/big_vision) | Big Vision: SigLIP/PaliGemma training code (JAX) |

## Relationship diagram

```
OpenAI CLIP ViT-L/14 ──────────────────────────────────── KosMos-2 (frozen)
                                                              │
SigLIP-SO400M/14 (v1) ─────────────────────────────────── PaliGemma-1/2 (frozen)
        │                                                     │
        └── SigLIP-2 SO400M/14 (v2, improved training) ── SpatialVLA, Pi0, GR00T (frozen)
                │
                ├── SigLIP-2 Large patch16 ── continued pretrain ── Qwen3-VL-2B/4B ViT
                └── SigLIP-2 SO400M patch16 ── continued pretrain ── Qwen3-VL-8B ViT

DataComp (from scratch) ── CLIP pretrain ── Qwen2.5-VL ViT (no standalone checkpoint)
```

## Papers

- CLIP: Radford et al., 2021 (OpenAI)
- SigLIP: arXiv 2303.15343 (Zhai et al., 2023)
- SoViT (Shape-Optimized ViT): arXiv 2305.13035 (Alabdulmohsin et al., NeurIPS 2023)
- SigLIP-2: arXiv 2502.14786 (Tschannen et al., 2025)
- PaliGemma: arXiv 2407.07726 (Beyer et al., 2024)
- KosMos-2: arXiv 2306.14824 (Peng et al., 2023)
- Qwen2.5-VL: arXiv 2502.13923 (Bai et al., 2025)
- Qwen3-VL: arXiv 2511.21631 (Bai et al., 2025)

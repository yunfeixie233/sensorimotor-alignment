# Models and checkpoint sources

[README](../README.md) · [VLA recipe](reproduce-vla.md) · [VLM recipe](reproduce-vlm.md) · [WAM coverage](reproduce-wam.md)

This table preserves the original family-level catalog. For the exact VLA model IDs, output-directory names, and success rates, use [widowx_sr.csv](../record/fig_3.1_pa_predicts_sr_1x3/data/widowx_sr.csv). The VLM guide lists the eight checkpoint IDs and family flags.

The current paper's six-WAM cohort adds Fast-WAM, FlowWAM, and Kairos to the three released models below. The paper also evaluates five real-robot policies: MolmoB0T, MolmoB0T-Img, MolmoB0T-Pi0, π₀-DROID, and π₀.₅-DROID. Those experiments do not yet have complete recipes in this release.

| Cohort                       | Family                  | # models                                                  | HuggingFace                                                                                            |
|------------------------------|-------------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **VLAs** (§3.1, n=12)        | StarVLA                 | 5 (Qwen-GR00T / Qwen-OFT / Qwen3VL variants on Bridge / RT-1) | [StarVLA org](https://huggingface.co/StarVLA)                                                       |
|                              | CogACT                  | 3 (Small / Base / Large)                                  | [CogACT org](https://huggingface.co/CogACT)                                                            |
|                              | GR00T                   | 2 (N1.5, N1.6 on Bridge)                                  | [GR00T-N1.6 card](https://huggingface.co/nvidia/GR00T-N1.6-bridge)                                     |
|                              | Pi0 (lerobot port)      | 1                                                         | [HaomingSong/lerobot-pi0-bridge](https://huggingface.co/HaomingSong/lerobot-pi0-bridge)                |
|                              | SpatialVLA              | 1                                                         | [IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge) |
| **Raw VLMs** (§3.1, n=8)     | KosMos-2                | 1                                                         | [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224)                |
|                              | PaliGemma 1 / 2         | 2                                                         | [google/paligemma2-3b-mix-224](https://huggingface.co/google/paligemma2-3b-mix-224)                      |
|                              | Qwen2.5-VL              | 2 (3B, 7B)                                                | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)                      |
|                              | Qwen3-VL                | 3 (2B, 4B, 8B)                                            | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)                          |
| **Released WAM subset** (n=3)         | Motus                   | 1                                                         | [project page](https://github.com/jasper0314-huang/Motus)                                              |
|                              | LingBot-VA              | 1                                                         | [project page](https://github.com/LingBot-AI)                                                          |
|                              | Vidar                   | 1                                                         | [project page](https://github.com/SgtVincent/Vidar)                                                    |
| **Historical standalone catalog** (listed as n=23) | CLIP-L / OpenCLIP-L / EVA-CLIP-L / SigLIP-2 (contrastive) | 6 (1 + 1 + 1 + 3 SigLIP-2 sizes) | family pages on HF (`openai`, `laion`, `BAAI`, `google`) |
|                              | DINO v1-B / DINOv2 / iBOT-B (self-distillation) | 6 (1 DINO v1 + 4 DINOv2 sizes + 1 iBOT) | `facebook` HF org |
|                              | MAE (reconstruction)    | 3 (Base / Large / Huge)                                   | `facebook` HF org |
|                              | I-JEPA / V-JEPA / V-JEPA 2 / V-JEPA 2.1 (joint-embedding predictive) | 8 (1 + 2 + 3 + 2)  | `facebook` HF org |

The original “46 models” total combined 12 VLAs, 8 VLMs, 3 WAMs, and a catalog of 23 standalone encoders. It describes the earlier release, not the current paper's experimental coverage. The historical JEPA catalog lists eight configurations, while the explicit command in the retained recipe selects seven model IDs. Check the selection before comparing counts.

The standalone/ImageNet experiment remains available under [diagnostics](diagnostics.md#historical-vision-encoder-experiment). It is separate from the current paper's experiment on 14 frozen vision backbones under policy learning.

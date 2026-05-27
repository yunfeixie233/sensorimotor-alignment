# BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligence

<div align="center" class="authors">
    <a href="https://scholar.google.com/citations?user=pfXQwcQAAAAJ&hl=en" target="_blank">Xuewu Lin</a>,
    <a href="https://wzmsltw.github.io/" target="_blank">Tianwei Lin</a>,
    <a href="https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en" target="_blank">Lichao Huang</a>,
    <a href="https://openreview.net/profile?id=~HONGYU_XIE2" target="_blank">Hongyu Xie</a>,
    <a href="https://scholar.google.com/citations?user=HQfc8TEAAAAJ&hl=en" target="_blank">Zhizhong Su</a>
</div>

<div align="center" style="line-height: 3;">
  <a href="https://linxuewu.github.io/BIP3D-page/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Homepage-BIP3D-green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Lin_BIP3D_Bridging_2D_Images_and_3D_Perception_for_Embodied_Intelligence_CVPR_2025_paper.html" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-CVPR2025-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/HorizonRobotics/BIP3D_Tiny_Det" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Prerequest

### Prepare the data

Download the [EmbodiedScan dataset](https://github.com/OpenRobotLab/EmbodiedScan) and create symbolic links.

```bash
cd projects/bip3d_grounding
mkdir data
ln -s path/to/embodiedscan ./data/embodiedscan
```

Download datasets [ScanNet](https://github.com/ScanNet/ScanNet), [3RScan](https://github.com/WaldJohannaU/3RScan), [Matterport3D](https://github.com/niessner/Matterport), and optionally download [ARKitScenes](https://github.com/apple/ARKitScenes). Adjust the data directory structure as follows:

```text
projects/bip3d_grounding
└──data
    ├──embodiedscan
    │   ├──embodiedscan_infos_train.pkl
    │   ├──embodiedscan_infos_val.pkl
    │   ...
    ├──3rscan
    │   ├──00d42bed-778d-2ac6-86a7-0e0e5f5f5660
    │   ...
    ├──scannet
    │   └──posed_images
    ├──matterport3d
    │   ├──17DRP5sb8fy
    │   ...
    └──arkitscenes
        ├──Training
        └──Validation
```

### Prepare pre-trained weights

Download the required Grounding-DINO pre-trained weights: [Swin-Tiny](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth) and [Swin-Base](https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth).

```bash
cd projects/bip3d_grounding
mkdir ckpt

# Swin-Tiny
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth -O ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth
python tools/ckpt_rename.py ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth --output ./ckpt

# Swin-Base
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth -O ckpt/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
python tools/ckpt_rename.py ckpt/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth --output ./ckpt
```

Download bert config and pretrain weights from [huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main).

```text
projects/bip3d_grounding
└──ckpt
    ├──groundingdino_swint_ogc_mmdet-822d7e9d.pth
    ├──groundingdino_swint_ogc_mmdet-822d7e9d-rename.pth  # generated after rename
    ├──groundingdino_swinb_cogcoor_mmdet-55949c9c.pth
    ├──groundingdino_swinb_cogcoor_mmdet-55949c9c-rename.pth  # generated after rename
    └──bert-base-uncased
        ├──config.json
        ├──tokenizer_config.json
        ├──tokenizer.json
        ├──pytorch_model.bin
        ...
```

### Generate anchors by K-means

```bash
cd projects/bip3d_grounding

mkdir anchor_files
python3 tools/anchor_bbox3d_kmeans.py data/embodiedscan/embodiedscan_infos_train.pkl
```

You can also download the anchor file we provide.

### Install BIP3D Dependency

#### Install pytorch3d

This project requires pytorch3d. Since the specific PyTorch3D build (CPU, CUDA 11.x, CUDA 12.x, etc.) depends on your system, please install it manually first by following the [official instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

#### From pip

```bash
pip install robo_orchard_lab[bip3d]
```

#### From source

```bash
make install EXTRA_ARGS=[bip3d]
```

### (Optional) Compile Deformable Aggregation

To reduce startup time, you can pre-compile the required CUDA operations. Note: This is not mandatory.

```bash
cd robo_orchard_lab/ops/deformable_aggregation
pip install -e . --no-deps --config-settings editable_mode=compat
```

## Run locally

* Required free GPU memory >= 24GB to running the preset training config

```bash
cd projects/bip3d_grounding
CONFIG=config_bip3d_det.py

# train with single-gpu
python3 train.py --config ${CONFIG}

# train with multi-gpu
accelerate launch \
    --multi-gpu \
    --num-processes 8 \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    train.py --config ${CONFIG}

# eval with single-gpu
python3 train.py --config ${CONFIG} --eval_only

# eval with multi-gpu
accelerate launch \
    --multi-gpu \
    --num-processes 8 \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    train.py --config ${CONFIG} --eval_only
```

## Results on EmbodiedScan Benchmark

### detection

|Model | Overall | Head | Common | Tail | Small | Medium | Large | ScanNet | 3RScan | MP3D |
|  :----  | :---: |:---: | :---: | :---: | :---:| :---:|:---:|:---: | :---: | :----: |
|BIP3D | 0.2500 |0.3248|0.2179|0.2042|0.0933|0.2582|0.1569|0.2786|0.3987|0.1238|


## :handshake: Acknowledgement
[EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan)

[mmdet](https://github.com/open-mmlab/mmdetection)

[mmdet3d](https://github.com/open-mmlab/mmdetection3d)

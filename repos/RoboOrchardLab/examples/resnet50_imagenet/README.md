# Introduction

Training and evaluation scripts for resnet50 on the ImageNet dataset.

This is just an example of how to use the `robo_orchard_lab` library to train a model. Not all
features are implemented, and **NO performance guarantees** are made!

# Usage

Use the `train.py` script to train the model. The script accepts several command-line arguments to customize the training process.

You can use the `--help` flag to see all available options:

```bash
python3 train.py --help
```

## Single GPU Training

This example use python backend directly and will use GPU 0 by default.

```bash
python3 train.py --dataset.data_root /path/to/imagenet
```

## Multi GPU Training

```bash
accelerate launch \
    --multi-gpu \
    --num-processes 4 \
    --gpu-ids 0,1,2,3 \
    train.py --dataset.data_root /path/to/imagenet
```

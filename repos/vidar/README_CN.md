# Vidar & Vidarc Embodied Video Fundation Model

<img src='examples/vidar_logo.png' width=90>

<a href='https://arxiv.org/abs/2507.12898'><img src='https://img.shields.io/badge/arXiv-2507.12898-b31b1b.svg'></a>
<a href='https://openreview.net/forum?id=gsvjCTIYPb'><img src='https://img.shields.io/badge/openreview-gsvjCTIYPb-b31b1b.svg'></a>
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://embodiedfoundation.github.io/vidar_anypos)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Xiang-cd/vidar)


## 📝 Table of Contents
- [🔥新闻(News)](#-新闻-news)
- [📖 简介 (Introduction)](#-简介-introduction)
- [🔧 环境配置 (Installation)](#-环境配置-installation)
- [⚡ 推理 (Inference)](#-推理-inference)
- [🖊️ 引用 (Citation)](#%EF%B8%8F-引用-citation)
- [🙏 致谢 (Acknowledgements)](#-致谢-acknowledgements)

## 🔥 新闻 (News)
- **[2025.12]**: 代码库初版发布。
- **[2025.07]**: Vidar paper 在 [arXiv](https://arxiv.org/abs/2507.12898) 上线。

## 📖 简介 (Introduction)
Vidar：面向低样本通用操作的统一具身视频基座模型
Vidar 是一个统一的**具身视频扩散模型**，它借助互联网级视频先验与跨平台机器人轨迹数据，解决机器人操作中数据稀缺、平台适配难的核心问题。
Vidar 采用 “视频生成 + 动作解码” 两阶段策略，整合了两大核心组件 —— 具身视频扩散模型与掩码逆动力学模型（MIDM）；同时通过带物理感知重排序的测试时缩放策略，实现对未知任务、未知背景及未知摄像头布局的稳健泛化。
此外，Vidar 通过统一观测空间（整合多视角图像、机器人类型、摄像头布局、任务指令）对齐跨平台异构数据，并采用 “通用预训练→具身领域预训练→目标域微调” 三阶段训练流程，从海量无标注视频中捕捉物理一致性与时序连贯性，最终仅需约 20 分钟人类演示数据，即可在新机器人平台上实现低样本适配。

Vidarc：面向闭环控制的自回归视频基座模型
Vidarc 是一款专为机器人闭环控制设计的新型**自回归具身视频扩散模型**，旨在解决数据稀缺场景下机器人操作的高延迟、grounding不足两大核心痛点。
它通过融合自回归视频生成与掩码逆动力学模型，将环境实时反馈融入推理流程，实现低延迟、高精度的闭环控制，同时在未知机器人平台与动态环境中保持强泛化性与误差修正能力。



## 🔧 环境配置 (Installation)
执行以下命令
```bash
conda env create --file vidar.yaml
conda activate vidar
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


## ⚡ 推理 (Inference)

下载预训练模型权重：
[Wan2.2](https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B), 并放置于`Wan2.2-TI2V-5B`
[Vidar/Vidarc](https://huggingface.co/Xiang-cd/vidar)，并放置于`vidar_ckpts`

### 使用example推理
```bash
# 使用vidarc推理
output_dir="output/test"
python generate_causal.py \
            --task ti2v-5B \
            --size "640*736" \
            --ckpt_dir ./Wan2.2-TI2V-5B \
            --convert_model_dtype \
            --pt_dir vidar_ckpts/vidarc.pt \
            --dataset_json examples/robotwin_example.json \
            --output_dir "$output_dir"

# 使用vidar推理
python generate.py \
    --task ti2v-5B \
    --size "640*736" \
    --ckpt_dir ./Wan2.2-TI2V-5B \
      --convert_model_dtype \
      --pt_dir vidar_ckpts/vidar.pt \
    --dataset_json examples/robotwin_example.json \
    --output_dir "$output_dir"
```

### Robotwin 测评

查看 [eval code]( https://github.com/thu-ml/vidar-robotwin.git), 并配置测评环境。
```bash
# clone related code
git clone https://github.com/thu-ml/vidar-robotwin.git

# read related README at vidar-robotwin dir.
```



## 🖊️ 引用 (Citation)
如果您觉得本项目对您的研究有帮助，请引用我们的文章：

```bibtex
@misc{feng2025vidarembodiedvideodiffusion,
      title={Vidar: Embodied Video Diffusion Model for Generalist Manipulation}, 
      author={Yao Feng and Hengkai Tan and Xinyi Mao and Chendong Xiang and Guodong Liu and Shuhe Huang and Hang Su and Jun Zhu},
      year={2025},
      eprint={2507.12898},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.12898}, 
}

```

## 🙏 致谢 (Acknowledgements)
本项目参考了以下开源项目，特此感谢：
- [Wan2.2](https://github.com/Wan-Video/Wan2.2/)


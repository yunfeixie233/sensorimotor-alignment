# Guide for Evaluating PickPlaceAgent on the RoboTwin 2.0 Platform (Example: `place_empty_cup` Task)

This guide explains how to evaluate **PickPlaceAgent** on the **RoboTwin 2.0** platform, using the `place_empty_cup` task as an example.

---
## Step 1: Prepare environment 
Install environment local refer to [INSTALL.md](./docs/INSTALL.md)


## Step 2: Prepare the Model Checkpoint and fix checkpoints

- Download Finegrasp weight in [Finegrasp](https://huggingface.co/HorizonRobotics/Finegrasp) repo and save it to `ckpts` folder
- Download Sa2VA-4B weight in [Sa2VA-4B](https://huggingface.co/ByteDance/Sa2VA-4B) repo and and save it to `thirdparty` folder
- Download GenpPose weight in [GenPose](https://github.com/Omni6DPose/GenPose2?tab=readme-ov-file#%EF%B8%8F-download-dataset-and-models) repo and save it to `thirdparty/GenPose2/results/ckpts` folder

Replace `ckpt_root` in the deploy_policy.py with the checkpoints path

> Notice: the `ckpt_root` is default set to `/3rdparty/robo_orchard_lab/projects/pick_place_agent`. All checkpoints are ready at this path on the evaluation server for the [Challenge Cup](https://developer.d-robotics.cc/tiaozhanbei-2025) RoboTwin simulation benchmark.



```
.
├── ckpts
│   └── finegrasp_sim.pth
└── thirdparty
    ├── GenPose2
    │   ├── results
    │   │   └── ckpts
    │   │       ├── EnergyNet
    │   │       │   └── energynet.pth
    │   │       ├── ScaleNet
    │   │       │   └── scalenet.pth
    │   │       └── ScoreNet
    │   │           └── scorenet.pth
    └── Sa2VA-4B
```



## Step 3: Copy the Policy Folder to the RoboTwin 2.0 Project Directory
```bash
cd ..
cp -r pick_place_agent {ROBOTWIN2_PATH}/policy/
```
Replace `{ROBOTWIN2_PATH}` with the path to your local RoboTwin 2.0 project.

## Step 4: Run the Model Evaluation Script
```bash
# set depth & pointcloud to true
cd {ROBOTWIN2_PATH}/policy/pick_place_agent
sh eval.sh
```
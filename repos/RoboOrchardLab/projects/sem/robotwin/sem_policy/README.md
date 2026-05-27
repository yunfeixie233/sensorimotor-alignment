
# Guide for Evaluating SEM Models on the RoboTwin 2.0 Platform (Example: `place_empty_cup` Task)

This guide explains how to evaluate **SEM models** on the **RoboTwin 2.0** platform, using the `place_empty_cup` task as an example.

---

## Step 1: Copy the Config File to the Policy Folder

```bash
cd projects/sem/robotwin/sem_policy
cp ../config_sem_robotwin.py ./
```

## Step 2: Modify the Config Settings

For Local evaluation, open `config_sem_robotwin.py` and update the URDF path to match the RoboTwin 2.0 asset path.

For server evaluation, copy the urdf file to submission folder.

```bash
cp ${ROBOTWIN_DIR}/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf ./
```
Then open `config_sem_robotwin.py` and update the URDF path as:
```python
urdf="/workspace/policy/custom_policy/arx5_description_isaac.urdf",
```

## Step 3: Prepare the Model Checkpoint

```bash
CHECKPOINT_DIR=checkpoints/place_empty_cup
mkdir -p ${CHECKPOINT_DIR}
cp {PATH_TO_YOUR_SAFETENSOR_CKPT} ${CHECKPOINT_DIR}
```

Replace `{PATH_TO_YOUR_SAFETENSOR_CKPT}` with the path to your `.safetensors` model file.

We also provide a trained checkpoint for `place_empty_cup` task in  https://huggingface.co/HorizonRobotics/SEM-RoboTwin-Tiny.

## Step 4: Local Evaluation - Copy the Policy Folder to the RoboTwin 2.0 Project Directory

```bash
cd ..
cp -r sem_policy {ROBOTWIN2_PATH}/policy/
```

Replace `{ROBOTWIN2_PATH}` with the path to your local RoboTwin 2.0 project.

## Step 5: Local Evaluation - Run the Model Evaluation Script

```bash
cp {ROBOTWIN2_PATH}/task_config/_embodiment_config.yml {ROBOTWIN2_PATH}/task_config/agent_config.yml 
```
change `depth` in the evaluation `${task_config}`(e.g. `task_config/demo_clean.yml`) to true

```
cd {ROBOTWIN2_PATH}/policy/sem_policy
sh eval.sh
```

## Step 6: Server Evaluation - Submit Codes and Checkpoint

```bash
./submit upload \
    --api-key {YOUR_API_KEY} \
    --submission-id {SUBMISSION_ID} \
    --dir ./sem_policy \
    --checkpoint-dir ./sem_policy/checkpoints/place_empty_cup
```

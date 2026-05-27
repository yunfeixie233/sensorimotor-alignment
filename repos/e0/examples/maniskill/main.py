import collections
import dataclasses
import logging
import os
import pathlib
import random
import shutil
from collections import deque
from pathlib import Path
from typing import Callable, List, Type

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import tqdm
import tyro
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image

env_seed = 0
random.seed(env_seed)
os.environ['PYTHONHASHSEED'] = str(env_seed)
np.random.seed(env_seed)
torch.manual_seed(env_seed)
torch.cuda.manual_seed(env_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 9023
    replan_steps: int = 10

    # #################################################################################################################
    # # Maniskill environment-specific parameters
    # #################################################################################################################

    env_id : str = "PickCube-v1"  # all, PushCube-v1, PickCube-v1, StackCube-v1, PegInsertionSide-v1, PlugCharger-v1
    obs_mode : str = "rgb"
    render_mode: str = "rgb_array"
    reward_mode: str = "dense"
    shader : str = "default"
    # sim_backend :str = ""

    max_episode_steps : int = 400

    # #################################################################################################################
    # # Utils
    # #################################################################################################################
    model_name: str = "your_model_name"  # Name for save
    video_out_path: str = "./sim_output/maniskill"  # Path to save videos

    random_exp_n : int = 1
    total_episodes : int = 25

    seed: int = 7


def save_image(image, save_path):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in [1, 3]:  
        image = np.transpose(image, (1, 2, 0))  
    elif image.ndim == 4:
        image = np.transpose(image[0], (1, 2, 0))

    if image.ndim == 3 and image.shape[2] == 1:  
        image = image.squeeze(-1)


    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(image).save(save_path)


def eval_maniskill(args:Args):

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = _get_maniskill_env(args)

    video_dir = Path(args.video_out_path) / args.model_name / args.env_id
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)


    for exp_n in range(args.random_exp_n):
        success_count = 0
        for episode in tqdm.trange(args.total_episodes):

            obs, _ = env.reset(seed = episode + args.seed)
            img = env.render().squeeze(0).detach().cpu().numpy()
            proprio = obs['agent']['qpos'][:, :-1]
            proprio = proprio.detach().cpu().numpy().squeeze(0)

            action_plan = collections.deque()
            global_steps = 0
            video_frames = []
            done = False
            while global_steps < args.max_episode_steps and not done:

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/state": proprio,
                        "prompt" : task2lang[args.env_id],
                    }

                    action_chunk = client.infer(element)["actions"]
                    assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                # print(f"Step {global_steps}, action: {action}")
                obs, reward, terminated, truncated, info = env.step(action)

                # obs
                img = env.render().squeeze(0).detach().cpu().numpy()
                proprio = obs['agent']['qpos'][:, :-1]
                proprio = proprio.detach().cpu().numpy().squeeze(0)
                video_frames.append(img)
                global_steps += 1

                if terminated or truncated:
                    assert "success" in info, sorted(info.keys())
                    if info['success']:
                        success_count += 1
                        done = True
                        break
            
            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            imageio.mimwrite(
                video_dir / f"exp_{exp_n}_episode_{episode}_{suffix}.mp4",
                [np.asarray(x) for x in video_frames],
                fps=20,
            )
            print(f"Trial {episode + 1} finished, success: {info['success']}, steps: {global_steps}")


    success_rate = success_count / (args.total_episodes * args.random_exp_n) * 100
    print(f"Success rate: {success_rate}%")
    log_filename = f"success_rate_{success_rate}-episodes_num_{args.total_episodes}.txt"
    log_filepath = video_dir / log_filename
    with Path.open(log_filepath, "w", encoding="utf-8") as f:
        f.write(f"Total success rate: {success_rate}\n")
        f.write(f"Total episodes: {args.total_episodes}\n")




def _get_maniskill_env(args):
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="dense" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        )
    
    return env

def eval_maniskill_all(args: Args):
    if args.env_id =="all":
        task_list = ["PushCube-v1", "PickCube-v1", "StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1"]

    print(f"task list : {task_list}")
    for task in task_list:
        args.env_id = task
        eval_maniskill(args)



def main(args:Args):
    if "all" in args.env_id:
        eval_maniskill_all(args)     
    else:
        eval_maniskill(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
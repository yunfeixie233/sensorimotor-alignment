import collections
import dataclasses
import json
import logging
import math
import os
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from VLABench.envs import load_env
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.base import Policy
from VLABench.robots import *
from VLABench.tasks import *
# install vlabench in venv first
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

VLABENCH_DUMMY_ACTION = [0.0] * 6 + [0.04, 0.04]
VLABENCH_ENV_RESOLUTION = 480  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 480
    replan_steps: int = 20

    #################################################################################################################
    # VLABench environment-specific parameters
    #################################################################################################################

    
    eval_track: str = None
    track: str = None  # track_1, track_2, track_3, track_4, track_6
    num_steps_wait: int = 15  # Number of steps to wait for objects to stabilize i n sim
    n_episode: int = 50  # Number of rollouts per task

    intention_score_threshold: float = 0.2

    #################################################################################################################
    # Utils
    #################################################################################################################

    metrics: str = "success_rate intention_score progress_score"
    model_name: str = "your_model_name"
    save_dir: str = "./sim_output/vlabench"
    visulization: bool = True

    seed: int = 7  # Random Seed (for reproducibility)

    def __post_init__(self):
        base_path = "./third_party/VLABench/VLABench/configs/evaluation/tracks"
        track_map = {
            "track_1": "track_1_in_distribution.json",
            "track_2": "track_2_cross_category.json",
            "track_3": "track_3_common_sense.json",
            "track_4": "track_4_semantic_instruction.json",
            "track_6": "track_6_unseen_texture.json",
        }
        if self.track:
            if self.track not in track_map:
                raise ValueError(f"Unsupported eval_track: {self.eval_track}")
            self.episode_config_path = f"{base_path}/{track_map[self.track]}"
            print(f"Using episode config path: {self.episode_config_path}")
            if self.track == "track_3":
                self.tasks = "select_fruit select_toy select_painting"
            else:
                self.tasks = "select_fruit select_toy select_painting select_poker select_mahjong"




class EvalPolicy(Policy):
    def __init__(self, client, replan_steps=5):
        self.model = client
        self.replan_steps=replan_steps
        self.action_plan = collections.deque(maxlen=replan_steps)
    
    def reset(self):
        self.action_plan.clear()
    
    def predict(self, obs, **kwargs):
        if len(self.action_plan) == 0:
            second_image, _, image, image_wrist = obs["rgb"]
            state = obs["ee_state"]
            last_action = obs["last_action"].copy()
            pos, quat, gripper_state = state[:3], state[3:7], state[-1]
            ee_euler = quaternion_to_euler(quat)
            pos -= np.array([0, -0.4, 0.78])
            state = np.concatenate([pos, ee_euler, np.array(gripper_state).reshape(-1)])
            # last_action = np.concatenate([last_action, np.array(gripper_state).reshape(-1)])
            policy_input = {
                "observation/image": image,
                "observation/second_image": second_image,
                "observation/wrist_image": image_wrist,
                "observation/state": state,
                "prompt": obs["instruction"]
            }
            action_chunk = self.model.infer(policy_input)["actions"]
            assert (
                len(action_chunk) >= self.replan_steps
            ), f"We want to replan every {self.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
            self.action_plan.extend(action_chunk[: self.replan_steps])
        action = self.action_plan.popleft()
        target_pos, target_euler, gripper = action[:3], action[3:6], action[-1]
        if gripper >= 0.1:
            gripper_state = np.ones(2)*0.04
        else:
            gripper_state = np.zeros(2)
        target_pos = target_pos.copy()
        target_pos += np.array([0, -0.4, 0.78])
        return target_pos, target_euler, gripper_state
    
    @property
    def name(self):
        return "policy"



def main(args:Args) -> None:
    if args.eval_track is not None:  
        if not args.eval_track.endswith(".json"):
            args.eval_track = args.eval_track + ".json"
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", args.eval_track), "r") as f:
            tasks = json.load(f)
    else:
        tasks = args.tasks.split(" ")

    metrics = args.metrics.split(" ")
    assert isinstance(tasks, list)

    with open(args.episode_config_path, "r")  as f:
        episode_configs=json.load(f)
    
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    policy = EvalPolicy(
        client=client,
        replan_steps=args.replan_steps
    )
    save_dir = f"{args.save_dir}/{args.model_name}"

    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_configs,
        max_substeps=1,
        save_dir=save_dir,
        visulization=args.visulization,
        metrics=metrics,
        #intention_score_threshold=args.intention_score_threshold
    )
    
    result = evaluator.evaluate(policy)
    print(result)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
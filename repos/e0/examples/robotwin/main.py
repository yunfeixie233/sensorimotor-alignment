import sys
import os
import subprocess
import dataclasses
import logging

sys.path.append("/home/jusheng/RoboTwin")
sys.path.append("/home/jusheng/RoboTwin/description/utils")
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import *

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import shutil
import yaml
from datetime import datetime
import importlib
import argparse
import pdb
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 20

    #################################################################################################################
    # RoboTwin environment-specific parameters
    #################################################################################################################
    
    test_num: int = 100
    task_name: str = "beat_block_hammer"
    task_config : str = "demo_clean"
    instruction_type : str = "unseen"
    embodiment : str = "aloha-agilex"

    ckpt_setting : int = 30000
    #################################################################################################################
    # Utils
    #################################################################################################################
    model_name: str = "your_model_name"
    video_out_path: str = "./sim_output/robotwin"
    eval_video_log: bool = True

    seed: int = 7  # Random Seed (for reproducibility)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def get_camera_config(camera_type):
    camera_config_path = "/home/jusheng/RoboTwin/task_config/_camera_config.yml"
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def eval_robotwin(usr_input_args: Args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_name = usr_input_args.task_name
    task_config = usr_input_args.task_config # demo_clean / demo_randomized
    policy_name = usr_input_args.model_name
    instruction_type = usr_input_args.instruction_type
    ckpt_setting = usr_input_args.ckpt_setting
    embodiment_type = usr_input_args.embodiment # "aloha-agilex"
    replan_steps = usr_input_args.replan_steps

    embodiment_type = [embodiment_type]
    
    #### demo_clean / demo_randomized
    with open(f"/home/jusheng/RoboTwin/task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["policy_name"] = policy_name
    args["replan_steps"] = replan_steps

    def get_embodiment_file(embodiment_type):
        embodiment_asset_dir = "/home/jusheng/RoboTwin/assets/embodiments"
        robot_file_dict = {
            "aloha-agilex": os.path.join(embodiment_asset_dir, "aloha-agilex/"),
            "piper": os.path.join(embodiment_asset_dir, "piper"),
            "franka-panda": os.path.join(embodiment_asset_dir, "franka-panda/"),
            "ARX-X5": os.path.join(embodiment_asset_dir, "ARX-X5"),
            "ur5-wsg": os.path.join(embodiment_asset_dir, "ur5-wsg"),
        }
        robot_file = robot_file_dict[embodiment_type]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open("/home/jusheng/RoboTwin/task_config/_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # print(len(embodiment_type))
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"{usr_input_args.video_out_path}/{usr_input_args.model_name}/{task_name}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    if usr_input_args.eval_video_log:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")


    TASK_ENV = class_decorator(args["task_name"])
    #usr_input_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    #usr_input_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_input_args.seed
    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = usr_input_args.test_num
    topk = 1

    policy = _websocket_client_policy.WebsocketClientPolicy(usr_input_args.host, usr_input_args.port)

    st_seed, suc_num = eval_policy(task_name, TASK_ENV, args, policy, st_seed,
                                   test_num=test_num, video_size=video_size, instruction_type=instruction_type,)
    suc_nums.append(suc_num)

    # topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    
    log_filename = f"success_rate_{float(np.array(suc_nums) / float(test_num))}-episodes_num_{test_num}.txt"
    
    # file_path = os.path.join(save_dir, f"_result.txt")
    file_path = os.path.join(save_dir, log_filename)
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name, TASK_ENV, args, policy, st_seed, test_num=100, video_size=None, instruction_type=None,):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        policy.reset()

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()

            input_state = observation["joint_action"]["vector"]
            instruction = TASK_ENV.get_instruction()
            img_front = np.transpose(observation["observation"]["head_camera"]["rgb"], (2, 0, 1))
            img_right = np.transpose(observation["observation"]["right_camera"]["rgb"], (2, 0, 1))
            img_left = np.transpose( observation["observation"]["left_camera"]["rgb"], (2, 0, 1))
            element = { 
                        "images": {
                            "cam_high": img_front,
                            "cam_left_wrist": img_left,
                            "cam_right_wrist": img_right,
                        },
                        "state":input_state,
                        "prompt": instruction,
                        }

            action_chunk = policy.infer(element)["actions"]

            for action in action_chunk[:args["replan_steps"]]:
                TASK_ENV.take_action(action) 
                observation = TASK_ENV.get_obs()

            if TASK_ENV.eval_success:
                succ = True
                break
        
        # task_total_reward += TASK_ENV.episode_score

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()


        if TASK_ENV.eval_video_path is not None:
            old_video = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"
            result_tag = "success" if succ else "fail"
            new_video = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_{result_tag}.mp4"


            if os.path.exists(old_video):
                shutil.move(old_video, new_video)
                print(f"[Video saved as] {new_video}")
            else:
                print(f"[Warning] video file not found: {old_video}")


        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc

def list_py_dirs(root_dir: str) -> list[str]:
    result = []
    for name in os.listdir(root_dir):
        if name.endswith(".py") and not name.startswith("_"):
            result.append(name[:-3])
    return result

def main(args: Args):
    if args.task_name == "all":
        task_name_list = list_py_dirs("RoboTwin/envs")
        print(f"total {len(task_name_list)} tasks")
        for task in task_name_list:
            args.task_name = task
            eval_robotwin(args)
    else:
        eval_robotwin(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
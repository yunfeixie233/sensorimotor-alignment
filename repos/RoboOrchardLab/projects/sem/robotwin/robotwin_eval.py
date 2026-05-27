# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import json
import logging
import os
import sys

import yaml
from utils import load_checkpoint, load_config  # type: ignore

from robo_orchard_lab.dataset.robotwin.close_loop_eval import (
    class_decorator,
    evaluation,
    update_task_config,
)
from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)
logging.disable(logging.CRITICAL)


def main(args):
    config = load_config(args.config)
    build_model = config.build_model
    build_dataset = config.build_dataset
    config = config.config

    print("\n" + json.dumps(config, indent=4))

    model = build_model(config)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = config.get("checkpoint")
    load_checkpoint(model, checkpoint)
    model.eval()
    model.requires_grad_()
    model.cuda()
    val_dataset = build_dataset(config, lazy_init=True)[1]
    if args.task_names == "all":
        task_names = os.listdir(os.path.join(args.robotwin_dir, "task_config"))
        task_names = [
            x[:-4]
            for x in task_names
            if x.endswith(".yml") and (not x.startswith("_"))
        ]
    else:
        task_names = args.task_names.split(",")

    print(f"eval tasks: {task_names}")
    for task_name in task_names:
        workspace = os.path.abspath(os.path.join(args.workspace, task_name))
        if not os.path.exists(workspace):
            os.makedirs(workspace, exist_ok=True)

        sys.path.insert(0, os.path.abspath(args.robotwin_dir))
        os.chdir(args.robotwin_dir)
        with open(
            f"./task_config/{task_name}.yml", "r", encoding="utf-8"
        ) as f:
            task_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        from envs import CONFIGS_PATH

        task_config = update_task_config(task_config, CONFIGS_PATH)

        print(f"task_name: {task_name}")
        print(f"num_test: {args.num_test}")

        task_env = class_decorator(task_name)
        task_env.test_num = 0

        seed = args.start_seed
        render_freq = task_config["render_freq"]

        results = dict()
        num_success = 0
        while len(results) < args.num_test:
            if args.expert_check:
                try:
                    task_config["render_freq"] = 0
                    task_env.setup_demo(
                        now_ep_num=len(results),
                        seed=seed,
                        is_test=True,
                        **task_config,
                    )
                    task_env.play_once()
                    task_env.close()
                    if not (
                        task_env.plan_success and task_env.check_success()
                    ):
                        seed += 1
                        continue
                except Exception as e:
                    print(f"expert failed at seed[{seed}]: {e}")
                    task_env.close()
                    seed += 1
                    continue

            task_config["render_freq"] = render_freq
            task_config["eval_mode"] = True
            task_env.setup_demo(
                now_ep_num=len(results), seed=seed, is_test=True, **task_config
            )

            eval_result = evaluation(task_env, model, val_dataset, workspace)
            task_env.close()
            results[seed] = eval_result
            num_success += eval_result

            if task_env.render_freq:
                task_env.viewer.close()
            print(
                f"seed[{seed}]: {'success' if eval_result else 'fail'}, "
                f"success rate: {num_success}/{len(results)}"
            )
            task_env.test_num += 1
            seed += 1

        results["success_rate"] = num_success / len(results)
        with open(os.path.join(workspace, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--task_names", type=str, default="all")
    parser.add_argument("--camera_type", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--start_seed", type=int, default=100000)
    parser.add_argument("--expert_check", action="store_false")
    parser.add_argument("--robotwin_dir", type=str, default="./robotwin")
    parser.add_argument("--workspace", type=str, default="./workspace")
    args = parser.parse_args()
    workspace_root = args.workspace
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    main(args)

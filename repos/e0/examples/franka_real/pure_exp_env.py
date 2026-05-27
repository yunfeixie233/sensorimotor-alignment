import collections
import logging
import time
from typing import List, Optional
import sys

import cv2
import dm_env
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import *
from base_env import BaseEnv

class PureExpEnv(BaseEnv):
    def __init__(self,
                robot_ip: str = "your robot ip",
                task_name: str = "test",
                action_space: str = "joint",
                camera_num: int = 2,
                motion_mode: str  = "abs",
                setup_robots: bool = True,
                smooth_actions: bool = False,
                **kwargs):
        super().__init__(robot_ip=robot_ip, setup_robots=setup_robots, camera_num=camera_num, action_space=action_space, 
                motion_mode = motion_mode, is_franky=True, is_ada_multicamera=False, **kwargs)

        self.smooth_actions = smooth_actions
        self.dt = 1
        self.max_episode_steps = kwargs.get("max_episode_steps")

        self.init_task_settings(task_name) 
        self._reset_task() 



    def init_task_settings(self, task_name):
        self.task_name = task_name
        self.task_list = list(TASK_DICT.keys())
        assert self.task_name in self.task_list, "Please check your task_name !"
        
        self.task_prompt = TASK_DICT[self.task_name]
        self.reset_position_by_task = True 


    ############################### reset ###############################
    def _reset_task(self,):
        self.step_num = 0
        self.help_num = 0
        
        self.grasp_num = 0
        self.put_num = 0
        self.last_grasp_time = 0
        self.last_put_time = 0

        self.wait_for_grasp = True
        self.wait_for_put = False
        self.task_success = False

        self.final_retry_times = 0

    def _reset_joints(self):
        if self.reset_position_by_task:
            start_joints = START_DICT.get(self.task_name, None)
            if start_joints is not None:
                self.robot.set_joints(start_joints)
            else:
                self.robot.home_joints()
        else:
            self.robot.home_joints() if self.action_space == "joint" else self.robot.home_pose()


    def reset(self, *, fake=False):
        self._reset_task()
        if not fake:
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation())


    ###############################  get ###############################

    def get_task_name(self,):
        return self.task_name

    def get_task_prompt(self,):
        return self.task_prompt

    def get_observation(self):
        obs = collections.OrderedDict()
        obs["gripper_width"] = self.get_gripper_width()
        obs["ee_pose"] = self.get_ee_pose()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["images"] = self.get_images()
        obs["prompt"] = self.get_task_prompt()
        return obs

    def _step_example(self, action):
        joints_action = action[:7]
        gripper_action = action[-1]
        self.move_joints_abs(joints_action, smooth=self.smooth_actions)

        print(gripper_action)
        if gripper_action < 0.1 and (not self.wait_for_grasp):
            time.sleep(1)
            self.robot.open()
            self.wait_for_grasp = True
        elif gripper_action > 0.75 and self.wait_for_grasp:
            time.sleep(1)
            success = self.robot.grasp(width=0.05, epsilon_outer=0.1)
            time.sleep(1)
            if success:
                self.wait_for_grasp = False 
                self.task_success = True

        if self.task_success:
            return dm_env.TimeStep(step_type=dm_env.StepType.LAST, reward=self.get_reward(), discount=0, observation=self.get_observation())
        else:
             return dm_env.TimeStep(step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation())



    def step(self, action):

        self.step_num += 1
        print(f"step_num : {self.step_num}")
        if self.task_name == "pick_block":
            output = self._step_example(action)

        return output


if __name__=="__main__":
    camera_num = 3
    camera_num = 2 
    setup_robots = False
    exp_env = PureExpEnv(
        setup_robots = setup_robots,
        camera_num = camera_num,
    )
    exp_env.show_images()

import collections
import logging
import time
from typing import List, Optional
import sys

import cv2
import dm_env
import numpy as np
from scipy.spatial.transform import Rotation as R

from camera import Camera, MultiRealSenseCapture
from franky_robot import FrankaFrankyRobot
from utils import *


class BaseEnv:
    def __init__(self, 
                robot_ip: str = "your robot ip",
                setup_robots: bool = True,
                camera_num: int = 2,
                action_space: str = "joint",
                motion_mode:str = "abs",
                is_franky: bool = True,
                is_ada_multicamera: bool = False,
                **kwargs):

        self.robot = None
        self.robot_ip = robot_ip
        self.action_space = action_space
        self.motion_mode = motion_mode
        self.camera_num = camera_num
        self.is_franky = is_franky
        self.is_ada_multicamera = is_ada_multicamera

        if setup_robots:
            self.setup_robots()
        self.setup_camera()


    def setup_robots(self,):
        if self.is_franky:
            self.robot = FrankaFrankyRobot(self.robot_ip)
            print("(franky) Robot is ready.\n")
        else:
            raise NotImplementedError("Not support now.")

    def setup_camera(self,):
        if self.is_ada_multicamera:
            raise NotImplementedError("Not support now.")
        else:
            print("Use old camera setting.\n")
            self.camera_list = []
            for i in range(self.camera_num):
                camera = Camera(WIDTH, HEIGHT, FPS, device_index = i)
                self.camera_list.append(camera)
            print(f"{self.camera_num} camera(s) is ready")

    ############################### reset ###############################

    def _reset_joints(self):
        self.robot.home_joints() if self.action_space == "joint" else self.robot.home_pose()


    def _reset_gripper(self):
        self.robot.set_gripper(width=0.08)
    

    def reset(self, *, fake=False):
        raise NotImplementedError("You need to rewrite this function for each env.")

    ###############################  get ###############################.

    def get_reward(self,):
        return 0
    
    def get_gripper_width(self,):
        return self.robot.gripper_width

    def get_ee_pose(self,):
        return self.robot.ee_pose

    def get_qpos(self):
        return self.robot.qpos

    def get_qvel(self):
        return self.robot.qvel

    def get_images(self):
        obs_dict = {}
        if self.is_ada_multicamera:
            raise NotImplementedError("Not support now.")
        else:
            for i, camera in enumerate(self.camera_list):
                camera.align_frames()
                obs_dict[f"camera_{i}"] = camera.get_rgb_frame()
        return obs_dict
    
    def get_observation(self):
        raise NotImplementedError("You need to rewrite this function for each env.")

    ############################### set ###############################

    def move_joints_abs(self, qpos, smooth=False, smooth_steps=10,):
        if smooth:
            self.robot.set_joints_smooth(target_joints=qpos, steps=smooth_steps, async_mode=True)
        else:
            self.robot.set_joints(qpos)
    
    def move_ee_abs(self, ee_t = None, ee_q = None):
        ee_t = self.robot.ee_t if ee_t is None else np.array(ee_t)
        ee_q = self.robot.ee_q if ee_q is None else np.array(ee_q)
        ee_pose = np.append(ee_t, ee_q)
        self.robot.set_ee_pose(ee_pose)

    def move_joints_rel(self, delta_qpos):
        raise NotImplementedError()
        now_qpos = self.robot.qpos
        update_qpos = now_qpos + delta_qpos
        self.robot.set_joints(update_qpos)

    def move_ee_rel(self, delta_ee_t, delta_ee_q=None):
        raise NotImplementedError()
        self.robot.set_ee_pose_relative(delta_ee_t)



    def step(self, action):
        raise NotImplementedError("You need to rewrite this function for each env.")


    def show_images(self):
        if self.is_ada_multicamera:
            raise NotImplementedError("Not support now.")
        else:
            while True:
                images = self.get_images()
                for cam_name, img in images.items():
                    cv2.imshow(cam_name, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
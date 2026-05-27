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

# ruff: noqa: E501, N803
import copy
import logging
import os
import sys

import cv2
import numpy as np
from envs.utils import transforms
from envs.utils.action import ArmTag

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from agent_policy import PickPlaceAgent  # noqa: E402
from agent_utils.pose_post_process_utils import (  # noqa: E402
    PosePostProcess,
)
from agent_utils.viz_utils import viz_pose  # noqa: E402
from config.agent_config import AgentConfig  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _pose_to_matrix(pose):
    """Convert a pose in the format [x, y, z, qw, qx, qy, qz to a 4x4 transformation matrix."""
    x, y, z, qw, qx, qy, qz = pose
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rotation
    pose_mat[:3, 3] = [x, y, z]
    return pose_mat


def _matrix_to_pose(pose_mat):
    """Convert a 4x4 transformation matrix to a pose in the format [x, y, z, qx, qy, qz, qw]."""
    pose = np.zeros(7)
    translation = pose_mat[:3, 3]
    rotation = pose_mat[:3, :3]
    quaternion = Rotation.from_matrix(rotation).as_quat()
    pose[:3] = translation
    pose[3:] = quaternion
    return pose


class AgentPolicy:  # noqa: N801
    def __init__(self, ckpt_root, enable_viz=False):
        self.enable_viz = enable_viz
        self.ckpt_root = ckpt_root
        self.agent = self._build_model()

    def _build_model(
        self,
    ):
        config = AgentConfig(
            grasp_model_config=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "finegrasp/config.py",
            ),
            grasp_model_ckpt=os.path.join(
                self.ckpt_root,
                "ckpts/finegrasp_sim.pth",
            ),
            grounding_model_path=os.path.join(
                self.ckpt_root,
                "thirdparty/Sa2VA-4B",
            ),
            pose_model_path=os.path.join(
                self.ckpt_root,
                "thirdparty/GenPose2/results/ckpts/",
            ),
            grounding_gpu_id=0,
            visualize_pose_after_workspace_filter=False,
            visualize_pose_after_grounding_filter=False,
            visualize_pose_after_topdown_filter=False,
        )
        agent = PickPlaceAgent(config)
        return agent

    def run_active_vision(self, obs, pick_object, TASK_ENV):
        mask, grounding_rgb, grounding_depth = self.agent.run_grounding(
            rgb=obs["rgb_head_camera"],
            depth=obs["depth_head_camera"],
            object_prompt=pick_object,
        )
        grounding_depth[mask == 0] = 0.0
        camera_info = self.agent.get_camera_info(obs, "head_camera")
        pose_post_process = PosePostProcess(camera_info)

        # find max z-axis point
        grounding_points_world = pose_post_process.get_grounding_point(
            grounding_depth
        )
        max_z_index = np.argmax(grounding_points_world[:, 2])
        max_z_point = grounding_points_world[max_z_index]

        max_z_pose = max_z_point.tolist() + [0.5, -0.5, 0.5, 0.5]
        max_z_pose[2] += 0.2

        left_pose = copy.deepcopy(max_z_pose)
        right_pose = copy.deepcopy(max_z_pose)

        if self.enable_viz:
            concatenated_rgb = np.concatenate(
                [
                    obs["rgb_head_camera"],
                    obs["rgb_left_camera"],
                    obs["rgb_right_camera"],
                ],
                axis=1,
            )
            cv2.imwrite(
                "before_active_rgb.png",
                cv2.cvtColor(concatenated_rgb, cv2.COLOR_RGB2BGR),
            )

        try:
            # Move the left arm to the active pose
            arm_tag = ArmTag("left")
            left_actions = self.plan_path(
                arm_tag, TASK_ENV, pose=left_pose, gripper=0.8, sample_num=50
            )
            for action in left_actions:
                observation = TASK_ENV.get_obs()
                TASK_ENV.take_action(action)
        except Exception:
            # Move the right arm to the active pose
            arm_tag = ArmTag("right")
            try:
                right_actions = self.plan_path(
                    arm_tag,
                    TASK_ENV,
                    pose=right_pose,
                    gripper=0.8,
                    sample_num=50,
                )
                for action in right_actions:
                    observation = TASK_ENV.get_obs()
                    TASK_ENV.take_action(action)
            except Exception:
                logger.error("No valid active vision, skip this step.")

        if self.enable_viz:
            observation = TASK_ENV.get_obs()
            after_obs = encode_obs(observation, "left_camera")
            after_obs.update(encode_obs(observation, "head_camera"))
            after_obs.update(encode_obs(observation, "right_camera"))
            concatenated_rgb = np.concatenate(
                [
                    after_obs["rgb_head_camera"],
                    after_obs["rgb_left_camera"],
                    after_obs["rgb_right_camera"],
                ],
                axis=1,
            )
            cv2.imwrite(
                "after_activte_rgb.png",
                cv2.cvtColor(concatenated_rgb, cv2.COLOR_RGB2BGR),
            )

        # update observation, only update left and right to avoid obstacles
        observation = TASK_ENV.get_obs()
        obs_left_camera = encode_obs(observation, "left_camera")
        obs_right_camera = encode_obs(observation, "right_camera")
        obs.update(obs_left_camera)
        obs.update(obs_right_camera)
        return obs

    def get_pick_pose(
        self,
        obs,
        TASK_ENV,  # noqa: N803
        pick_object,
        place_object,
    ):
        """Return pick pose in world coord."""
        pick_pose_tcp_mat, place_pose_init_mat, pick_obj_6dpose = (
            self.agent.run_pick(
                obs=obs,
                pick_object=pick_object,
                place_object=place_object,
                vis=self.enable_viz,
            )
        )
        pick_pose_tcp = _matrix_to_pose(pick_pose_tcp_mat)
        pick_pose_tcp = np.concatenate(
            (pick_pose_tcp[:3], pick_pose_tcp[3:][[3, 0, 1, 2]])
        )

        pick_pose_ee = self._move_along_arm(pick_pose_tcp, 0.12)
        print("Pick Pose:", pick_pose_ee)

        if self.enable_viz:
            hist_robot_state = np.concatenate(
                [
                    np.array(pick_pose_ee).reshape(1, -1),
                    np.array(pick_pose_tcp).reshape(1, -1),
                ],
            )
            viz_pose(
                obs=obs,
                camera_name="head_camera",
                hist_robot_state=hist_robot_state,
                save_path="pick_pose.png",
            )

        return (
            pick_pose_ee,
            place_pose_init_mat,
            pick_obj_6dpose,
        )  # noqa

    def get_place_pose(
        self,
        obs,
        place_pose_init_mat,
        arm_tag,
        TASK_ENV,
        pick_object,
        pick_pose,
        prev_6dpose,
        tracking=True,
    ):
        """Return place pose in world coord."""
        current_arm_pose = TASK_ENV.get_arm_pose(arm_tag=arm_tag)

        prev_6dpose = prev_6dpose[None, None, :]
        ee_trans = self.agent.run_place(
            obs=obs,
            pick_object=pick_object,
            place_pose_world=place_pose_init_mat,
            tracking=tracking,
            prev_pose=prev_6dpose,
            vis=self.enable_viz,
        )
        arm_pose_mat = _pose_to_matrix(current_arm_pose)
        place_pose_mat = ee_trans @ arm_pose_mat

        y_axis = place_pose_mat[:3, :3][:, 1]
        x_dot = np.dot(y_axis, np.array([1, 0, 0]))
        if x_dot > np.cos(np.deg2rad(60)):
            logger.info("Reverse Pose.")
            r_x_180 = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            new_rotation_matrix = place_pose_mat[:3, :3] @ r_x_180
            new_orientation_list = Rotation.from_matrix(
                new_rotation_matrix
            ).as_quat(scalar_first=True)
            place_pose = np.array(
                [
                    place_pose_mat[0, 3],
                    place_pose_mat[1, 3],
                    place_pose_mat[2, 3],
                    new_orientation_list[0],
                    new_orientation_list[1],
                    new_orientation_list[2],
                    new_orientation_list[3],
                ]
            )
        else:
            place_pose = _matrix_to_pose(place_pose_mat)
            place_pose = np.concatenate(
                (place_pose[:3], place_pose[3:][[3, 0, 1, 2]])
            )
        # use pick z-axis to set place z-axis
        place_pose[2] = pick_pose[2] + 0.05
        print("Place Pose:", place_pose)
        if self.enable_viz:
            current_arm_pose = TASK_ENV.get_arm_pose(arm_tag=arm_tag)
            hist_robot_state = np.concatenate(
                [
                    np.array(current_arm_pose).reshape(1, -1),
                    np.array(place_pose).reshape(1, -1),
                ],
            )
            viz_pose(obs, "head_camera", hist_robot_state, "place_pose.png")

        return place_pose

    def _move_along_arm(self, origin_pose, value):
        displacement = np.zeros(7, dtype=np.float64)
        dir_vec = transforms._toPose(origin_pose).to_transformation_matrix()[
            :3, 0
        ]
        dir_vec /= np.linalg.norm(dir_vec)
        displacement[:3] = -value * dir_vec
        move_pose = origin_pose + displacement
        return move_pose

    def plan_path(
        self,
        arm_tag,
        TASK_ENV,
        gripper,
        pose=None,
        last_action=None,
        sample_num=100,
    ):
        if arm_tag == "left":
            if pose is not None:
                left_pose = TASK_ENV.robot.left_plan_path(pose)["position"]
                left_gripper = np.tile(gripper, (left_pose.shape[0], 1))
                left_action = np.column_stack([left_pose, left_gripper])
                right_action = np.tile(
                    TASK_ENV.robot.get_right_arm_jointState(),
                    (len(left_action), 1),
                )
            else:
                left_gripper = TASK_ENV.robot.left_plan_grippers(
                    TASK_ENV.robot.get_left_gripper_val(), gripper
                )["result"].reshape(-1, 1)
                left_pose = np.tile(last_action[:6], (len(left_gripper), 1))
                left_action = np.concatenate([left_pose, left_gripper], axis=1)
                right_action = np.tile(last_action[7:], (len(left_action), 1))
        elif arm_tag == "right":
            if pose is not None:
                right_pose = TASK_ENV.robot.right_plan_path(pose)["position"]
                right_gripper = np.tile(gripper, (right_pose.shape[0], 1))
                right_action = np.column_stack([right_pose, right_gripper])
                left_action = np.tile(
                    TASK_ENV.robot.get_left_arm_jointState(),
                    (len(right_action), 1),
                )
            else:
                right_gripper = TASK_ENV.robot.right_plan_grippers(
                    TASK_ENV.robot.get_right_gripper_val(), gripper
                )["result"].reshape(-1, 1)
                right_pose = np.tile(
                    last_action[7:13], (len(right_gripper), 1)
                )
                right_action = np.concatenate(
                    [right_pose, right_gripper], axis=1
                )
                left_action = np.tile(last_action[:7], (len(right_action), 1))
        else:
            raise ValueError(
                f"Invalid arm_tag: {arm_tag}. Must be 'left' or 'right'."
            )

        actions = np.concatenate([left_action, right_action], axis=1)

        if len(actions) > sample_num:
            actions = actions[:: len(actions) // sample_num]
        return actions

    def get_pick_action(self, pick_pose, arm_tag, TASK_ENV):
        # pick the object
        action1 = self.plan_path(
            arm_tag, TASK_ENV, pose=pick_pose, gripper=0.8, sample_num=50
        )
        # close gripper
        action2 = self.plan_path(
            arm_tag,
            TASK_ENV,
            gripper=0.0,
            last_action=action1[-1],
            sample_num=50,
        )
        actions = np.concatenate([action1, action2], axis=0)
        return actions

    def get_place_action(self, place_pose, arm_tag, TASK_ENV):
        # place the object
        action1 = self.plan_path(
            arm_tag, TASK_ENV, pose=place_pose, gripper=0.0, sample_num=50
        )
        # open gripper
        action2 = self.plan_path(
            arm_tag,
            TASK_ENV,
            gripper=1.0,
            last_action=action1[-1],
            sample_num=50,
        )
        actions = np.concatenate([action1, action2], axis=0)
        return actions

    def get_lift_action(
        self, pose, arm_tag, TASK_ENV, gripper, lift_height=0.02
    ):
        # lift the object by lift_height along z-axis
        lift_pase = pose.copy()
        lift_pase[2] += lift_height
        action = self.plan_path(
            arm_tag, TASK_ENV, pose=lift_pase, gripper=gripper, sample_num=50
        )
        return action


def encode_obs(observation, camera="head_camera"):
    obs = {
        # (240, 320, 3), uint8, 0~255
        f"rgb_{camera}": observation["observation"][camera]["rgb"],
        # (240, 320), float64, mm
        f"depth_{camera}": observation["observation"][camera]["depth"],
        # (3, 3), float32
        f"intrinsic_cv_{camera}": observation["observation"][camera][
            "intrinsic_cv"
        ],
        # (3, 4), float32
        f"extrinsic_cv_{camera}": observation["observation"][camera][
            "extrinsic_cv"
        ],
    }
    return obs


def get_model(usr_args):
    # ckpt_root need to change to checkpoint path
    model = AgentPolicy(
        ckpt_root="/3rdparty/robo_orchard_lab/projects/pick_place_agent",
        enable_viz=False,
    )
    return model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation, "head_camera")
    obs.update(encode_obs(observation, "left_camera"))
    obs.update(encode_obs(observation, "right_camera"))

    instruction = TASK_ENV.get_instruction()  # noqa: F841
    pick_object = "colored plastic cup"
    place_object = "gray coster"

    # run active vision to get grounding result
    try:
        logger.info("Running active vision ...")
        obs = model.run_active_vision(obs, pick_object, TASK_ENV)
    except Exception:
        logger.error("No valid active vision, skip this step.")

    try:
        (
            pick_pose,
            place_pose_init_mat,
            pick_obj_6dpose,
        ) = model.get_pick_pose(
            obs,
            TASK_ENV,
            pick_object=pick_object,
            place_object=place_object,
        )
    except Exception:
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        logger.error("No valid pick pose, skip this step.")
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    arm_tag = ArmTag("right" if pick_pose[0] > 0 else "left")
    try:
        actions = model.get_pick_action(pick_pose, arm_tag, TASK_ENV)
    except Exception:
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        logger.error("No valid pick action, skip this step.")
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    for action in actions:
        observation = TASK_ENV.get_obs()
        TASK_ENV.take_action(action)

    try:
        actions = model.get_lift_action(
            pick_pose, arm_tag, TASK_ENV, gripper=0.0, lift_height=0.03
        )
    except Exception:
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        logger.error("No valid pick lift action, skip this step.")
        return

    for action in actions:
        observation = TASK_ENV.get_obs()
        TASK_ENV.take_action(action)

    # above is place part, note it
    obs = encode_obs(observation, "head_camera")
    obs.update(encode_obs(observation, "left_camera"))
    obs.update(encode_obs(observation, "right_camera"))
    try:
        place_pose = model.get_place_pose(
            obs=obs,
            place_pose_init_mat=place_pose_init_mat,
            arm_tag=arm_tag,
            TASK_ENV=TASK_ENV,
            pick_object=pick_object,
            pick_pose=pick_pose,
            prev_6dpose=pick_obj_6dpose,
        )
    except Exception:
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        logger.error("No valid place pose, skip this step.")
        return

    try:
        actions = model.get_place_action(place_pose, arm_tag, TASK_ENV)
    except Exception:
        logger.error("No valid place action, skip this step.")
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    for action in actions:
        observation = TASK_ENV.get_obs()
        TASK_ENV.take_action(action)
    try:
        actions = model.get_lift_action(
            place_pose, arm_tag, TASK_ENV, gripper=1.0, lift_height=0.08
        )
    except Exception:
        logger.error("No valid place lift action, skip this step.")
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return
    for action in actions:
        observation = TASK_ENV.get_obs()
        TASK_ENV.take_action(action)
    TASK_ENV.take_action_cnt = TASK_ENV.step_lim


def reset_model(model):
    pass

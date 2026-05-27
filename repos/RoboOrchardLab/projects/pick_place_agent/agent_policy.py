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

import logging

import cv2
import numpy as np
import open3d as o3d
from agent_utils.grounding_utils import Grounding
from agent_utils.pose_estimation_utils import (
    PoseEstimation,
)
from agent_utils.pose_post_process_utils import (
    PosePostProcess,
)
from config.agent_config import AgentConfig
from finegrasp.infer import Inferencer as FineGraspInfer
from PIL import Image

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def visualize_depth(
    depth,
    min_valid_depth=0.1,
    max_valid_depth=2.0,
    colormap=cv2.COLORMAP_JET,
    invalid_color=(0, 0, 0),
):
    assert min_valid_depth is not None or max_valid_depth is not None

    # Convert uint16 format to meters as float
    if depth.dtype == np.uint16:
        depth = depth.astype(float) / 1000.0  # Convert to meters

    # Handle invalid values (NaN or non-positive values)
    mask_valid = depth > 0
    # Normalize to [0, 1] range
    depth_scaled = np.where(
        mask_valid,
        (depth - min_valid_depth) / (max_valid_depth - min_valid_depth),
        0,
    )

    # Clip and scale to [0, 255]
    depth_scaled = np.clip(depth_scaled * 255, 0, 255).astype(np.uint8)

    # Apply color mapping
    colored = cv2.applyColorMap(depth_scaled, colormap)

    # Mark invalid regions
    colored[~mask_valid] = invalid_color

    return colored


class PickPlaceAgent:
    def __init__(self, config: AgentConfig):
        self._initialize(config)

    def _initialize(self, config: AgentConfig):
        self.config = config

        self.grasp_infer = FineGraspInfer(
            self.config.grasp_model_config, self.config.grasp_model_ckpt
        )

        self.grounder = Grounding(
            self.config.grounding_model_path, self.config.grounding_gpu_id
        )

        self.pose_estimator = PoseEstimation(self.config.pose_model_path)

    def _depth_to_pointcloud(self, depth_image, camera_info):
        xmap, ymap = (
            np.arange(depth_image.shape[1]),
            np.arange(depth_image.shape[0]),
        )
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_image / camera_info["intrinsic"]["scale"]
        points_x = (
            (xmap - camera_info["intrinsic"]["cx"])
            / camera_info["intrinsic"]["fx"]
            * points_z
        )
        points_y = (
            (ymap - camera_info["intrinsic"]["cy"])
            / camera_info["intrinsic"]["fy"]
            * points_z
        )
        points = np.stack([points_x, points_y, points_z], axis=-1).astype(
            np.float32
        )
        return points

    def grasp_pose_viz(self, gg, rgb_image, depth_image, camera_info):
        points = self._depth_to_pointcloud(depth_image, camera_info)
        points_x = points[:, :, 0]
        points_y = points[:, :, 1]
        points_z = points[:, :, 2]
        mask = (points_z > 0) & (points_z < 2)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = rgb_image[mask].astype(np.float32) / 255.0
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([*grippers, cloud])

    def _gen_place_point(self, grounding_rgb, grounding_depth, camera_info):
        def _project_point_to_3d(point_2d, depth_image):
            u, v = point_2d
            x = (
                (u - camera_info["intrinsic"]["cx"])
                * depth_image
                / camera_info["intrinsic"]["fx"]
            )
            y = (
                (v - camera_info["intrinsic"]["cy"])
                * depth_image
                / camera_info["intrinsic"]["fy"]
            )
            z = depth_image
            return np.column_stack((x, y, z))

        mask = np.any(grounding_rgb != [0, 0, 0], axis=-1)
        y_coords, x_coords = np.where(mask)
        if x_coords.size == 0 or y_coords.size == 0:
            logger.error("No valid place point found in grounding.")
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        place_x = (x_min + x_max) // 2
        place_y = (y_min + y_max) // 2
        place_point_2d = np.array([place_x, place_y])
        place_point_depth = grounding_depth[place_y, place_x] / 1000.0
        place_point_3d = _project_point_to_3d(
            place_point_2d, place_point_depth
        )
        return place_point_3d.squeeze()

    def run_grasp_inference(
        self,
        rgb,
        depth,
        camera_info,
        max_gripper_width,
        voxel_size_cd=0.01,
        collision_thresh=-1,
        topk=50,
    ):
        points = self._depth_to_pointcloud(depth, camera_info)
        zmin, zmax = 0.0, 2.0
        mask = (points[..., 2] > zmin) & (points[..., 2] < zmax)
        points = points[mask].astype(np.float32)
        rgb = rgb[mask].astype(np.float32)
        gg_topN, _, _ = self.grasp_infer.infer(  # noqa: N806
            points,
            rgb,
            topk=topk,
            max_gripper_width=max_gripper_width,
            voxel_size_cd=voxel_size_cd,
            collision_thresh=collision_thresh,
        )
        return gg_topN

    def run_grounding(self, rgb, depth, object_prompt):
        object_rgb, object_depth, object_mask = self.grounder.get_object(
            rgb_img=rgb, depth_img=depth, object_prompt=object_prompt
        )
        return object_mask * 255, object_rgb, object_depth

    def run_pose_estimation(
        self,
        rgb,
        depth,
        object_mask,
        vis=False,
        note=None,
        tracking=False,
        prev_pose=None,
    ):
        pose = self.pose_estimator.get_object_pose(
            rgb_img=rgb,
            depth_img=depth,
            mask_img=object_mask,
            vis=vis,
            note=note,
            tracking=tracking,
            prev_pose=prev_pose,
        )
        return pose

    def get_camera_info(self, obs, camera):
        extrinsic_4x4 = np.vstack(
            [obs[f"extrinsic_cv_{camera}"], [0, 0, 0, 1]]
        )
        camera_info = {
            "intrinsic": {
                "fx": obs[f"intrinsic_cv_{camera}"][0, 0],
                "fy": obs[f"intrinsic_cv_{camera}"][1, 1],
                "cx": obs[f"intrinsic_cv_{camera}"][0, 2],
                "cy": obs[f"intrinsic_cv_{camera}"][1, 2],
                "scale": 1000,
            },
            "extrinsic": extrinsic_4x4,
            "workspace_limits": {
                "x_min": -0.5,
                "x_max": 0.5,
                "y_min": -0.5,
                "y_max": 0.5,
                "z_min": 0.0,
                "z_max": 1.0,
            },
        }

        return camera_info

    def get_world_grasp_pose(
        self, obs, camera, pick_object, enable_viz_open3d=False
    ):
        rgb = obs[f"rgb_{camera}"]
        depth = obs[f"depth_{camera}"]
        camera_info = self.get_camera_info(obs, camera)
        pose_post_process = PosePostProcess(camera_info)

        if rgb is None or depth is None:
            logger.error("RGB or depth image is None.")
            raise ValueError("RGB or depth image is None.")

        pick_grounding_mask, pick_grounding_rgb, pick_grounding_depth = (
            self.run_grounding(rgb=rgb, depth=depth, object_prompt=pick_object)
        )
        pick_grounding_rgb[pick_grounding_mask == 0] = 0.0
        pick_grounding_depth[pick_grounding_mask == 0] = 0.0

        try:
            gg_topN_croped = self.run_grasp_inference(  # noqa: N806
                pick_grounding_rgb,
                pick_grounding_depth,
                camera_info,
                voxel_size_cd=0.01,
                collision_thresh=0.01,
                max_gripper_width=0.10,
            )  # noqa: N806
        except Exception:
            logger.info(f"Camera {camera} failed to run grasp inference.")

        grasp_poses_croped_img = pose_post_process.get_grasp_poses_img(
            gg_topN_croped
        )
        grasp_poses_croped_world = pose_post_process.grasp_poses_to_world(
            grasp_poses_croped_img
        )

        if enable_viz_open3d is True:
            self.grasp_pose_viz(gg_topN_croped, rgb, depth, camera_info)

        grasp_poses_after_workspace_filter, gg_picks_after_workspace_filter = (
            pose_post_process.workspace_filter(
                grasp_poses_croped_world, gg_topN_croped
            )
        )
        if enable_viz_open3d is True:
            self.grasp_pose_viz(
                gg_picks_after_workspace_filter, rgb, depth, camera_info
            )

        grasp_poses_after_grounding_filter, gg_picks_after_grounding_filter = (
            pose_post_process.grounding_filter(
                grasp_poses_after_workspace_filter,
                gg_picks_after_workspace_filter,
                pick_grounding_depth,
            )
        )
        if len(grasp_poses_after_grounding_filter) == 0:
            logger.info(
                f"Camera {camera} no grasp poses after grounding filter."
            )

        if self.config.visualize_pose_after_grounding_filter:
            self.grasp_pose_viz(
                gg_picks_after_grounding_filter, rgb, depth, camera_info
            )

        # top-down filter
        grasp_poses_after_topdown_filter, gg_picks_after_topdown_filter = (
            pose_post_process.top_down_filter(
                grasp_poses_after_grounding_filter,
                gg_picks_after_grounding_filter,
                max_angle_deg=60,
            )
        )
        if len(grasp_poses_after_topdown_filter) == 0:
            logger.info(
                f"Camera {camera} No grasp poses after top-down filter."
            )

        if self.config.visualize_pose_after_topdown_filter:
            self.grasp_pose_viz(
                gg_picks_after_topdown_filter, rgb, depth, camera_info
            )

        return grasp_poses_after_topdown_filter, gg_picks_after_topdown_filter

    def run_pick(self, obs, pick_object, place_object, vis=False):
        # open3d_viz need GUI
        enable_viz_open3d = False

        grasp_poses_world_head_camera, gg_picks_world_head_camera = (
            self.get_world_grasp_pose(
                obs=obs,
                camera="head_camera",
                pick_object=pick_object,
                enable_viz_open3d=enable_viz_open3d,
            )
        )
        grasp_poses_world_left_camera, gg_picks_world_left_camera = (
            self.get_world_grasp_pose(
                obs=obs,
                camera="left_camera",
                pick_object=pick_object,
                enable_viz_open3d=enable_viz_open3d,
            )
        )
        grasp_poses_world_right_camera, gg_picks_world_right_camera = (
            self.get_world_grasp_pose(
                obs=obs,
                camera="right_camera",
                pick_object=pick_object,
                enable_viz_open3d=enable_viz_open3d,
            )
        )

        # merge the grasp poses from different cameras
        grasp_poses_world = []
        for poses in [
            grasp_poses_world_head_camera,
            grasp_poses_world_left_camera,
            grasp_poses_world_right_camera,
        ]:
            if poses is not None:
                grasp_poses_world.extend(poses)
        logger.info(f"len(grasp_poses_world) is {len(grasp_poses_world)}")

        camera_info_head_camera = self.get_camera_info(obs, "head_camera")
        camera_info_left_camera = self.get_camera_info(obs, "left_camera")
        camera_info_right_camera = self.get_camera_info(obs, "right_camera")

        if enable_viz_open3d:
            self.grasp_pose_viz(
                gg_picks_world_head_camera,
                obs["rgb_head_camera"],
                obs["depth_head_camera"],
                camera_info_head_camera,
            )
            self.grasp_pose_viz(
                gg_picks_world_left_camera,
                obs["rgb_left_camera"],
                obs["depth_left_camera"],
                camera_info_left_camera,
            )
            self.grasp_pose_viz(
                gg_picks_world_right_camera,
                obs["rgb_right_camera"],
                obs["depth_right_camera"],
                camera_info_right_camera,
            )

        # pick pose
        pick_ee_pose = grasp_poses_world[0]

        # generate place point
        pick_mask_img, _, _ = self.run_grounding(
            rgb=obs["rgb_head_camera"],
            depth=obs["depth_head_camera"],
            object_prompt=pick_object,
        )
        _, place_grounding_rgb, place_grounding_depth = self.run_grounding(
            rgb=obs["rgb_head_camera"],
            depth=obs["depth_head_camera"],
            object_prompt=place_object,
        )
        place_point_img = self._gen_place_point(
            place_grounding_rgb, place_grounding_depth, camera_info_head_camera
        )
        if place_point_img is None:
            logger.error("No place point found in grounding.")
        place_point_img_homo = np.append(place_point_img, 1.0)
        place_point_world = (
            np.linalg.inv(np.array(camera_info_head_camera["extrinsic"]))
            @ place_point_img_homo
        )[:3]

        # get 6d pose of the pick object
        pick_object_pose_img = (
            self.run_pose_estimation(
                obs["rgb_head_camera"],
                obs["depth_head_camera"],
                pick_mask_img,
                vis=vis,
                note="pick",
            )[0]
            .cpu()
            .numpy()
            .squeeze()
        )
        object_pose_world = (
            np.linalg.inv(np.array(camera_info_head_camera["extrinsic"]))
            @ pick_object_pose_img
        )
        place_pose_world = np.eye(4)
        place_pose_world[:3, :3] = object_pose_world[:3, :3]
        place_pose_world[:3, 3] = place_point_world

        return pick_ee_pose, place_pose_world, pick_object_pose_img

    def run_place(
        self,
        obs,
        pick_object,
        place_pose_world,
        tracking=False,
        prev_pose=None,
        vis=False,
    ):
        """Generate place end-effector pose after picking the object."""
        rgb, depth = (
            obs["rgb_head_camera"],
            obs["depth_head_camera"].astype(np.uint16),
        )
        camera_info = self.get_camera_info(obs, "head_camera")
        pick_mask_img, pick_grounding_rgb, _ = self.run_grounding(
            rgb=rgb, depth=depth, object_prompt=pick_object
        )
        if np.all(pick_mask_img == 0):
            logger.error("No pick object found in grounding.")
        if vis:
            cv2.imwrite(
                "grounding.png",
                cv2.cvtColor(pick_grounding_rgb, cv2.COLOR_RGB2BGR),
            )

        # get 6d pose of the pick object, now it is grasped by the robot
        pick_object_pose_img = (
            self.run_pose_estimation(
                rgb,
                depth,
                pick_mask_img,
                vis=vis,
                note="place",
                tracking=tracking,
                prev_pose=prev_pose,
            )[0]
            .cpu()
            .numpy()
            .squeeze()
        )

        pick_object_pose_world = (
            np.linalg.inv(np.array(camera_info["extrinsic"]))
            @ pick_object_pose_img
        )
        place_trans = place_pose_world @ np.linalg.inv(pick_object_pose_world)
        return place_trans

    def visualize_world_pose(self, obs, pose_world, length=0.05):
        """Visualize a world pose on the RGB image.

        Args:
            rgb_image: Input RGB image (H, W, 3)
            pose_world: 4x4 transformation matrix in world coordinates
            length: Length of the coordinate axes in meters

        Returns:
            Image with coordinate axes drawn at the pose position
        """
        # Convert world pose to camera coordinates
        camera_info = self.get_camera_info(obs, "head_camera")
        rgb_image = obs["rgb_head_camera"]
        pose_cam = camera_info["extrinsic"] @ pose_world

        # Create coordinate axes in 3D (origin and endpoints of X,Y,Z axes)
        points_3d = np.array(
            [
                [0, 0, 0, 1],  # Origin
                [length, 0, 0, 1],  # X axis
                [0, length, 0, 1],  # Y axis
                [0, 0, length, 1],  # Z axis
            ]
        ).T

        # Transform points to camera frame
        points_cam = pose_cam @ points_3d

        # Project 3D points to 2D image coordinates
        fx = camera_info["intrinsic"]["fx"]
        fy = camera_info["intrinsic"]["fy"]
        cx = camera_info["intrinsic"]["cx"]
        cy = camera_info["intrinsic"]["cy"]

        points_2d = []
        for i in range(4):
            x = points_cam[0, i] / points_cam[2, i] * fx + cx
            y = points_cam[1, i] / points_cam[2, i] * fy + cy
            points_2d.append((x, y))

        # Draw axes on the image
        img = rgb_image.copy()
        origin = tuple(np.round(points_2d[0]).astype(int))

        # X axis (red)
        x_end = tuple(np.round(points_2d[1]).astype(int))
        cv2.line(img, origin, x_end, (255, 0, 0), 2)

        # Y axis (green)
        y_end = tuple(np.round(points_2d[2]).astype(int))
        cv2.line(img, origin, y_end, (0, 255, 0), 2)

        # Z axis (blue)
        z_end = tuple(np.round(points_2d[3]).astype(int))
        cv2.line(img, origin, z_end, (0, 0, 255), 2)

        return img


if __name__ == "__main__":
    config = AgentConfig(
        grasp_model_config="finegrasp/config.py",
        grasp_model_ckpt="ckpts/finegrasp_sim.pth",
        grounding_model_path="thirdparty/Sa2VA-4B",
        pose_model_path="thirdparty/GenPose2/results/ckpts/",
        grounding_gpu_id=0,
        visualize_pose_after_workspace_filter=False,
        visualize_pose_after_grounding_filter=False,
        visualize_pose_after_topdown_filter=False,
    )
    pick_place_agent = PickPlaceAgent(config)
    example_rgb = np.array(Image.open("data/example_color.png").convert("RGB"))
    example_depth = np.array(Image.open("data/example_depth.png"))
    obs = dict()
    for camera in ["head_camera", "left_camera", "right_camera"]:
        obs[f"rgb_{camera}"] = example_rgb
        obs[f"depth_{camera}"] = example_depth
        obs[f"intrinsic_cv_{camera}"] = np.array(
            [[358.64218, 0.0, 160.0], [0.0, 358.64218, 120.0], [0.0, 0.0, 1.0]]
        )
        obs[f"extrinsic_cv_{camera}"] = np.array(
            [
                [1.0000000e00, 1.1920929e-07, -5.9604645e-08, 3.2000184e-02],
                [5.9604645e-08, -8.0000007e-01, -5.9999990e-01, 4.5000005e-01],
                [-1.1920929e-07, 5.9999996e-01, -8.0000007e-01, 1.3500001e00],
            ]
        )
    camera_info = pick_place_agent.get_camera_info(obs, "head_camera")
    example_pick_object = "colored plastic cup"
    example_place_object = "gray coster"

    # unit tests
    gg_picks = pick_place_agent.run_grasp_inference(
        example_rgb, example_depth, camera_info, max_gripper_width=0.10
    )
    mask, grounding_rgb, grounding_depth = pick_place_agent.run_grounding(
        example_rgb, example_depth, example_pick_object
    )
    estimated_pose = pick_place_agent.run_pose_estimation(
        grounding_rgb, grounding_depth, mask
    )

    # run the agent
    # get obs from initial set
    example_pick_ee_pose, example_place_pose, _ = pick_place_agent.run_pick(
        obs=obs,
        pick_object=example_pick_object,
        place_object=example_place_object,
    )

    # get obs after picking
    # you would get a transform from current ee pose to place ee pose
    ee_trans = pick_place_agent.run_place(
        obs=obs,
        pick_object=example_pick_object,
        place_pose_world=example_place_pose,
    )

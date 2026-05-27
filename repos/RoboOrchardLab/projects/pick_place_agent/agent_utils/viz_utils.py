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

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def draw_robot_state(
    imgs,
    projection_mat,
    robot_state,
    ee_indices=(6, 13),
    channel_conversion=True,
):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    if isinstance(projection_mat, torch.Tensor):
        projection_mat = projection_mat.cpu().numpy()
    if isinstance(robot_state, torch.Tensor):
        robot_state = robot_state.cpu().numpy()

    vis_imgs = []
    for img_index in range(imgs.shape[0]):
        img = imgs[img_index]
        img = np.ascontiguousarray(img)
        for joint_index in range(robot_state.shape[0]):
            rot = Rotation.from_quat(
                robot_state[joint_index, 3:], scalar_first=True
            ).as_matrix()
            trans = robot_state[joint_index, :3]

            if joint_index in ee_indices:
                axis_length = 0.1
            else:
                axis_length = 0.03

            points = np.float32(
                [
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length],
                    [0, 0, 0],
                ]
            )
            points = points @ rot.T + trans

            pts_2d = points @ projection_mat[img_index, :3, :3].T
            pts_2d = pts_2d + projection_mat[img_index, :3, 3]  # (4,3)
            depth = pts_2d[:, 2]
            pts_2d = pts_2d[:, :2] / depth[:, None]  # (4,2)

            if depth[3] < 0.02:
                continue

            pts_2d = pts_2d.astype(np.int32)
            for i in range(3):
                if depth[i] < 0.02:
                    continue
                cv2.circle(
                    img, (pts_2d[i, 0], pts_2d[i, 1]), 6, (0, 0, 255), -1
                )
                if i == 3:
                    continue
                color = [0, 0, 0]
                color[i] = 255
                cv2.line(
                    img,
                    (pts_2d[i, 0], pts_2d[i, 1]),
                    (pts_2d[3, 0], pts_2d[3, 1]),
                    tuple(color),
                    3,
                )
        vis_imgs.append(img)
        if len(vis_imgs) == 4:
            vis_imgs = np.concatenate(
                [
                    np.concatenate(vis_imgs[:2], axis=1),
                    np.concatenate(vis_imgs[2:], axis=1),
                ],
                axis=0,
            )
        else:
            vis_imgs = np.concatenate(vis_imgs, axis=1)
        vis_imgs = np.uint8(vis_imgs)
        if channel_conversion:
            vis_imgs = vis_imgs[..., ::-1]
        return vis_imgs


def viz_pose(obs, camera_name, hist_robot_state, save_path):
    imgs = obs[f"rgb_{camera_name}"][None, :]
    projection_mat = (
        obs[f"intrinsic_cv_{camera_name}"] @ obs[f"extrinsic_cv_{camera_name}"]
    )
    projection_mat = projection_mat[None, :]
    vis_imgs = draw_robot_state(imgs, projection_mat, hist_robot_state)
    vis_imgs = np.ascontiguousarray(vis_imgs)
    cv2.imwrite(save_path, vis_imgs)


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

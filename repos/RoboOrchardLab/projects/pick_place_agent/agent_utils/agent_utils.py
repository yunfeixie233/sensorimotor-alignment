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

import base64

import cv2
import numpy as np
import open3d as o3d


def parse_rgb_image(base64_str: str) -> np.ndarray:
    """Transform base64 string to image.

    Args:
        base64_str (str): Base64 encoded string of the RGB image.

    Returns:
        np.ndarray: Decoded RGB image as a NumPy array, normalized to [0, 1].
    """
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) / 255.0
    return img


def parse_depth_image(base64_str: str) -> np.ndarray:
    """Transform base64 string to image.

    Args:
        base64_str (str): Base64 encoded string of the depth image.

    Returns:
        np.ndarray: Decoded depth image as a NumPy array.
    """
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img


def encode_rgb_image(image):
    """Transform image to base64 string.

    Args:
        image (np.ndarray): Input RGB image as a NumPy array.

    Returns:
        str: Base64 encoded string of the image.
    """
    if image.dtype != np.uint8 and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    _, img_encoded = cv2.imencode(".jpg", image)
    return base64.b64encode(img_encoded).decode("utf-8")


def encode_depth_image(image):
    """Transform depth image to base64 string.

    Args:
        image (np.ndarray): Input depth image as a NumPy array.

    Returns:
        str: Base64 encoded string of the depth image.
    """
    _, img_encoded = cv2.imencode(".png", image)
    return base64.b64encode(img_encoded).decode("utf-8")


def grasp_pose_viz(config, gg, rgb_image, depth_image):
    """Visualize grasp poses and point cloud.

    Args:
        config (dict): Configuration dictionary containing camera intrinsics.
        gg (list): List of grasp geometries to visualize.
        rgb_image (np.ndarray): RGB image as a NumPy array.
        depth_image (np.ndarray): Depth image as a NumPy array.
    """
    camera_intrinsics = config["camera_info"]["camera_intrinsics"]
    # get point cloud
    fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
    cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]
    scale = camera_intrinsics["scale"]

    xmap, ymap = (
        np.arange(depth_image.shape[1]),
        np.arange(depth_image.shape[0]),
    )
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth_image / scale
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy

    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = rgb_image[mask].astype(np.float32)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([*grippers, cloud])

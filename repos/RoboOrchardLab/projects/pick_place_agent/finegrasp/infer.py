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
import importlib
import json
import logging
import os

import numpy as np
import open3d
import requests
import torch
from graspnetAPI.graspnet_eval import GraspGroup
from PIL import Image
from safetensors.torch import load_model

from robo_orchard_lab.models.finegrasp.utils import (
    ModelFreeCollisionDetector,
    pred_decode,
)
from robo_orchard_lab.utils import depth_to_range_image, seed_everything

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_finegrasp_minkunet.py",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="finegrasp.safetensors",
    )
    parser.add_argument("--viz", type=bool, default=False)
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()
    return args


class GetFile:
    def __init__(self, url):
        self.url = url
        self.need_remove = False

    def __enter__(self):
        if not self.url.startswith("http"):
            return self.url
        file_name = "_" + self.url.split("/")[-1]
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        self.url = file_name
        self.need_remove = True
        return file_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.need_remove:
            os.system(f"rm {self.url}")


class Inferencer:
    def __init__(self, config, checkpoint_path, device="cuda"):
        config = load_config(config)
        build_model = config.build_model
        model_cfgs, model = build_model()
        self.model_cfgs = model_cfgs
        self.model = model
        self.device = device

        # Load checkpoint
        self.load_checkpoint(self.model, checkpoint_path)
        self.model.eval()
        self.model.to(device)

    def load_checkpoint(self, model, checkpoint=None, **kwargs):
        if checkpoint is None:
            return

        logger.info(f"load checkpoint: {checkpoint}")
        with GetFile(checkpoint) as checkpoint:
            if checkpoint.endswith(".safetensors"):
                load_model(model, checkpoint, **kwargs)
            else:
                state_dict = torch.load(checkpoint)
                state_dict = state_dict["model_state"]
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict, strict=False, **kwargs
                )
                logger.info(
                    f"num of missing_keys: {len(missing_keys)},"
                    f"num of unexpected_keys: {len(unexpected_keys)}"
                )

    def get_normal(self, point_cloud):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(point_cloud)

        # compute normals
        pcd.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        return np.asarray(pcd.normals)

    def infer(
        self,
        points,
        colors,
        topk,
        max_gripper_width=None,
        voxel_size_cd=0.01,
        collision_thresh=0.01,
    ):
        # sample points random
        num_sample_points = 20000
        if len(points) >= num_sample_points:
            idxs = np.random.choice(
                len(points), num_sample_points, replace=False
            )
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(
                len(points),
                num_sample_points - len(points),
                replace=True,
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        points = points[idxs].astype(np.float32)
        colors = colors[idxs].astype(np.float32)
        coordinates_for_voxel = (
            points.astype(np.float32) / self.model_cfgs.voxel_size
        )
        cloud_normal = self.get_normal(points) / 255

        point_clouds = torch.from_numpy(points).unsqueeze(0).to(self.device)
        cloud_colors = torch.from_numpy(colors).unsqueeze(0).to(self.device)
        cloud_normal = (
            torch.from_numpy(cloud_normal).unsqueeze(0).float().to(self.device)
        )
        coordinates_for_voxel = (
            torch.from_numpy(coordinates_for_voxel)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        data_dict = {
            "point_clouds": point_clouds,
            "cloud_colors": cloud_colors,
            "cloud_normal": cloud_normal,
            "coordinates_for_voxel": coordinates_for_voxel,
        }

        with torch.no_grad():
            end_points = self.model(data_dict)
            grasp_preds = pred_decode(
                end_points,
                self.model_cfgs.grasp_max_width,
                self.model_cfgs.num_seed_points,
            )

        preds = grasp_preds[0].detach().cpu().numpy()

        # filter the predictions which width is larger than max_gripper_width
        if max_gripper_width is not None:
            width_mask = preds[:, 1] < max_gripper_width
            preds = preds[width_mask]

        gg = GraspGroup(preds)

        # collision detection
        if collision_thresh > 0:
            cloud = data_dict["point_clouds"]
            mfcdetector = ModelFreeCollisionDetector(
                cloud[0].cpu().numpy(), voxel_size=voxel_size_cd
            )
            collision_mask = mfcdetector.detect(
                gg,
                approach_dist=0.05,
                collision_thresh=collision_thresh,
            )
            gg = gg[~collision_mask]

        return_dict = {}
        return_dict["cloud_points"] = np.asarray(
            data_dict["point_clouds"].cpu()
        )
        return_dict["cloud_colors"] = np.asarray(
            data_dict["cloud_colors"].cpu()
        )

        if len(gg) == 0:
            print("No Grasp detected after collision detection!")

        gg = gg.nms().sort_by_score()
        gg_pick_topk = gg[0:topk]
        print(gg_pick_topk.scores)
        print("grasp score:", gg_pick_topk[0].score)
        return_dict["gg_pick_topk"] = gg_pick_topk
        return (
            gg_pick_topk,
            return_dict["cloud_points"],
            return_dict["cloud_colors"],
        )


if __name__ == "__main__":
    args = parse_config()
    seed_everything(2025)
    inferencer = Inferencer(
        config=args.config,
        checkpoint_path=args.checkpoint_path,
    )

    intrinsic_file = "data/000000_meta.json"
    rgb_image_path = "data/example_color.png"
    depth_image_path = "data/example_depth.png"

    meta = json.load(open(intrinsic_file, "r"))
    intrinsics = meta["camera"]["intrinsics"]
    cx, cy, fx, fy = (
        intrinsics["cx"],
        intrinsics["cy"],
        intrinsics["fx"],
        intrinsics["fy"],
    )
    intrinsic_matrix = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
    )

    depth = np.array(Image.open(depth_image_path), dtype=np.float32)
    colors = np.array(Image.open(rgb_image_path), dtype=np.float32) / 255.0

    points = depth_to_range_image(
        depth, camera_intrinsic=intrinsic_matrix, depth_scale=1000.0
    )

    # set workspace to filter output grasps
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    zmin, zmax = 0.0, 2.0

    x_mask = (points[..., 0] > xmin) & (points[..., 0] < xmax)
    y_mask = (points[..., 1] > ymin) & (points[..., 1] < ymax)
    z_mask = (points[..., 2] > zmin) & (points[..., 2] < zmax)
    mask = x_mask & y_mask & z_mask
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    # without collision detection
    inferencer.infer(
        points, colors, topk=50, max_gripper_width=0.10, collision_thresh=-1
    )

    # with collision detection
    inferencer.infer(
        points, colors, topk=50, voxel_size_cd=0.01, collision_thresh=0.01
    )

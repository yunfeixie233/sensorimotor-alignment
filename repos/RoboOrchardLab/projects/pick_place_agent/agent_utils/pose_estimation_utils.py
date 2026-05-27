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

import warnings

import cv2
import numpy as np
import torch
from cutoop.eval_utils import DetectMatch
from PIL import Image
from thirdparty.GenPose2.datasets.datasets_infer import (  # noqa: E501
    InferDataset,
)
from thirdparty.GenPose2.runners.infer import (  # noqa: E501
    GenPose2,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class PoseEstimation:
    def __init__(
        self,
        ckpt_path="thirdparty/GenPose2/results/ckpts/",
    ):
        self.pose_estimator = GenPose2(
            ckpt_path + "ScoreNet/scorenet.pth",
            ckpt_path + "EnergyNet/energynet.pth",
            ckpt_path + "ScaleNet/scalenet.pth",
        )

    def visualize_pose(
        self, data: InferDataset, all_final_pose, all_final_length
    ):
        color_img = cv2.cvtColor(data.color, cv2.COLOR_RGB2BGR)
        all_final_pose = all_final_pose[0].cpu().numpy()
        all_final_length = all_final_length[0].cpu().numpy()
        for _index, (obj_pose, obj_length) in enumerate(
            zip(all_final_pose, all_final_length, strict=False)
        ):
            color_img = DetectMatch._draw_image(
                vis_img=color_img,
                pred_affine=obj_pose,
                pred_size=obj_length,
                gt_affine=None,
                gt_size=None,
                gt_sym_label=None,
                camera_intrinsics=data.cam_intrinsics,
                draw_pred=True,
                draw_gt=False,
                draw_label=False,
                draw_pred_axes_length=0.1,
                draw_gt_axes_length=None,
                thickness=True,
            )
        return color_img

    def get_object_pose(
        self,
        rgb_img,
        depth_img,
        mask_img,
        vis=False,
        note=None,
        tracking=False,
        prev_pose=None,
    ):
        data = InferDataset.alternetive_init_custom(
            depth_img=depth_img / 1000.0,
            color_img=rgb_img,
            mask_img=mask_img,
            img_size=self.pose_estimator.cfg.img_size,
            device=self.pose_estimator.cfg.device,
            n_pts=self.pose_estimator.cfg.num_points,
        )
        if isinstance(prev_pose, np.ndarray):
            prev_pose = torch.from_numpy(prev_pose).to(
                self.pose_estimator.cfg.device
            )
        pose, length = self.pose_estimator.inference(
            data, prev_pose=prev_pose, tracking=tracking, tracking_T0=0.15
        )
        if vis:
            color_image_w_pose = self.visualize_pose(data, pose, length)
            cv2.imwrite(
                f"{note}_6dpose.png",
                color_image_w_pose,
            )
        return pose


if __name__ == "__main__":
    pose_estimator = PoseEstimation()
    rgb_image = np.array(Image.open("data/example_color.png")).astype(np.uint8)
    depth_image = np.array(Image.open("data/example_depth.png")).astype(
        np.float32
    )
    mask_image = np.array(Image.open("data/grounding_mask.png")).astype(
        np.uint8
    )
    object_pose = pose_estimator.get_object_pose(
        rgb_image, depth_image, mask_image
    )

    print("Estimated Object Pose:", object_pose)

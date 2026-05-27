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

import os
import pickle

import numpy as np
import torch
import tqdm
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from sklearn.cluster import KMeans


def sample(ids, n):
    if n == len(ids):
        return np.copy(ids)
    elif n > len(ids):
        return np.concatenate([ids, sample(ids, n - len(ids))])
    else:
        interval = len(ids) / n
        output = []
        for i in range(n):
            output.append(ids[int(interval * i)])
        return np.array(output)


def bbox_transform(bbox, matrix):
    # bbox: n, 9
    # matrix: 4, 4
    # output: n, 9
    if bbox.shape[0] == 0:
        return bbox
    if not isinstance(matrix, torch.Tensor):
        matrix = bbox.new_tensor(matrix)
    points = bbox[:, :3]
    constant = points.new_ones(points.shape[0], 1)
    points_extend = torch.concat([points, constant], dim=-1)
    points_trans = torch.matmul(points_extend, matrix.transpose(-2, -1))[:, :3]
    size = bbox[:, 3:6]
    ori_matrix = euler_angles_to_matrix(bbox[:, 6:], "ZXY")
    rot_matrix = matrix[:3, :3].expand_as(ori_matrix)
    final = torch.bmm(rot_matrix, ori_matrix)
    angle = matrix_to_euler_angles(final, "ZXY")

    bbox = torch.cat([points_trans, size, angle], dim=-1)
    return bbox


def kmeans(
    ann_file,
    output_file,
    z_min=-0.2,
    z_max=3,
):
    ann = pickle.load(open(ann_file, "rb"))
    all_cam_bbox = []
    for x in tqdm.tqdm(ann["data_list"]):
        bbox = np.array([y["bbox_3d"] for y in x["instances"]])
        ids = np.arange(len(x["images"]))
        ids = sample(ids, 50)
        for idx in ids:
            image = x["images"][idx]
            global2cam = np.linalg.inv(
                x["axis_align_matrix"] @ image["cam2global"]
            )

            _bbox = np.copy(bbox[image["visible_instance_ids"]])
            if _bbox.shape[0] == 0:
                continue
            mask = np.logical_and(_bbox[:, 2] > z_min, _bbox[:, 2] < z_max)
            _bbox = _bbox[mask]
            _bbox = bbox_transform(torch.Tensor(_bbox), global2cam).numpy()
            all_cam_bbox.append(_bbox)

    all_cam_bbox = np.concatenate(all_cam_bbox)
    print("start to kmeans, please wait")
    cluster_cam = KMeans(n_clusters=100).fit(all_cam_bbox).cluster_centers_
    cluster_cam[:, 3:6] = np.log(cluster_cam[:, 3:6])
    output_path = os.path.split(output_file)[0]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.save(output_file, cluster_cam)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="anchor bbox3d kmeans for embodiedscan dataset"
    )
    parser.add_argument("ann_file")
    parser.add_argument(
        "--output_file",
        default="./anchor_files/embodiedscan_kmeans_det_cam_log_z-0.2-3.npy",
    )
    parser.add_argument("--z_min", default=-0.2)
    parser.add_argument("--z_max", default=3)
    args = parser.parse_args()
    kmeans(args.ann_file, args.output_file, args.z_min, args.z_max)

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

from typing import Literal

from pydantic import BaseModel, Field


class TrainerParam(BaseModel):
    lr_scheduler_step_at: Literal["epoch", "step"] = Field(
        default="step",
        description="Frequency of updating the learning rate (in steps)",
    )
    max_step: int = Field(
        default=None, ge=1, description="Maximum number of steps for training"
    )
    max_epoch: int = Field(
        default=None, ge=1, description="Maximum number of epochs for training"
    )
    step_eval_freq: int = Field(
        default=None,
        ge=1,
        description="Frequency of evaluation during training (in steps)",
    )
    epoch_eval_freq: int = Field(
        default=None,
        ge=1,
        description="Frequency of evaluation during training (in epochs)",
    )
    resume_from: str = Field(
        default=None,
        description="Path to the checkpoint for resuming training",
    )
    resume_share_dir: str = Field(
        default=None,
        description="Path to the shared directory for resuming training",
    )
    grad_clip_mode: Literal["norm", "value"] = Field(
        default=None, description="Gradient clipping mode (norm or value)"
    )
    grad_clip_value: float = Field(
        default=None, ge=0.0, description="Maximum value for gradient clipping"
    )
    grad_max_norm: float = Field(
        default=None, ge=0.0, description="Maximum norm for gradient clipping"
    )
    grad_norm_type: int = Field(
        default=2, ge=1, description="Type of norm for gradient clipping"
    )
    train_split: str = Field(
        default="train",
        description="Dataset split to use for training",
    )
    test_split: str = Field(
        default="test",
        description="Dataset split to use for testing",
    )
    data_root: str = Field(
        default=None,
        description="Path to the root directory of the dataset",
    )
    lr: float = Field(
        default=0.008, ge=0.0, description="Learning rate for the optimizer"
    )
    train_batch_size: int = Field(
        default=4, ge=1, description="Batch size for training"
    )
    train_num_workers: int = Field(
        default=16,
        ge=0,
        description="Number of worker threads for data loading during training",  # noqa: E501
    )
    eval_batch_size: int = Field(
        default=16, ge=1, description="Batch size for evaluation"
    )
    eval_num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of worker threads for data loading during evaluation",  # noqa: E501
    )
    test_mode: list = Field(
        default=["test", "test_scale"],
        description="List of test modes to evaluate the model on",
    )
    save_epoch_freq: int = Field(
        default=None,
        ge=1,
        description="Frequency of evaluation during training (in epochs)",
    )
    save_step_freq: int = Field(
        default=None,
        ge=1,
        description="Frequency of saving checkpoints during training",
    )
    step_log_freq: int = Field(
        default=64,
        ge=1,
        description="Frequency of logging during training (in steps)",
    )
    world_size: int = Field(
        default=1,
        ge=1,
        description="Number of processes for distributed training",
    )


class DatasetParam(BaseModel):
    data_root: str = Field(
        ..., description="Path to the root directory of the dataset"
    )
    camera: Literal["realsense", "kinect"] = Field(
        default="realsense", description="Camera type used for capturing data"
    )
    num_sample_points: int = Field(
        default=20000,
        ge=1,
        description="Number of points to sample from the point cloud",
    )
    voxel_size: float = Field(
        default=0.005,
        ge=0.0,
        description="Voxel size used for point cloud voxelization",
    )
    remove_outlier: bool = Field(
        default=True,
        description="Whether to remove outlier points from the point cloud",
    )
    remove_invisible: bool = Field(
        default=True,
        description="Whether to remove invisible points from the dataset",
    )
    augment: bool = Field(
        default=True,
        description="Whether to apply data augmentation during training",
    )
    load_label: bool = Field(
        default=True, description="Whether to load labels for the dataset"
    )
    split: Literal[
        "train", "test", "test_seen", "test_similar", "test_novel"
    ] = Field(
        default="train",
        description="Dataset split to use (train, val, or test)",
    )
    batch_size: int = Field(
        default=16, ge=1, description="Batch size for training or evaluation"
    )
    num_workers: int = Field(
        default=16,
        ge=0,
        description="Number of worker threads for data loading",
    )
    use_new_graspness: bool = Field(
        default=False,
        description="Whether to use the new graspness calculation method",
    )


class OptimizerParam(BaseModel):
    lr: float = Field(
        default=0.00025, ge=0.0, description="Learning rate for the optimizer"
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        description="Weight decay (L2 regularization) factor",
    )
    warmup_epoch: int = Field(
        default=1,
        ge=0,
        description="Number of warmup epochs for learning rate scheduling",
    )


class MetricParam(BaseModel):
    voxel_size_cd: float = Field(
        default=0.005,
        ge=0.0,
        description="Voxel size used for collision detection",
    )
    collision_thresh: float = Field(
        default=0.01,
        ge=-1,
        description="Threshold for collision detection",
    )
    camera: Literal["realsense", "kinect"] = Field(
        default="realsense", description="Camera type used for capturing data"
    )
    test_mode: list = Field(
        default=["test", "test_seen", "test_similar", "test_novel"],
        description="List of test modes to evaluate the model on",
    )
    num_test_proc: int = Field(
        default=16,
        ge=1,
        description="Number of processes to use for testing",
    )
    grasp_max_width: float = Field(
        default=0.1,
        ge=0.0,
        description="Maximum gripper width for grasp generation",
    )
    num_seed_points: int = Field(
        default=4096,
        ge=1,
        description="Maximum number of points sampled from the point cloud",
    )
    data_root: str = Field(
        default=None,
        description="Path to the root directory of the dataset",
    )

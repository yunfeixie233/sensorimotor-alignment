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

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for the PickPlaceAgent.

    This class defines the configuration parameters for the PickPlaceAgent.

    Attributes:
        grasp_model_config_path (str): Path to the grasp model configuration.
        grounding_model_path (str): Path to the grounding model file.
        grounding_gpu_id (int): GPU ID for the grounding model.
        grounding_prompt (str): Prompt for the grounding model.

    """

    grasp_model_config: str = Field(
        default="grasp_model_config.py",
        description="Path to the grasp model configuration file.",
    )
    grasp_model_ckpt: str = Field(
        default="grasp_model.pth",
        description="Path to the grasp model checkpoint file.",
    )
    grounding_model_path: str = Field(
        default="Sa2VA-4B", description="Path to the grounding model file."
    )
    grounding_gpu_id: int = Field(
        default=1, description="GPU ID for the grounding model."
    )
    visualize_pose_after_workspace_filter: bool = Field(
        default=False,
        description="Whether to visualize the pose after workspace filtering.",
    )
    visualize_pose_after_grounding_filter: bool = Field(
        default=False,
        description="Whether to visualize the pose after grounding filtering.",
    )
    visualize_pose_after_topdown_filter: bool = Field(
        default=False,
        description="Whether to visualize the pose after top-down filtering.",
    )
    pose_model_path: str = Field(
        default="pose_model.pth",
        description="Path to the pose model checkpoint file.",
    )

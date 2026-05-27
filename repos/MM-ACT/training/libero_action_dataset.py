import os
import json
from typing import Any, Dict, List, Callable, Optional, Sequence
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from training.utils import image_transform_tensor
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
import re
from collections import defaultdict

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def quantize_state(values, bins: int = 1024) -> List[int]:
    """Map [-1,1] values to integer tokens in [0, bins-1],input MAST be 1-D"""
    tokens = []
    for v in values:
        v = max(-1.0, min(1.0, float(v)))
        idx = int(round((v + 1) / 2 * (bins - 1)))
        tokens.append(idx)
    return tokens


def quantize_action(actions: torch.Tensor, bins: int = 1024) -> torch.Tensor:
    actions = actions.to(torch.float32).clamp(-1.0, 1.0)
    return torch.round((actions + 1) / 2 * (bins - 1)).to(torch.long)


class ActionGenerationFromLeRobotDataset(LeRobotDataset):
    """
    Action-only Libero dataset in lerobot format,
    usable as a template for other lerobot-style robot datasets.
    """

    def __init__(
        self,
        prev_action_size: int,
        chunk_size: int,
        vocab_offset: int,
        action_vocab_size=2048,
        resolution: int = 256,
        dataset_fps: int = 10,
        third_image_name: str = "observation.images.image",
        wrist_image_name: list = ["observation.images.wrist_image"],
        action_dim: int = 7,
        use_prev_action: bool = True,
        use_norm: bool = True,
        libero_dataset: bool = False,
        *args,
        **kwargs,
    ):
        self.prev_action_size = int(prev_action_size)
        self.chunk_size = int(chunk_size)
        self.vocab_offset = int(vocab_offset)
        self.resolution = int(resolution)
        self.action_vocab_size = int(action_vocab_size)
        self.third_image_name = third_image_name
        self.wrist_image_name = wrist_image_name
        self.image_transform_tensor = image_transform_tensor
        self.dataset_fps = dataset_fps
        self.action_dim = action_dim
        self.use_prev_action = use_prev_action
        self.use_norm = use_norm
        self.libero_dataset = libero_dataset
        delta_action = [
            -(k) / self.dataset_fps for k in range(self.prev_action_size, 0, -1)
        ] + [  # [-0.6, ..., -0.1]
            i / self.dataset_fps for i in range(0, self.chunk_size)
        ]  # [0.0, 0.1, ..., 2.3]
        self.root_path = kwargs.get("root", None)
        if self.use_norm:
            stats_path = os.path.join(self.root_path, "meta", "stats.json")
            with open(stats_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.s_mean = np.asarray(s["observation.state"]["mean"], dtype=np.float32)
            self.s_std = np.asarray(s["observation.state"]["std"], dtype=np.float32)
            self.a_mean = np.asarray(s["action"]["mean"], dtype=np.float32)[:6]
            self.a_std = np.asarray(s["action"]["std"], dtype=np.float32)[:6]
            self._k = 3.0
            self._eps = 1e-8
        super().__init__(delta_timestamps={"action": delta_action}, *args, **kwargs)

    def _norm(self, x, mean, std):
        std = np.maximum(std, self._eps)
        y = (np.asarray(x, dtype=np.float32) - mean) / (self._k * std)
        return torch.as_tensor(np.clip(y, -1.0, 1.0), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = super().__getitem__(idx)
        text = base.get("task", "")
        images: List[torch.Tensor] = []
        if self.third_image_name:  # head-image also in this
            third_image = self.image_transform_tensor(
                base.get(self.third_image_name, None), self.resolution
            )
            images.append(third_image)
        for item in self.wrist_image_name:
            wrist_image = self.image_transform_tensor(
                base.get(item, None), self.resolution
            )
            images.append(wrist_image)
        if len(images) == 0:
            images.append(torch.zeros(3, self.resolution, self.resolution))
        state_vals = base.get("observation.state", [])
        if self.libero_dataset:
            state_vals[3:6] = torch.tensor(
                [x / 5.0 for x in state_vals[3:6]]
            )  # scale up state's in libero
        all_action_vals = base.get("action", None)
        action_vals = all_action_vals[-self.chunk_size :]
        all_action_pad = base.get("action_is_pad", None)[: self.prev_action_size]
        prev_action_vals = all_action_vals[: self.prev_action_size][~all_action_pad]
        if self.use_norm and not self.libero_dataset:
            state_vals = self._norm(state_vals, self.s_mean, self.s_std)
            action_vals[..., :6] = self._norm(
                action_vals[..., :6], self.a_mean, self.a_std
            )
            prev_action_vals[..., :6] = self._norm(
                prev_action_vals[..., :6], self.a_mean, self.a_std
            )
        state_tokens = (
            torch.tensor(
                quantize_state(state_vals, bins=self.action_vocab_size),
                dtype=torch.long,
            )
            + self.vocab_offset
        )
        # quantize continous actions to discrete action tokens
        raw_action_tokens = quantize_action(action_vals, bins=self.action_vocab_size)
        flat_action_tokens = [a for chunk in raw_action_tokens for a in chunk]

        action_tokens = (
            torch.tensor(flat_action_tokens, dtype=torch.long) + self.vocab_offset
        )
        if self.use_prev_action:
            raw_prev_action_tokens = quantize_action(
                prev_action_vals, bins=self.action_vocab_size
            )
            flat_prev_action_tokens = [
                a for chunk in raw_prev_action_tokens for a in chunk
            ]
            if len(flat_prev_action_tokens) > 0:
                prev_action_tokens = (
                    torch.tensor(flat_prev_action_tokens, dtype=torch.long)
                    + self.vocab_offset
                )
            else:
                prev_action_tokens = torch.tensor([], dtype=torch.long)
        else:
            prev_action_tokens = torch.tensor([], dtype=torch.long)
        return {
            "text": text,
            "images": images,
            "state_tokens": state_tokens,
            "prev_action_tokens": prev_action_tokens,
            "action_tokens": action_tokens,
            "action_dim": self.action_dim,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]):
        texts = [b["text"] for b in batch]
        images = [b["images"] for b in batch]
        state_tokens = [b["state_tokens"] for b in batch]
        prev_action_tokens = [b["prev_action_tokens"] for b in batch]
        action_tokens = [b["action_tokens"] for b in batch]
        action_dims = [b["action_dim"] for b in batch]
        return (
            images,
            texts,
            state_tokens,
            prev_action_tokens,
            action_tokens,
            action_dims,
        )


class ActionGenerationWithVqaFromLeRobotDataset(LeRobotDataset):
    """
    Similar to ActionGenerationFromLeRobotDataset, add libero task-planning annotations to dataset
    """

    def __init__(
        self,
        prev_action_size: int,
        chunk_size: int,
        vocab_offset: int,
        action_vocab_size=1024,
        resolution: int = 256,
        dataset_fps: int = 10,
        third_image_name: str = "observation.images.image",
        wrist_image_name: list = ["observation.images.wrist_image"],
        action_dim: int = 7,
        use_prev_action: bool = True,
        use_norm: bool = True,
        libero_dataset: bool = False,
        libero_vqa_dataset_path: str = "",
        *args,
        **kwargs,
    ):
        self.prev_action_size = int(prev_action_size)
        self.chunk_size = int(chunk_size)
        self.vocab_offset = int(vocab_offset)
        self.resolution = int(resolution)
        self.action_vocab_size = int(action_vocab_size)
        self.third_image_name = third_image_name
        self.wrist_image_name = wrist_image_name
        self.image_transform_tensor = image_transform_tensor
        self.dataset_fps = dataset_fps  # important
        self.action_dim = action_dim
        self.use_prev_action = use_prev_action
        self.use_norm = use_norm
        self.libero_dataset = libero_dataset

        self.vqa_data = defaultdict(dict)

        with open(
            libero_vqa_dataset_path, "r", encoding="utf-8"
        ) as f:  # open text annotation
            for line in f:
                record = json.loads(line.strip())

                ep_id = int(re.search(r"\d+", record["episode_id"]).group())
                frame_idx = record["frame_index"]

                self.vqa_data[ep_id][frame_idx] = record

        delta_action = [
            -(k) / self.dataset_fps for k in range(self.prev_action_size, 0, -1)
        ] + [  # [-0.6, ..., -0.1]
            i / self.dataset_fps for i in range(0, self.chunk_size)
        ]  # [0.0, 0.1, ..., 2.3]
        self.root_path = kwargs.get("root", None)
        if self.use_norm:
            stats_path = os.path.join(self.root_path, "meta", "stats.json")
            with open(stats_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.s_mean = np.asarray(s["observation.state"]["mean"], dtype=np.float32)
            self.s_std = np.asarray(s["observation.state"]["std"], dtype=np.float32)
            self.a_mean = np.asarray(s["action"]["mean"], dtype=np.float32)[:6]
            self.a_std = np.asarray(s["action"]["std"], dtype=np.float32)[:6]
            self._k = 3.0
            self._eps = 1e-8
        super().__init__(delta_timestamps={"action": delta_action}, *args, **kwargs)

    def _norm(self, x, mean, std):
        std = np.maximum(std, self._eps)
        y = (np.asarray(x, dtype=np.float32) - mean) / (self._k * std)
        return torch.as_tensor(np.clip(y, -1.0, 1.0), dtype=torch.float32)

    def process_description(self, vqa_info):
        description = (
            f"My task is to {vqa_info['task']}."
            f"I need to finish it by executing the following subtask: {vqa_info['all_subtasks']}."
            f"In my view, I can notice that the tasks I have completed are: {vqa_info['completed_subtasks']}."
        )
        description += (
            f"{vqa_info['cot_reasoning']}"
            f"So currently I need to {vqa_info['current_subtask']}."
        )

        return description

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = super().__getitem__(idx)
        text = base.get("task", "")
        episode_index = base["episode_index"]
        frame_index = base["frame_index"]
        vqa_info = self.vqa_data[int(episode_index)][int(frame_index)]
        description = self.process_description(vqa_info)

        images: List[torch.Tensor] = []
        if self.third_image_name:  # include head image
            third_image = self.image_transform_tensor(
                base.get(self.third_image_name, None), self.resolution
            )
            images.append(third_image)
        for item in self.wrist_image_name:
            wrist_image = self.image_transform_tensor(
                base.get(item, None), self.resolution
            )
            images.append(wrist_image)
        if len(images) == 0:
            images.append(torch.zeros(3, self.resolution, self.resolution))
        state_vals = base.get("observation.state", [])
        if self.libero_dataset:
            state_vals[3:6] = torch.tensor([x / 5.0 for x in state_vals[3:6]])
        all_action_vals = base.get("action", None)
        action_vals = all_action_vals[-self.chunk_size :]
        all_action_pad = base.get("action_is_pad", None)[: self.prev_action_size]
        prev_action_vals = all_action_vals[: self.prev_action_size][~all_action_pad]
        if self.use_norm and not self.libero_dataset:
            state_vals = self._norm(state_vals, self.s_mean, self.s_std)
            action_vals[..., :6] = self._norm(
                action_vals[..., :6], self.a_mean, self.a_std
            )
            prev_action_vals[..., :6] = self._norm(
                prev_action_vals[..., :6], self.a_mean, self.a_std
            )
        state_tokens = (
            torch.tensor(
                quantize_state(state_vals, bins=self.action_vocab_size),
                dtype=torch.long,
            )
            + self.vocab_offset
        )
        raw_action_tokens = quantize_action(action_vals, bins=self.action_vocab_size)
        flat_action_tokens = [a for chunk in raw_action_tokens for a in chunk]
        action_tokens = (
            torch.tensor(flat_action_tokens, dtype=torch.long) + self.vocab_offset
        )
        if self.use_prev_action:
            raw_prev_action_tokens = quantize_action(
                prev_action_vals, bins=self.action_vocab_size
            )
            flat_prev_action_tokens = [
                a for chunk in raw_prev_action_tokens for a in chunk
            ]
            if len(flat_prev_action_tokens) > 0:
                prev_action_tokens = (
                    torch.tensor(flat_prev_action_tokens, dtype=torch.long)
                    + self.vocab_offset
                )
            else:
                prev_action_tokens = torch.tensor([], dtype=torch.long)
        else:
            prev_action_tokens = torch.tensor([], dtype=torch.long)
        return {
            "text": text,
            "images": images,
            "state_tokens": state_tokens,
            "prev_action_tokens": prev_action_tokens,
            "action_tokens": action_tokens,
            "action_dim": self.action_dim,
            "description": description,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]):
        texts = [b["text"] for b in batch]
        images = [b["images"] for b in batch]
        state_tokens = [b["state_tokens"] for b in batch]
        prev_action_tokens = [b["prev_action_tokens"] for b in batch]
        action_tokens = [b["action_tokens"] for b in batch]
        action_dims = [b["action_dim"] for b in batch]
        descriptions = [b["description"] for b in batch]
        return (
            images,
            texts,
            state_tokens,
            prev_action_tokens,
            action_tokens,
            action_dims,
            descriptions,
        )

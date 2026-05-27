import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image



@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if (self.model_type == _model.ModelType.PI0 or self.model_type == _model.ModelType.PI05) else np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs



@dataclasses.dataclass(frozen=True)
class FrankaThreeViewInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        second_image = _parse_image(data["observation/second_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": second_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb":np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs




@dataclasses.dataclass(frozen=True)
class FrankaEEOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])} # 6 ee + gripper



@dataclasses.dataclass(frozen=True)
class FrankaJointsOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])} # 7 joint + gripper
    


@dataclasses.dataclass(frozen=True)
class FrankaAdaSmoothJointsOutputs(transforms.DataTransformFn):
    min_alpha: float = 0.1
    max_alpha: float = 0.6
    history_size: int = 10
    _history: list = dataclasses.field(default_factory=list, init=False)

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :8])
        H, D = actions.shape

        cur_std = np.std(actions)
        if self._history:
            hist_std = np.mean([np.std(h) for h in self._history])
        else:
            hist_std = cur_std

        ratio = cur_std / (hist_std + 1e-6)
        alpha = np.clip(self.max_alpha / (1 + ratio), self.min_alpha, self.max_alpha)

        smoothed = np.zeros_like(actions)
        smoothed[0] = actions[0]
        for t in range(1, H):
            smoothed[t] = alpha * actions[t] + (1 - alpha) * smoothed[t - 1]


        deltas = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
        mean_delta = np.mean(deltas) + 1e-6
        adjusted = smoothed.copy()
        for t in range(1, H):
            step = adjusted[t] - adjusted[t - 1]
            norm = np.linalg.norm(step)
            if norm > 1e-6:
                adjusted[t] = adjusted[t - 1] + step / norm * mean_delta

        self._history.append(smoothed)
        if len(self._history) > self.history_size:
            self._history.pop(0)

        return {"actions": adjusted.astype(np.float32)}
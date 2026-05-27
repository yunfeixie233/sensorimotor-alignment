import os
import cv2
from datetime import datetime
from typing import List, Optional  # noqa: UP035

import dm_env
import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from pure_exp_env import PureExpEnv

class FrankaRealEnvironment(_environment.Environment):

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,
        render_height: int = 224,
        render_width: int = 224,
        camera_num = 2,
        action_space = "joint",
        state_space = "joint",
        motion_mode = "abs",
        cal_euler_order = "zyx",
        task_name = "test",
        task_prompt = "pick up something",
        use_quat = False,
        use_degrees = False,
        start_wo_reset = False,
        final_retry_times = 0,
        convert_bgr_to_rgb = False,
        smooth_actions = False,
        **kwargs,
    ) -> None:

        assert action_space in ("joint", "ee"), f" You can't use {action_space} to control robot !!!"
        assert state_space in ("joint", "ee"), f" You can't use {state_space} as robot state !!!"
        assert motion_mode in ("abs", "rel"), f" You can't set motion mode to {motion_mode} !!!"
        assert camera_num >= 2, f" You need to use more than 2 cameras , instead of {camera_num} !!!"

        self._env = PureExpEnv(
                task_name = task_name,
                action_space = action_space,
                motion_mode = motion_mode,
                camera_num = camera_num,
                smooth_actions = smooth_actions,
                max_episode_steps = kwargs.get("max_episode_steps", None),
            )
        self.use_input_task_prompt = False
        self._camera_num = camera_num
        self._action_space = action_space
        self._state_space = state_space
        self._task_prompt = task_prompt
        self._render_height = render_height
        self._render_width = render_width
        self.convert_bgr_to_rgb = convert_bgr_to_rgb
        self.start_wo_reset = start_wo_reset
        self.check_imgs = False
        self._ts = None


    @override
    def reset(self) -> None:
        self._ts = self._env.reset(fake=self.start_wo_reset)


    @override
    def is_episode_complete(self) -> bool:
        if self._ts.step_type == dm_env.StepType.LAST:
             return True
        return False


    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation

        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k] 

        for cam_name in obs["images"]:
            # img = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width))
            img = image_tools.convert_to_uint8(obs["images"][cam_name])
            if self.convert_bgr_to_rgb:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs["images"][cam_name] = einops.rearrange(rgb, "h w c -> c h w")

        state =  obs["qpos"] if self._state_space == "joint" else obs["ee_pose"]
        state = np.append(state, obs["gripper_width"])

        if self.use_input_task_prompt:
            prompt = self._task_prompt
        else:
            prompt = obs.get("prompt", "Do something.")

        if self._camera_num == 3 :
            image = obs["images"]["camera_0"]
            wrist_image = obs["images"]["camera_1"]
            second_image = obs["images"]["camera_2"]
            input_obs = {
                "observation/image": image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "prompt": prompt,
                "observation/second_image": second_image,
            }

        elif self._camera_num == 2:
            image = obs["images"]["camera_0"]
            wrist_image = obs["images"]["camera_1"]
            input_obs = {
                "observation/image": image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "prompt": prompt,
            }
   
        if obs.get("reset_agent", False):
            input_obs.update({"reset_agent": True})

        if self.check_imgs:
            print(input_obs)
            self.save_imgs_to_check(input=input_obs, task_name=None)
            self.check_imgs = False

        return input_obs


    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])



    def save_imgs_to_check(self, input: dict, task_name: str = None) -> None:
        image = input.get("observation/image", None)
        wrist_image = input.get("observation/wrist_image", None)

        if image is None or wrist_image is None:
            print(" Warning: Missing image or wrist_image, skip saving.")
            return

        save_dir = os.path.join(os.getcwd(), "check_images")
        os.makedirs(save_dir, exist_ok=True)
        name = task_name if task_name is not None else "default"
        main_path = os.path.join(save_dir, f"{name}_main.png")
        wrist_path = os.path.join(save_dir, f"{name}_wrist.png")

        def to_bgr(img):
            if img.ndim == 3 and img.shape[0] in [3, 4]:  # CHW → HWC
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if self.convert_bgr_to_rgb:
            cv2.imwrite(main_path, to_bgr(image))
            cv2.imwrite(wrist_path, to_bgr(wrist_image))
        else:
            cv2.imwrite(main_path, image)
            cv2.imwrite(wrist_path, wrist_image)

        print(f"Saved example images to check:")
        print(f"   {main_path}")
        print(f"   {wrist_path}")
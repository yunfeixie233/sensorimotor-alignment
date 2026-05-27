import dataclasses
from typing_extensions import override
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias
import tyro

import openpi.transforms as _transforms
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.models.model as _model

import openpi.models.e0_diff_hybrid as e0_diff_hybrid


import openpi.policies.robotwin_aloha_policy as robotwin_aloha_policy

ModelType: TypeAlias = _model.ModelType


def get_robotwin_configs():
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import TrainConfig
    from openpi.training.config import DataConfigFactory, ModelTransformFactory
   
    @dataclasses.dataclass(frozen=True)
    class LeRobotRoboTwinAlohaDataConfig(DataConfigFactory):
        use_delta_joint_actions: bool = True
        default_prompt: str | None = None
        adapt_to_pi: bool = False

        # Repack transforms.
        repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=_transforms.Group(inputs=[
            _transforms.RepackTransform({
                "images": {
                    "cam_high": "observation.images.top"
                },
                "state": "observation.state",
                "actions": "action",
            })
        ]))
        # Action keys that will be used to read the action sequence from the dataset.
        action_sequence_keys: Sequence[str] = ("action", )

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            data_transforms = _transforms.Group(
                inputs=[robotwin_aloha_policy.RoboTwinAlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
                outputs=[robotwin_aloha_policy.RoboTwinAlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
            )
            if self.use_delta_joint_actions:
                delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
            )



    config_list = [
        TrainConfig(
            name="e0_diff_hybrid_robotwin",
            model=e0_diff_hybrid.E0DiffHybridConfig(
                action_dim = 32,
                action_horizon = 50,
                max_token_len = 200,
                bins = 2048,
                onehot_decay= 0.1,
                decay_noise = False,
                ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            data=LeRobotRoboTwinAlohaDataConfig(
                repo_id="robotwin_multitask",
                adapt_to_pi=False,
                assets=AssetsConfig(assets_dir="./assets",),
                repack_transforms=_transforms.Group(inputs=[
                    _transforms.RepackTransform({
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    })
                ]),
                base_config=DataConfig(prompt_from_task=True,),
            ),
        ),
    ]
    
    assert len({config.name for config in config_list}) == len(config_list)
    return config_list
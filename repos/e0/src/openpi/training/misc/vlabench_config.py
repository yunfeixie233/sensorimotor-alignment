import dataclasses
from typing_extensions import override
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import openpi.models.model as _model
import openpi.transforms as _transforms
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


import openpi.models.e0_diff_hybrid as e0_diff_hybrid

import openpi.policies.vlabench_policy as vlabench_policy


ModelType: TypeAlias = _model.ModelType

def get_vlabench_configs():
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import TrainConfig
    from openpi.training.config import DataConfigFactory, ModelTransformFactory


    @dataclasses.dataclass(frozen=True)
    class LeRobotVLABenchDataConfig(DataConfigFactory):
        extra_delta_transform: bool = True

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = _transforms.Group(
                inputs=[vlabench_policy.VLABenchInputs(model_type=model_config.model_type)],
                outputs=[vlabench_policy.VLABenchOutputs()],
            )

            if self.extra_delta_transform:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )
    

    @dataclasses.dataclass(frozen=True)
    class LeRobotVLABenchThreeViewDataConfig(DataConfigFactory):
        extra_delta_transform: bool = True
        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/second_image":"second_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = _transforms.Group(
                inputs=[vlabench_policy.VLABenchNewInputs(model_type=model_config.model_type)],
                outputs=[vlabench_policy.VLABenchOutputs()],
            )

            if self.extra_delta_transform:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )



    config_list =  [
        TrainConfig(
            name="e0_diff_hybrid_vlabench",
            model=e0_diff_hybrid.E0DiffHybridConfig(
                action_dim = 7,
                action_horizon = 50,
                max_token_len = 48, 
                bins = 2048,
                onehot_decay= 0.1, 
                decay_noise = False, 
                ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            data=LeRobotVLABenchThreeViewDataConfig(
                repo_id="vlabench",
                extra_delta_transform=True,
                base_config=DataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="./assets",),
            ),
        ),
       
    ]

    assert len({config.name for config in config_list}) == len(config_list)
    return config_list
import dataclasses
from typing_extensions import override
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

import openpi.models.e0_diff_hybrid as e0_diff_hybrid

import openpi.policies.franka_policy as franka_policy

ModelType: TypeAlias = _model.ModelType


def get_franka_configs():

    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import TrainConfig
    from openpi.training.config import DataConfigFactory, ModelTransformFactory
    

    @dataclasses.dataclass(frozen=True)
    class LeRobotFrankaDataConfig(DataConfigFactory):

        action_space : str = "joint" # ee  or joint
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


            print(f" =================== The action space for policy is {self.action_space} =================== ")
            if self.action_space == "ee":
                data_output_transform = franka_policy.FrankaEEOutputs()
            elif self.action_space == "joint":
                data_output_transform = franka_policy.FrankaJointsOutputs() 
            # data_output_transform  = franka_policy.FrankaAdaSmoothJointsOutputs()

            data_transforms = _transforms.Group(
                inputs=[franka_policy.FrankaInputs(model_type=model_config.model_type)],
                outputs=[data_output_transform],
            )

            if self.extra_delta_transform:
                if self.action_space == "ee":
                    delta_action_mask = _transforms.make_bool_mask(6, -1)
                elif self.action_space == "joint":
                    delta_action_mask = _transforms.make_bool_mask(7, -1)
    
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


    config_list = [

            TrainConfig(
                name="e0_diff_hybrid_franka",
                model=e0_diff_hybrid.E0DiffHybridConfig(
                        action_dim = 8,
                        action_horizon = 50,
                        max_token_len = 48,
                        bins = 2048,
                        onehot_decay= 0.1,
                        decay_noise = False,
                    ),
                num_train_steps=30000,
                optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
                ema_decay=0.999,
                weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
                data=LeRobotFrankaDataConfig(
                    repo_id="franka_short",
                    base_config=DataConfig(prompt_from_task=True,),
                    assets=AssetsConfig(assets_dir="./assets"),
                    action_space = "joint",
                    extra_delta_transform = True,
                    ),
            ),

        ]
    


    assert len({config.name for config in config_list}) == len(config_list)
    return config_list
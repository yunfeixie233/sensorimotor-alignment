import dataclasses
from typing_extensions import override
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import tyro
import etils.epath as epath

import openpi.models.model as _model
import openpi.transforms as _transforms
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

import openpi.models.e0_diff_hybrid as e0_diff_hybrid

import openpi.policies.libero_policy as libero_policy

ModelType: TypeAlias = _model.ModelType



def get_libero_configs():
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import TrainConfig
    from openpi.training.config import DataConfigFactory, ModelTransformFactory
 
    @dataclasses.dataclass(frozen=True)
    class E0LeRobotLiberoDataConfig(DataConfigFactory):
        extra_delta_transform: bool = False
        use_quantile_norm: bool | None = None

        @override
        def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
            asset_id = self.assets.asset_id or repo_id
            
            use_quantile_norm = (self.use_quantile_norm if self.use_quantile_norm is not None else (model_config.model_type != ModelType.PI0))
            return dataclasses.replace(
                self.base_config or DataConfig(),
                repo_id=repo_id,
                asset_id=asset_id,
                norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
                use_quantile_norm=use_quantile_norm,
            )

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
                ])

            data_transforms = _transforms.Group(
                inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
                outputs=[libero_policy.LiberoOutputs()],
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


    config_list = [
        TrainConfig(
            name="e0_diff_hybrid_libero",
            model=e0_diff_hybrid.E0DiffHybridConfig(
                action_dim = 32,
                action_horizon = 10,
                bins = 2048,
                onehot_decay= 0.1,
                max_token_len = 64,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            data=E0LeRobotLiberoDataConfig(
                repo_id="libero",
                base_config=DataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="./assets"),
                ),
        ),

    ]
    
    assert len({config.name for config in config_list}) == len(config_list)
    return config_list
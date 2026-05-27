import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip

from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from openpi.models.e0_base import make_attn_mask, posemb_sincos

logger = logging.getLogger("e0")

PALIGEMMA_VOCAB_SIZE = 257_152

@dataclasses.dataclass(frozen=True)
class E0DiffHybridConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    backbone_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    bins : int = 2048
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 48
    onehot_decay : float = 0.1
    decay_noise : bool = False
    alpha : int = 0.5
    sigma_min : float = 0.0 
    sigma_max : float = 1.0
    discrete_state_input: bool = True  # type: ignore

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.E0DiffHybrid

    @override
    def create(self, rng: at.KeyArrayLike) -> "E0DiffHybrid":
        return E0DiffHybrid(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec


    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.backbone_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)



class E0DiffHybrid(_model.BaseModel):
    def __init__(self, config: E0DiffHybridConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        backbone_variant = _gemma.get_config(config.backbone_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.onehot_decay = config.onehot_decay
        self.bins = config.bins
        self.decay_noise = config.decay_noise
        self.alpha = config.alpha
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max


        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[backbone_variant, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=backbone_variant.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.e0_action_in_proj = nnx.Linear(self.bins, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.e0_action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.e0_action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
    
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.e0_action_out_proj = nnx.Linear(action_expert_config.width, self.bins, rngs=rngs)


    @at.typecheck
    def tokenize(
        self, actions: at.Float[at.Array, "b h d"]
    ) -> at.Int[at.Array, "b hd"]:
        norm_actions = (actions + 1) / 2.0
        token_ids = jnp.floor(norm_actions * self.bins).astype(jnp.int32)
        token_ids = jnp.clip(token_ids, 0, self.bins - 1)
        b, h, d = token_ids.shape
        token_ids = token_ids.reshape(b, h * d)
        return token_ids

    @at.typecheck
    def extract_actions(
        self,
        tokens: at.Int[at.Array, "b *"],
    ) -> at.Float[at.Array, "b h d"]:
        action_horizon = self.action_horizon
        action_dim = self.action_dim
        tokens = tokens[:, :action_horizon * action_dim]
        actions = tokens.astype(jnp.float32) / self.bins * 2 - 1
        b = tokens.shape[0]
        actions = actions.reshape(b, action_horizon, action_dim)
        return actions


    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1],))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask


    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions, discrete_noisy_actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Float[at.Array, "b emb"] | None]:
        input_mask = []
        ar_mask = []
        tokens = []

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        e0_action_tokens = self.e0_action_in_proj(discrete_noisy_actions)
        e0_time_emb = posemb_sincos(timestep, self.e0_action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        e0_action_expert_tokens = e0_action_tokens
        adarms_cond = time_emb

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens.append(e0_action_expert_tokens)
        input_mask.append(jnp.ones(e0_action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon*self.action_dim - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond


    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        
        discrete_token_actions = self.tokenize(actions)
        discrete_actions_onehot = jax.nn.one_hot(discrete_token_actions, num_classes = self.bins) # b hd bins
        discrete_noise = jax.random.normal(noise_rng, discrete_actions_onehot.shape)
        discrete_x_t = time_expanded * discrete_noise + (1 - time_expanded) * discrete_actions_onehot
       
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, discrete_x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)

        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond])


        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon * (self.action_dim + 1) : -self.action_horizon * self.action_dim])
        action_logits = self.e0_action_out_proj(suffix_out[:, -self.action_horizon * self.action_dim :])

        mse_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        log_probs = jax.nn.log_softmax(action_logits, axis=-1)
        ce_loss = -jnp.sum(discrete_actions_onehot * log_probs, axis=-1)
        ce_loss = jnp.mean(ce_loss)
        
        loss = 0.5 * mse_loss + 0.5 * ce_loss
        return loss

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, *,
                    num_steps: int | at.Int[at.Array, ""] = 10, noise: at.Float[at.Array, "b ah ad"] | None = None,) -> _model.Actions:

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        x_t = noise
        discrete_x_t = jax.random.normal(rng, (batch_size, self.action_horizon * self.action_dim, self.bins))
        init_pred_ids = jnp.zeros((batch_size, self.action_horizon * self.action_dim), dtype=jnp.int32)
        time = jnp.array(1.0, dtype=jnp.float32)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        alpha = self.alpha
        sigma_min = self.sigma_min 
        sigma_max = self.sigma_max


        def step(carry):
            x_t, discrete_x_t, _, time, rng = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, discrete_x_t, jnp.broadcast_to(time, batch_size))

            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1] + suffix_tokens.shape[1])

            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon * (self.action_dim + 1) : -self.action_horizon * self.action_dim])
            
            action_logits = self.e0_action_out_proj(suffix_out[:, -self.action_horizon * self.action_dim :])
            pred_ids = jnp.argmax(action_logits, axis=-1)
            pred_onehot = jax.nn.one_hot(pred_ids, num_classes=self.bins)  # (b, h*d, bins)
            pred_onehot_decay = pred_onehot * self.onehot_decay
            onehot_noise = jax.random.normal(rng, pred_onehot.shape)

            if self.decay_noise:
                sigma_t = jnp.clip((time ** alpha) * (sigma_max - sigma_min) + sigma_min, sigma_min, sigma_max)
                sigma_t = sigma_t[..., None, None]
                discrete_x_t_next = (1.0 - sigma_t) * pred_onehot_decay + sigma_t * onehot_noise
            else:
                discrete_x_t_next = pred_onehot_decay + onehot_noise

            return (x_t + dt * v_t, discrete_x_t_next, pred_ids, time + dt, rng)

        def cond(carry):
            _, _, _, time, _ = carry
            return time >= -dt / 2

        x_0, _, pred_ids, _, _ = jax.lax.while_loop(cond, step, (x_t, discrete_x_t, init_pred_ids, time, rng))
        discrete_output_actions = self.extract_actions(pred_ids)

        output_actions = x_0 * 0.5 + discrete_output_actions * 0.5
        return output_actions




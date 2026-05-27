import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)



def make_structured_mask(input_mask, mask_ar, num_agents=2, dense_agents=False, *, append_agents: bool = True):
    """
    Structured mask builder.

    input_mask: token-level mask, shape [B, N] or [N]
    mask_ar: block-level autoreg indicator, shape (S,) or (B, S)
    num_agents: number of agent tokens (appended to sequence if append_agents=True)
    append_agents: whether to append agent tokens to input_mask/mask_ar inside this function.
                   If you are calling this function with pre-embedded tensors that DO NOT
                   include agents, pass append_agents=False.
    Returns: attn_mask shape [B, S_total, S_total] (token-level full attention mask).
    """
    input_mask = jnp.asarray(input_mask)
    mask_ar = jnp.asarray(mask_ar)

    # normalize dims
    if input_mask.ndim == 1:
        input_mask = input_mask[None, :]
    elif input_mask.ndim != 2:
        raise TypeError(f"input_mask must be 1D or 2D, got shape {input_mask.shape}")

    if mask_ar.ndim == 1:
        mask_ar = mask_ar[None, :]
    elif mask_ar.ndim == 2:
        pass
    else:
        raise TypeError(f"mask_ar must be 1D or 2D (block-level), got shape {mask_ar.shape}")

    B = input_mask.shape[0]
    # broadcast mask_ar batch if needed
    if mask_ar.shape[0] != B:
        if mask_ar.shape[0] == 1:
            mask_ar = jnp.broadcast_to(mask_ar, (B, mask_ar.shape[1]))
        else:
            raise ValueError("mask_ar batch-size does not match input_mask batch-size")

    # Optionally append agent tokens (token-level)
    if append_agents:
        agent_mask = jnp.ones((B, num_agents), dtype=input_mask.dtype)
        input_mask = jnp.concatenate([input_mask, agent_mask], axis=1)

        agent_mask_ar = jnp.concatenate(
            [jnp.ones((B, 1), dtype=mask_ar.dtype), jnp.zeros((B, max(0, num_agents - 1)), dtype=mask_ar.dtype)],
            axis=1,
        )
        mask_ar = jnp.concatenate([mask_ar, agent_mask_ar], axis=1)

    # block ids -> causal blocks
    block_ids = jnp.cumsum(mask_ar, axis=1)
    attn_mask = block_ids[:, None, :] <= block_ids[:, :, None]

    if dense_agents:
        same_block = block_ids[:, None, :] == block_ids[:, :, None]
        attn_mask = jnp.logical_or(attn_mask, same_block)

    # valid token mask
    valid_mask = input_mask[:, None, :] & input_mask[:, :, None]
    return attn_mask & valid_mask


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask





def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


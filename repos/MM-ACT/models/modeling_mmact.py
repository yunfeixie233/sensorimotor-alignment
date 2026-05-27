from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers import PretrainedConfig

from PIL import Image

from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .modeling_mmada import MMadaConfig, MMadaModelLM
from .sampling import cosine_schedule, mask_by_random_topk

log = logging.getLogger(__name__)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel-max trick for sampling from categorical distributions.

    For MDM-style discrete diffusion, low-precision Gumbel max can hurt quality,
    so we keep this in float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Pre-compute, for each sample and each diffusion step, how many tokens
    should be transitioned.

    Args:
        mask_index: (B, L) bool/0-1, 1 for tokens that can be updated.
        steps:     total number of reverse diffusion steps.

    Returns:
        (B, steps) int64 tensor with the number of tokens to transfer at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (B, 1)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


class MMACTConfig(MMadaConfig):
    """
    Config for the MMACT model.

    This reuses MMadaConfig and adds a few MMACT-specific fields while
    changing the HF ``model_type`` so that AutoConfig can distinguish it.
    """

    model_type = "mmact"

    def __init__(
        self,
        **kwargs,
    ):
        # Let the parent handle common fields
        super().__init__(**kwargs)


class MMACTModelLM(MMadaModelLM):
    """
    MMACT model.

    This class *inherits* from MMadaModelLM and adds the extra utilities from
    the modified MMada implementation (action decoding, multi-task
    forward, multi losses, etc.).  The original MMada implementation is left
    untouched.
    """

    config_class = MMACTConfig
    base_model_prefix = "model"

    def __init__(self, config: MMACTConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        # Reuse MMadaModelLM initialization
        super().__init__(config, *args, **kwargs)

    # ------------------------------------------------------------------
    # Action generation (MaskGIT-style over action tokens)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def action_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        prompt_id: Optional[int] = None,
        timesteps: int = 6,
        guidance_scale: float = 0.0,  # not used in our work
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
        generator: Optional[torch.Generator] = None,
        config: Optional[PretrainedConfig] = None,
        chunk_size: int = 24,
        action_dim: int = 7,
        mask_token_id: int = 126336,
        vocab_offset: int = 134656,
        action_vocab_size: int = 1024,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Parallel MaskGIT-style decoding for *action* tokens.
        """
        if prompt_id is None:
            raise ValueError("action_generate requires `prompt_id` (index of <soa>).")

        num_action_tokens = chunk_size * action_dim
        num_new_special_tokens = 0

        # Slice covering the action block
        action_slice = slice(prompt_id + 1, prompt_id + 1 + num_action_tokens)

        # Shift away vocab offset so we can work in [0, action_vocab_size)
        input_ids_minus_offset = input_ids[:, action_slice].clone()
        input_ids_minus_offset = torch.where(
            input_ids_minus_offset == mask_token_id,
            mask_token_id,
            input_ids_minus_offset - vocab_offset - num_new_special_tokens,
        )

        for step in range(timesteps):
            attention_bias = (
                (attention_mask[:, :, None] & attention_mask[:, None, :])
                .bool()
                .unsqueeze(1)
            )
            logits = self(input_ids, attention_bias=attention_bias).logits
            logits = logits[
                :,
                action_slice,
                vocab_offset : vocab_offset + action_vocab_size,
            ]
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, probs.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(
                *logits.shape[:-1]
            )

            unknown_map = input_ids_minus_offset == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_offset)

            ratio = float(step + 1) / float(timesteps)
            mask_ratio = noise_schedule(torch.tensor(ratio, device=input_ids.device))

            selected_probs = torch.gather(
                probs, -1, sampled_ids.long()[..., None]
            ).squeeze(-1)
            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                torch.finfo(selected_probs.dtype).max,
            )

            mask_len = (
                (num_action_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            )
            mask_len = torch.max(
                torch.tensor([1], device=logits.device),
                torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len),
            )

            temperature_step = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(
                mask_len, selected_probs, temperature_step, generator=generator
            )

            input_ids[:, action_slice] = torch.where(
                masking,
                mask_token_id,
                sampled_ids + vocab_offset + num_new_special_tokens,
            )
            input_ids_minus_offset = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    # ------------------------------------------------------------------
    # multi-task forward with action / t2i / lm / mmu
    # ------------------------------------------------------------------
    def forward_process_vla(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        batch_size_t2i: int = 0,
        batch_size_lm: int = 0,
        batch_size_mmu: int = 0,
        batch_size_action: int = 0,
        max_image_seq_length: int = 128,
        max_action_prompt_len: int = 768,
        action_masks: Optional[torch.Tensor] = None,
        action_loss_type: str = "single",  # "single" | "multi" | "gaussian"
        p_mask_lm: Optional[torch.Tensor] = None,
        p_mask_mmu: Optional[torch.Tensor] = None,
        answer_lengths_mmu: Optional[torch.Tensor] = None,
        t2i_masks: Optional[torch.Tensor] = None,
        answer_lengths_lm: Optional[torch.Tensor] = None,
        action_err_token_len: int = 10,
        at_value: float = 1e-3,
    ):
        """
        Forward pass that can handle four task types(three in use) in a single
        batch:

        - action generation (first ``batch_size_action`` samples)
        - text-to-image generation (next ``batch_size_t2i`` samples)
        - LM-only samples (next ``batch_size_lm`` samples)
        - multi-modal understanding (last ``batch_size_mmu`` samples)
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # -------- Build attention bias --------
        attention_bias = torch.ones(
            batch_size, 1, seq_len, seq_len, device=input_ids.device
        )

        # ACTION part (comes first)
        if batch_size_action > 0 and action_masks is not None:
            attention_bias_action = (
                (action_masks[:, :, None] & action_masks[:, None, :])
                .bool()
                .unsqueeze(1)
            )
            attention_bias[:batch_size_action] = attention_bias_action

        # T2I part (comes right after action part)
        if batch_size_t2i > 0 and t2i_masks is not None:
            attention_bias_t2i = (
                (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
            )
            attention_bias[batch_size_action : batch_size_action + batch_size_t2i] = (
                attention_bias_t2i
            )

        logits = self(input_ids, attention_bias=attention_bias).logits
        self.output_size = logits.shape[-1]

        # ------------------------------------------------------------------
        # Loss: ACTION
        # ------------------------------------------------------------------
        if batch_size_action > 0:
            action_logits = logits[
                :batch_size_action, max_action_prompt_len + 1 :
            ].contiguous()
            action_labels = labels[
                :batch_size_action, max_action_prompt_len + 1 :
            ].contiguous()

            if action_loss_type == "multi":
                loss_action = self.modified_ce_with_margin_rule(
                    action_logits,
                    action_labels,
                    ignore_index=-100,
                    action_err_token_len=action_err_token_len,
                )
            elif action_loss_type == "gaussian":
                loss_action = self.gaussian_soft_ce_window(
                    action_logits,
                    action_labels,
                    radius=int(action_err_token_len / 2.0),
                    at_value=at_value,
                    ignore_index=-100,
                )
            elif action_loss_type == "single":
                loss_action = F.cross_entropy(
                    action_logits.view(-1, self.output_size),
                    action_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                raise ValueError(f"Unsupported action_loss_type: {action_loss_type}")
        else:
            loss_action = torch.tensor(0.0, device=input_ids.device)

        # ------------------------------------------------------------------
        # Loss: T2I
        # ------------------------------------------------------------------
        if batch_size_t2i > 0:
            start_t2i = batch_size_action
            end_t2i = start_t2i + batch_size_t2i
            loss_t2i = F.cross_entropy(
                logits[start_t2i:end_t2i, max_image_seq_length + 1 :]
                .contiguous()
                .view(-1, self.output_size),
                labels[start_t2i:end_t2i, max_image_seq_length + 1 :]
                .contiguous()
                .view(-1),
                ignore_index=-100,
            )
        else:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)

        # ------------------------------------------------------------------
        # Loss: LM-only
        # ------------------------------------------------------------------
        if batch_size_lm > 0:
            masked_indices = input_ids == self.config.mask_token_id
            start_lm = batch_size_action + batch_size_t2i
            end_lm = start_lm + batch_size_lm

            masked_indices_lm = masked_indices[start_lm:end_lm]
            p_mask_lm = p_mask_lm.to(masked_indices_lm.device)

            per_token_loss = (
                F.cross_entropy(
                    logits[start_lm:end_lm][masked_indices_lm]
                    .contiguous()
                    .view(-1, self.output_size),
                    labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                / p_mask_lm[masked_indices_lm]
            )

            if answer_lengths_lm is not None:
                answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
                loss_lm = torch.sum(
                    per_token_loss / answer_lengths_lm[masked_indices_lm]
                ) / float(logits[start_lm:end_lm].shape[0])
            else:
                loss_lm = per_token_loss.sum() / float(
                    logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1]
                )
        else:
            loss_lm = torch.tensor(0.0, device=input_ids.device)

        # ------------------------------------------------------------------
        # Loss: MMU
        # ------------------------------------------------------------------
        if batch_size_mmu > 0:
            masked_indices = input_ids == self.config.mask_token_id
            masked_indices_mmu = masked_indices[-batch_size_mmu:]
            p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
            answer_lengths_mmu = answer_lengths_mmu.to(masked_indices_mmu.device)

            per_token_loss = (
                F.cross_entropy(
                    logits[-batch_size_mmu:][masked_indices_mmu]
                    .contiguous()
                    .view(-1, self.output_size),
                    labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                / p_mask_mmu[masked_indices_mmu]
            )

            loss_mmu = torch.sum(
                per_token_loss / answer_lengths_mmu[masked_indices_mmu]
            ) / float(logits[-batch_size_mmu:].shape[0])
        else:
            loss_mmu = torch.tensor(0.0, device=input_ids.device)

        return logits, loss_action, loss_t2i, loss_lm, loss_mmu

    # ------------------------------------------------------------------
    # Soft loss variants used for action training
    # ------------------------------------------------------------------
    def modified_ce_with_margin_rule(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
        action_err_token_len: int = 5,
    ) -> torch.Tensor:
        """
        Margin-based variant of cross-entropy.

        If the argmax token is within ``action_err_token_len`` of the ground-truth id,
        we treat argmax as correct (use its log-prob); otherwise we use the
        ground-truth token's log-prob.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        max_logprob, argmax_ids = torch.max(log_probs, dim=-1)  # [B, T], [B, T]

        valid = labels.ne(ignore_index)  # [B, T]
        safe_labels = labels.masked_fill(~valid, 0)

        true_logprob = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        use_max = argmax_ids.sub(safe_labels).abs().le(action_err_token_len) & valid
        chosen_logprob = torch.where(use_max, max_logprob, true_logprob)
        per_token_loss = -chosen_logprob.masked_fill(~valid, 0.0)

        if reduction == "mean":
            denom = valid.sum().clamp_min(1)
            return per_token_loss.sum() / denom
        if reduction == "sum":
            return per_token_loss.sum()
        if reduction == "none":
            return per_token_loss
        raise ValueError(f"Unsupported reduction: {reduction}")

    def gaussian_soft_ce_window(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        radius: int = 5,
        at_value: float = 1e-3,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Soft cross-entropy over a local Gaussian window around the ground-truth id.

        Tokens within +/- ``radius`` of the GT id receive non-zero mass according
        to a discrete Gaussian kernel whose tail value at the window edge is
        approximately ``at_value``.
        """
        B, T, V = logits.shape

        valid = labels.ne(ignore_index)  # [B, T]
        if valid.sum() == 0:
            return logits.sum() * 0.0

        sigma = radius / math.sqrt(2.0 * math.log(1.0 / max(at_value, 1e-12)))
        offsets = torch.arange(-radius, radius + 1, device=logits.device)  # [2r+1]
        base_w = torch.exp(-0.5 * (offsets.float() / max(sigma, 1e-6)) ** 2)  # [2r+1]

        idx = labels.unsqueeze(-1) + offsets  # [B, T, 2r+1]
        in_range = (idx >= 0) & (idx < V)
        in_range &= valid.unsqueeze(-1)

        idx_clamped = idx.clamp(0, V - 1)

        w = base_w.view(1, 1, -1).expand_as(idx).clone()
        w = torch.where(in_range, w, torch.zeros_like(w))

        z = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w = w / z

        logprob = F.log_softmax(logits, dim=-1)  # [B, T, V]
        logprob_win = torch.gather(logprob, dim=-1, index=idx_clamped)  # [B, T, 2r+1]

        loss_pos = -(w * logprob_win).sum(dim=-1)  # [B, T]
        loss = (loss_pos * valid.float()).sum() / valid.float().sum()
        return loss

    def forward_process_action(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.Tensor,
        max_seq_length: int = 768,
        action_loss_type: Optional[str] = None,  # "single" | "multi" | "gaussian"
        action_err_token_len: int = 6,
        at_value: float = 0.3,
    ):
        """
        Standalone forward for action training only.
        """
        attention_bias_action = (
            (attention_masks[:, :, None] & attention_masks[:, None, :])
            .bool()
            .unsqueeze(1)
        )
        logits = self(input_ids, attention_bias=attention_bias_action).logits

        self.output_size = logits.shape[-1]
        action_logits = logits[:, max_seq_length + 1 :].contiguous()
        action_labels = labels[:, max_seq_length + 1 :].contiguous()

        if action_loss_type == "multi":
            loss_action_modified = self.modified_ce_with_margin_rule(
                action_logits,
                action_labels,
                ignore_index=-100,
                action_err_token_len=action_err_token_len,
            )
            return logits, loss_action_modified
        if action_loss_type == "gaussian":
            loss_action_soft = self.gaussian_soft_ce_window(
                action_logits,
                action_labels,
                ignore_index=-100,
                radius=int(action_err_token_len / 2.0),
                at_value=at_value,
            )
            return logits, loss_action_soft

        # default: plain cross-entropy
        loss_action = F.cross_entropy(
            action_logits.view(-1, self.output_size),
            action_labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss_action

    # ------------------------------------------------------------------
    # MMU generation: single-block discrete diffusion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def mmu_generate_single_block(
        self,
        idx: Optional[torch.LongTensor] = None,  # [L_p] or [1, L_p]
        sequence_ids: Optional[torch.LongTensor] = None,  # [L] or [1, L]
        prompt_mask: Optional[
            torch.LongTensor
        ] = None,  # [L] or [1, L], 1=prompt,0=to-gen
        gen_len: Optional[int] = None,
        steps: int = 64,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",  # or "random"
        mask_id: int = 126336,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Single-sample, *unpadded* single-block discrete diffusion generation.

        Two calling patterns:

        1) Provide only ``idx`` (prompt).  Then you *must* give ``gen_len``.
        2) Provide ``sequence_ids`` and ``prompt_mask``:
           - we treat the prefix where prompt_mask==1 as the prompt
           - by default we generate for the remaining positions
             (``L - prefix_len``) unless a smaller ``gen_len`` is passed.
        """
        use_pair = (sequence_ids is not None) and (prompt_mask is not None)
        use_idx = idx is not None
        if not (use_pair ^ use_idx):
            raise ValueError(
                "Provide either (idx) or (sequence_ids + prompt_mask), but not both."
            )

        # Normalize to [1, L] tensors and determine prefix length
        if use_idx:
            if idx.ndim == 1:
                idx = idx.unsqueeze(0)
            elif idx.ndim != 2 or idx.size(0) != 1:
                raise ValueError("`idx` must be 1D or [1, L].")
            device = idx.device
            prefix_len = idx.size(1)
            if gen_len is None:
                raise ValueError("When using raw `idx`, you must specify `gen_len`.")
            Tnew = int(gen_len)
        else:
            if sequence_ids.ndim == 1:
                sequence_ids = sequence_ids.unsqueeze(0)
            elif sequence_ids.ndim != 2 or sequence_ids.size(0) != 1:
                raise ValueError("`sequence_ids` must be 1D or [1, L].")

            if prompt_mask.ndim == 1:
                prompt_mask = prompt_mask.unsqueeze(0)
            elif prompt_mask.ndim != 2 or prompt_mask.size(0) != 1:
                raise ValueError("`prompt_mask` must be 1D or [1, L].")

            device = sequence_ids.device
            L = sequence_ids.size(1)
            prefix_len = int((prompt_mask == 1).sum().item())

            if prefix_len < 0 or prefix_len > L:
                raise ValueError("Invalid prefix length inferred from `prompt_mask`.")

            idx = sequence_ids[:, :prefix_len]
            tail_len = L - prefix_len
            if tail_len <= 0:
                Tnew = 0
            else:
                Tnew = int(tail_len if gen_len is None else min(gen_len, tail_len))

        # Build attention bias if provided
        if attention_mask is not None and (attention_mask == 0).any():
            attention_bias = (
                (attention_mask[:, :, None] & attention_mask[:, None, :])
                .bool()
                .unsqueeze(1)
            )
        else:
            attention_bias = None

        # Nothing to generate -> return prompt as-is
        if Tnew <= 0:
            meta = {
                "idx_width": prefix_len,
                "per_sample_gen_len": torch.tensor(
                    [0], device=device, dtype=torch.long
                ),
            }
            return idx, meta

        # Initialize sequence: prompt + Tnew <MASK>
        x = torch.full((1, prefix_len + Tnew), mask_id, dtype=idx.dtype, device=device)
        x[:, :prefix_len] = idx

        # Allowed window for generation
        allowed_abs = torch.zeros_like(x, dtype=torch.bool)
        allowed_abs[:, prefix_len : prefix_len + Tnew] = True

        # Precompute how many tokens to transfer at each step
        num_transfer_tokens = get_num_transfer_tokens(
            allowed_abs[:, prefix_len:], steps
        )  # [1, steps]

        for t in range(steps):
            # classifier-free guidance over the whole sequence
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[:, :prefix_len] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                logits = self(x_cat).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = self(x, attention_bias=attention_bias).logits

            # Gumbel-max sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [1, L_all]

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float64)
            else:
                raise NotImplementedError(f"Unknown remasking mode: {remasking}")

            mask_index = x == mask_id
            selectable = allowed_abs & mask_index
            if not torch.any(selectable):
                break

            confidence = torch.full_like(x0_p, float("-inf"))
            confidence[selectable] = x0_p[selectable]

            k = int(num_transfer_tokens[0, t].item())
            k = min(k, int(selectable.sum().item()))
            if k > 0:
                _, sel = torch.topk(confidence[0], k)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                transfer_index[0, sel] = True
                x[transfer_index] = x0[transfer_index]

        meta = {
            "idx_width": prefix_len,
            "per_sample_gen_len": torch.tensor([Tnew], device=device, dtype=torch.long),
        }
        return x, meta


# Register with Hugging Face Auto* APIs
AutoConfig.register("mmact", MMACTConfig)
AutoModelForCausalLM.register(MMACTConfig, MMACTModelLM)
AutoModel.register(MMACTConfig, MMACTModelLM)

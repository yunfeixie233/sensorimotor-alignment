import os
import sys
import torch
from transformers import AutoTokenizer
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.prompting_utils import UniversalPrompting
from models import MMACTModelLM, MAGVITv2
from training.utils import image_transform_tensor


class MMACT_Deployment:
    def __init__(
        self,
        model_path: str,
        vq_model_path: str,
        action_vocab_size=None,
        vocab_offset=None,
        device="cuda:0",
        timesteps=6,
        exec_steps=6,
        preprocessing_max_seq_length=1024,
        training_chunk_size=8,
        action_dim=16,
        robot_type: str = "franka",
    ):
        self.image_transform_tensor = image_transform_tensor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", local_files_only=True
        )
        self.device = device
        self.timesteps = timesteps
        self.exec_steps = exec_steps
        self.preprocessing_max_seq_length = preprocessing_max_seq_length
        self.training_chunk_size = training_chunk_size
        self.action_dim = action_dim
        self.model = MMACTModelLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        print("Finish loading checkpoint. Start loading vq-model.")
        self.vq_model = MAGVITv2.from_pretrained(vq_model_path).to(self.device)
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)
        print("Finish loading vq-model.")
        self.vocab_offset = (
            vocab_offset
            if vocab_offset
            else self.model.config.vocab_size - self.model.config.action_vocab_size
        )
        self.action_vocab_size = (
            self.action_vocab_size
            if action_vocab_size
            else self.model.config.action_vocab_size
        )
        if robot_type == "franka":  # match training padding method
            max_action_prompt_len = (
                self.preprocessing_max_seq_length
                - self.training_chunk_size * (self.action_dim * 2)
                - 2
            )
        else:  # training total len - chunk_size * action_dim - <soa><eoa>
            max_action_prompt_len = (
                self.preprocessing_max_seq_length
                - self.training_chunk_size * self.action_dim
                - 2
            )

        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            special_tokens=(
                "<|soi|>",
                "<|eoi|>",
                "<|sov|>",
                "<|eov|>",
                "<|t2i|>",
                "<|mmu|>",
                "<|t2v|>",
                "<|v2v|>",
                "<|lvg|>",
                "<|mm2a|>",
                "<|soa|>",
                "<|eoa|>",
                "<|7dim|>",
                "<|14dim|>",
                "<|sostate|>",
                "<|eostate|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.0,
            use_reserved_token=True,
            max_action_prompt_len=max_action_prompt_len,
        )

    def image_process_for_generate(self, images):
        """
        In our experience, whether images are fed into the vq-model in batches or individually can slimly affect performance.
        The best results are achieved when the input method matches the training setup.
        """
        image_tokens = []
        for imgs in images:
            img_tokens = []
            for img in imgs:
                # img_tokens = []
                # for img in imgs:
                #     img = img.to(self.device)
                #     with torch.no_grad():
                #         tokens = self.vq_model.get_code(img.unsqueeze(0))[0]
                #     tokens = tokens + len(self.uni_prompting.text_tokenizer)
                #     img_tokens.append(tokens.cpu())
                # image_tokens.append(img_tokens)
                img = img.to(self.device)
                with torch.no_grad():
                    tokens = self.vq_model.get_code(img.unsqueeze(0))[0]
                tokens = tokens + len(self.uni_prompting.text_tokenizer)
                img_tokens.append(tokens.cpu())
            image_tokens.append(img_tokens)
        return image_tokens

    def quantize_state_with_offset(self, values, bins: int = 1024) -> List[int]:
        """Map [-1,1] values to integer tokens in [0, bins-1],input MAST be 1-D"""
        tokens = []
        for v in values:
            v = max(-1.0, min(1.0, float(v)))
            idx = int(round((v + 1) / 2 * (bins - 1))) + self.vocab_offset
            tokens.append(idx)
        return tokens

    def dequantize_action_with_offset(
        self, action_tokens, bins: int = 1024
    ) -> torch.Tensor:
        action_tokens = action_tokens.to(torch.int).clamp(0, bins - 1)
        return (action_tokens / (bins - 1) * 2) - 1

    def input_process(self, inputs):
        action_dim = [int(self.action_dim)]

        images_tensor, text_task, state_tensor, prev_action_tokens = inputs
        text_task = [text_task]
        prev_action_tokens = [prev_action_tokens + self.vocab_offset]
        state_tokens = [
            torch.tensor(
                self.quantize_state_with_offset(
                    state_tensor, bins=self.action_vocab_size
                )
            )
        ]
        reshape_images_tensor = [
            self.image_transform_tensor(image_tensor) for image_tensor in images_tensor
        ]
        image_tokens = self.image_process_for_generate([reshape_images_tensor])
        input_ids, attention_masks, prompt_ids = self.uni_prompting(
            (
                image_tokens,
                text_task,
                state_tokens,
                prev_action_tokens,
                action_dim,
                self.device,
                self.training_chunk_size,
            ),
            "mm2a_gen",
        )
        return input_ids, attention_masks, prompt_ids[0]

    def get_actions(self, inputs):
        """
        Your inputs should include
        images_tensor(List,[head_image, wrist_image]), text_task, state_tensor,previous_action_tokens([] if not used in training)
        """
        input_ids, attention_masks, prompt_id = self.input_process(inputs)
        gen_token_ids = self.model.action_generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            timesteps=self.timesteps,
            guidance_scale=0,
            chunk_size=self.training_chunk_size,
            action_dim=self.action_dim,
            prompt_id=(prompt_id),
            uni_prompting=self.uni_prompting,
            temperature=0.0,
            action_vocab_size=self.action_vocab_size,
        )
        action_chunk = self.dequantize_action_with_offset(
            gen_token_ids, bins=self.action_vocab_size
        ).view(self.training_chunk_size, self.action_dim)

        return (
            action_chunk,
            gen_token_ids.view(self.training_chunk_size, self.action_dim)[0],
        )

# import packages and module here
import os
import sys
import numpy as np
from PIL import Image
import torch
import time
from transformers import AutoTokenizer
import torch.nn.functional as F
from typing import List
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.utils import image_transform_tensor
from training.prompting_utils import UniversalPrompting
from models import MMACTModelLM, MAGVITv2


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
        robot_type: str = "aloha-agilex",
        ignore_state=False,
        t2i_ignore_state=True,
        store_t2i=False,
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
        self.store_t2i = store_t2i
        self.get_action_num = 0
        self.get_actions_total_time = 0.0  # Total time elapsed (seconds)
        self.get_actions_call_count = 0  # Number of calls
        self.last_get_actions_time = 0.0  # Last call time (seconds)
        self.ignore_state_config = type(
            "cfg",
            (),
            {
                "training": type(
                    "t",
                    (),
                    {
                        "ignore_state": bool(ignore_state),
                        "t2i_ignore_state": bool(t2i_ignore_state),
                    },
                )()
            },
        )()
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

    def input_process(self, inputs, robot_type="aloha-agilex"):
        # Set action_dim based on robot_type
        if robot_type == "franka":
            action_dim = [7]  # Franka robot has 7 DOF
        elif robot_type == "aloha-agilex":
            action_dim = [
                int(self.action_dim)
            ]  # ALOHA-AgileX robot has 14 DOF (7 for each arm)
        else:
            raise ValueError(
                f"Unsupported robot_type: {robot_type}. Supported types: 'franka', 'aloha-agilex'"
            )
        images_tensor, text_task, state_tensor, prev_action_tokens = inputs
        text_task = [
            text_task
        ]  # mm2a_gen process for batch, so list type;match mm2a_gen process, can be refine in the future
        prev_action_tokens = [prev_action_tokens + self.vocab_offset]
        state_tokens = [
            torch.tensor(
                self.quantize_state_with_offset(
                    state_tensor, bins=self.action_vocab_size
                )
            )
        ]  # mm2a_gen process for batch, so list type
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
                self.training_chunk_size,  # chunk size
            ),
            "mm2a_gen",
            config=self.ignore_state_config,
        )

        if self.store_t2i:
            input_ids_t2i, attention_masks_t2i = self.uni_prompting(
                (
                    image_tokens,
                    text_task,
                    state_tokens,
                    prev_action_tokens,
                    action_dim,
                    self.device,
                    self.training_chunk_size,  # chunk size
                ),
                "t2i_action_gen",
                config=self.ignore_state_config,
            )
            with torch.no_grad():
                gen_token_ids = self.model.t2i_generate(
                    input_ids=input_ids_t2i,
                    attention_mask=attention_masks_t2i,
                    guidance_scale=0,
                    temperature=0,
                    timesteps=18,
                    # noise_schedule=mask_schedule,
                    # noise_type=config.training.get("noise_type", "mask"),
                    seq_len=256,
                    uni_prompting=self.uni_prompting,
                    # config=config,
                )
                gen_token_ids = torch.clamp(gen_token_ids, max=8191, min=0)
                images = self.vq_model.decode_code(gen_token_ids)

                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]
                # INSERT_YOUR_CODE
                # Save the generated images to the specified directory
                pil_images[0].save(
                    f"./generated_images/{self.get_action_num}_generate.png"
                )
        return input_ids, attention_masks, prompt_ids[0], action_dim

    def get_actions(self, inputs, robot_type="franka"):
        """
        your inputs should include
        "images_tensor(List,[head_image, wrist_image]), text_task, state_tensor,
        previous_action_tokens(None if at the beginning),"
        """
        start_time = time.time()
        self.get_action_num += 1
        if self.store_t2i:
            original_images = inputs[0][0]
            original_images = cv2.cvtColor(
                original_images.cpu().numpy(), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(
                f"./generated_images/{self.get_action_num}_original.png",
                original_images,
            )
        input_ids, attention_masks, prompt_id, action_dim = self.input_process(
            inputs, robot_type=robot_type
        )
        # print("TIMESTEP:", self.timesteps)
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

        elapsed = time.time() - start_time
        self.last_get_actions_time = elapsed
        self.get_actions_total_time += elapsed
        self.get_actions_call_count += 1
        avg_time = self.get_actions_total_time / self.get_actions_call_count

        # print(
        #     f"[get_actions] this call: {elapsed*1000:.2f} ms, "
        #     f"avg: {avg_time*1000:.2f} ms over {self.get_actions_call_count} calls"
        # )
        return (
            action_chunk,
            gen_token_ids.view(self.training_chunk_size, self.action_dim)[0],
        )

    # now, pre_action_tokens will return together when use get_actions


def encode_obs(observation):  # Post-Process Observation
    # observation: "observation", "pointcloud", "joint_action", "endpose"
    obs = {}
    obs["images_tensor"] = []
    obs["state_tensor"] = []
    images = observation["observation"]
    camera_names = ["head_camera", "left_camera", "right_camera"]

    for image_name in camera_names:
        image_data = images[image_name]
        obs["images_tensor"].append(torch.tensor(image_data["rgb"]))
    for pose_name, pose_data in observation["endpose"].items():
        if np.isscalar(pose_data):
            obs["state_tensor"].append(float(pose_data))
        else:
            obs["state_tensor"].extend(pose_data)
    obs["state_tensor"] = torch.tensor(obs["state_tensor"])
    obs["joint_state_tensor"] = torch.tensor(observation["joint_action"]["vector"])
    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    print("from depoly_policy:", usr_args.get("ignore_state", False))
    model = MMACT_Deployment(
        model_path=usr_args["model_path"],
        vq_model_path=usr_args["vq_model_path"],
        action_vocab_size=usr_args.get("action_vocab_size", None),
        vocab_offset=usr_args.get("vocab_offset", None),
        device=usr_args.get("device", "cuda:0"),
        timesteps=usr_args.get("timesteps", 6),
        exec_steps=usr_args.get("exec_steps", 6),
        preprocessing_max_seq_length=usr_args.get("preprocessing_max_seq_length", 1024),
        training_chunk_size=usr_args.get("training_chunk_size", 8),
        action_dim=usr_args.get("action_dim", 16),
        ignore_state=usr_args.get("ignore_state", False),
        t2i_ignore_state=usr_args.get("t2i_ignore_state", False),
        store_t2i=usr_args.get("store_t2i", False),
        robot_type=usr_args.get("robot_type", "aloha-agilex"),
    )
    return model  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    image_tensor = obs["images_tensor"]
    text_task = instruction
    state_tensor = obs["state_tensor"]

    flat_prev_actions_tensors = torch.tensor([])
    inputs = (image_tensor, text_task, state_tensor, flat_prev_actions_tensors)

    action_chunk, token_ids = model.get_actions(
        inputs, robot_type="aloha-agilex"
    )  # Get Action according to observation chunk
    action_chunk = action_chunk.detach().cpu()
    for action in action_chunk:  # Execute each step of the action
        TASK_ENV.take_action(
            action, action_type="ee"
        )

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)


def reset_model(model):
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass

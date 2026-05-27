# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool

EXAMPLE_PROMPT = {
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


class SingleItemDataset(Dataset):
    """Dataset for single item generation (CLI arguments)."""
    def __init__(self, prompt, image_path):
        self.prompt = prompt
        self.image_path = image_path

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Return path, load in main loop to handle polymorphism (Image obj vs path str)
        return {
            "prompt": self.prompt,
            "media_path": self.image_path if self.image_path else "",
            "filename_stem": None # Auto-generate based on time/prompt
        }


class JsonDataset(Dataset):
    def __init__(self, dataset_json):
        import json
        with open(dataset_json, "r") as f:
            self.items = json.load(f)
        logging.info(f"Found {len(self.items)} items in {dataset_json}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.dataset_json:
        assert args.output_dir is not None, "output_dir is required when using dataset_json"
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
    else:
        # Fallback to single item defaults
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
            args.image = EXAMPLE_PROMPT[args.task]["image"]

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    # New arguments for batch processing
    parser.add_argument(
        "--dataset_json",
        type=str,
        default=None,
        help="Path to the dataset directory (containing metas/ and videos/). If set, runs in batch mode."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated files. Required if dataset_json is set."
    )
    
    # Original arguments
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        help="How many frames of video are used as conditional frames. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--pt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint of .pt file.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=False,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to (only for single mode).")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from (only for single mode).")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    args, left_args = parser.parse_known_args()

    _validate_args(args)

    return args, left_args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    assert not args.use_prompt_extend, "Prompt extend is not supported in causal generation."

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Initialize Dataset
    if args.dataset_json:
        dataset = JsonDataset(args.dataset_json)
    else:
        dataset = SingleItemDataset(args.prompt, args.image)
    
    # Initialize Sampler and DataLoader
    # Note: If ulysses_size > 1, all GPUs work together on a single batch item (Sequence Parallel).
    #       So they must all receive the SAME data. We use SequentialSampler (default) or no sampler, and shuffle=False.
    #       If ulysses_size == 1 (Data Parallel), each GPU works on DIFFERENT data. We use DistributedSampler.
    sampler = None
    if dist.is_initialized() and args.ulysses_size == 1:
        sampler = DistributedSampler(dataset, shuffle=False)
    
    # Use batch_size=1
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=0)

    logging.info("Creating WanTI2VCausal pipeline.")
    wan_ti2v = wan.WanTI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        pt_dir=args.pt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    logging.info(f"Generating video ...")

    for batch_idx, batch_data in enumerate(dataloader):
        # Unwrap batch data (since bs=1, take index 0)
        prompt = batch_data["prompt"][0]
        media_path = batch_data["media_path"][0] if batch_data["media_path"][0] else None
        file_stem = batch_data["filename_stem"][0] if batch_data["filename_stem"][0] else None

        if rank == 0:
            logging.info(f"Processing: {prompt[:50]}...")
            if media_path:
                logging.info(f"With media: {media_path}")

        video = wan_ti2v.generate(
            prompt,
            img=media_path,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            num_conditional_frames=args.num_conditional_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model
        )

        # Determine output filename
        # Only rank 0 needs to determine path and save in Ulysses mode.
        # In DDP mode (ulysses_size=1), every rank saves its own work.
        if args.ulysses_size == 1 or rank == 0:
            if args.save_file and not args.dataset_json:
                save_path = args.save_file
            else:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                if file_stem:
                    # Batch mode: use file stem from metadata
                    out_name = f"{file_stem}_{formatted_time}.mp4"
                else:
                    # Single mode auto-name
                    formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
                    suffix = '.mp4'
                    out_name = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
                
                if args.output_dir:
                    save_path = os.path.join(args.output_dir, out_name)
                else:
                    save_path = out_name

            logging.info(f"Saving generated video to {save_path}")
            save_video(
                tensor=video[None],
                save_file=save_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        
        del video
        
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args, _ = _parse_args()
    generate(args)
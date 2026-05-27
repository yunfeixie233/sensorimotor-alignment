import math
import random
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple, Union


##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [
            (f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)
        ]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [
            (f"{key}.{idx}", v1)
            for idx, v1 in flatten_omega_conf(value, resolve=resolve)
        ]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              training utils
##################################################
def soft_target_cross_entropy(logits, targets, soft_targets):
    # ignore the first token from logits and targets (class id token)
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(
    image_tokens, mask_id, config, mask_schedule, is_train=True, seed=None
):
    batch_size, seq_len = image_tokens.shape

    if not is_train and seed is not None:
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)
        # print(f"Set seed to {seed}")

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=image_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get(
        "mask_contiguous_region_prob", None
    )

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(
            batch_size, seq_len, device=image_tokens.device
        ).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len**0.5)
        mask = torch.zeros(
            (batch_size, resolution, resolution), device=image_tokens.device
        )

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution),
                min(resolution, num_token_masked_),
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(
                num_token_masked_ / num_token_masked_height
            )
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
                batch_idx,
                start_idx_height : start_idx_height + num_token_masked_height,
                start_idx_width : start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens,
            low=0,
            high=config.model.codebook_size,
            device=image_tokens.device,
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
        config.training.get("predict_all_image_tokens", False)
        or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    if not is_train and seed is not None:
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        random.setstate(python_rng_state)

    return input_ids, labels, loss_weight, mask_prob


def mask_or_random_replace_action_tokens(
    action_tokens_list,
    mask_id,
    pad_id,
    config,
    mask_schedule,
    action_vocab_size,
    is_train=True,
    seed=None,
):
    """Per-sample masking within true lengths, then pad to max_len."""
    batch_size = len(action_tokens_list)
    device = action_tokens_list[0].device if batch_size > 0 else torch.device("cpu")
    lengths = torch.tensor(
        [t.size(0) for t in action_tokens_list], device=device, dtype=torch.long
    )
    max_len = int(lengths.max().item() if batch_size > 0 else 0)

    # Pad to max_len
    tokens = torch.full((batch_size, max_len), pad_id, device=device, dtype=torch.long)
    for i, seq in enumerate(action_tokens_list):
        if seq.numel() > 0:
            tokens[i, : seq.size(0)] = seq.to(device)

    # ---- deterministic eval (optional) ----
    if not is_train and seed is not None:
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)

    if (not is_train) and config.training.get("eval_mask_ratios", None):
        pool = config.training.eval_mask_ratios
        mask_prob = torch.tensor(
            random.choices(pool, k=batch_size), device=device, dtype=torch.float32
        )
    else:
        timesteps = torch.rand(batch_size, device=device)
        mask_prob = mask_schedule(timesteps).to(device)
        mask_prob = torch.clamp(mask_prob, min=float(config.training.min_masking_rate))
    num_token_masked = (lengths.to(torch.float32) * mask_prob).round().to(torch.long)
    num_token_masked = torch.where(
        lengths > 0,
        torch.clamp(num_token_masked, min=1),
        torch.zeros_like(num_token_masked),
    )

    if max_len == 0:
        valid_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        mask = valid_mask
    else:
        valid_mask = torch.arange(max_len, device=device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        r = torch.rand(batch_size, max_len, device=device)
        r_invalid_filled = r.masked_fill(~valid_mask, 2.0)
        sorted_r, _ = torch.sort(r_invalid_filled, dim=-1, descending=False)
        kth_idx = torch.clamp(num_token_masked, min=1) - 1
        row_idx = torch.arange(batch_size, device=device)
        kth_val = sorted_r[row_idx, kth_idx.clamp(min=0)]
        threshold = torch.where(
            num_token_masked > 0, kth_val, torch.full_like(kth_val, -1.0)
        )
        mask = (r <= threshold.unsqueeze(1)) & valid_mask
    noise_type = config.training.get("noise_type", "mask")
    if noise_type == "mask":
        input_ids = torch.where(
            mask, torch.as_tensor(mask_id, device=device, dtype=torch.long), tokens
        )
    elif noise_type == "random_replace":
        random_tokens = torch.randint(
            0, action_vocab_size, (batch_size, max_len), device=device, dtype=torch.long
        )
        input_ids = torch.where(mask, random_tokens, tokens)
    else:
        raise ValueError(f"noise_type {noise_type} not supported")

    if (
        config.training.get("predict_all_tokens", False)
        or noise_type == "random_replace"
    ):
        labels = tokens.clone()
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, tokens, torch.full_like(tokens, -100))
        loss_weight = None
    pad_mask = tokens.eq(pad_id)
    input_ids[pad_mask] = pad_id
    labels[pad_mask] = -100

    if not is_train and seed is not None:
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        random.setstate(python_rng_state)

    return input_ids, labels, loss_weight, mask_prob


##################################################
#              misc
##################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from torchvision import transforms


def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(
        resolution, interpolation=transforms.InterpolationMode.BICUBIC
    )(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )(image)
    return image


def image_transform_tensor(
    x: torch.Tensor,
    resolution: int = 256,
    normalize: bool = True,
    square_mode: str = "pad",  # "crop" or "pad"
    inplace: bool = False,
) -> torch.Tensor:
    t = x
    if t.ndim != 3:
        raise ValueError(f"Expect 3D tensor, got {t.shape}")

    # HWC -> CHW
    if t.shape[0] not in (1, 3):
        t = t.permute(2, 0, 1)

    if t.dtype != torch.float32:
        t = t.float()
    if t.max() > 1.0:
        t = t / 255.0

    C, H, W = t.shape

    if square_mode == "crop":
        scale = resolution / min(H, W)
        H2 = max(1, int(round(H * scale)))
        W2 = max(1, int(round(W * scale)))
        try:
            t = F.interpolate(
                t.unsqueeze(0),
                size=(H2, W2),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
        except TypeError:
            t = F.interpolate(
                t.unsqueeze(0), size=(H2, W2), mode="bicubic", align_corners=False
            )[0]
        top = max((H2 - resolution) // 2, 0)
        left = max((W2 - resolution) // 2, 0)
        t = t[:, top : top + resolution, left : left + resolution]
    elif square_mode == "pad":
        side = max(H, W)
        pad_h = side - H
        pad_w = side - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h > 0 or pad_w > 0:
            t = F.pad(
                t,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            )
        try:
            t = F.interpolate(
                t.unsqueeze(0),
                size=(resolution, resolution),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
        except TypeError:
            t = F.interpolate(
                t.unsqueeze(0),
                size=(resolution, resolution),
                mode="bicubic",
                align_corners=False,
            )[0]
    else:
        raise ValueError("square_mode must be 'crop' or 'pad'")

    if normalize:
        if inplace:
            t.sub_(0.5).div_(0.5)
        else:
            t = (t - 0.5) / 0.5

    return t


def image_transform_squash(image, resolution=256, normalize=True):
    image = transforms.Resize(
        (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC
    )(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )(image)
    return image

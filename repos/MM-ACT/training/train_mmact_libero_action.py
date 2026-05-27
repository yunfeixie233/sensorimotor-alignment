import os
import sys
from pathlib import Path
import torch
import logging
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from accelerate.logging import get_logger
from transformers import AutoTokenizer

from tqdm import tqdm
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import get_config, mask_or_random_replace_action_tokens
from training.prompting_utils import UniversalPrompting
from training.libero_action_dataset import ActionGenerationFromLeRobotDataset
from models import MMACTModelLM, MAGVITv2, get_mask_schedule
from models.configuration_llada import ModelConfig
from models.lr_schedulers import get_scheduler


def setup_logger(log_file: Path, is_main_process: bool) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    if is_main_process:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def main():
    config = get_config()
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.get(
            "gradient_accumulation_steps", 1
        ),
        mixed_precision=config.training.get("mixed_precision", "bf16"),
        log_with="tensorboard",
        project_dir=str(logging_dir),
        split_batches=True,
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = config.training.batch_size
    accelerator.init_trackers(
        project_name="mmact_train",
        config={
            "batch_size": config.training.batch_size,
            "lr": config.optimizer.params.learning_rate,
            "epochs": config.training.num_epochs,
        },
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.mmact.pretrained_model_path,
        padding_side="left",
        local_files_only=True,
    )
    model = MMACTModelLM.from_pretrained(
        config.model.mmact.pretrained_model_path, torch_dtype=torch.bfloat16
    )
    model.to(accelerator.device)
    action_vocab_size = model.config.action_vocab_size
    # libero(franka) action dim is 7, we pad to 14 for co-training other robots in the future
    max_action_prompt_len = (
        config.dataset.preprocessing.max_seq_length
        - config.training.chunk_size * 14
        - 2  # <soa><eoa>
    )

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_action_prompt_len=max_action_prompt_len,
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
        action_vocab_size=action_vocab_size,
    )

    # load VQ model for image tokenization
    vq_cfg = config.model.vq_model
    if vq_cfg.type == "magvitv2":
        vq_model = MAGVITv2.from_pretrained(vq_cfg.vq_model_name).to(accelerator.device)
    else:
        raise ValueError(f"Unsupported vq model type: {vq_cfg.type}")
    vq_model.eval()
    vq_model.requires_grad_(False)
    vocab_offset = model.config.vocab_size - model.config.action_vocab_size
    mask_id = model.config.mask_token_id
    mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))
    if not hasattr(config.training, "min_masking_rate"):
        config.training.min_masking_rate = 0.0

    dataset_file_paths = config.dataset.dataset_file_paths

    libero_dataset = [
        ActionGenerationFromLeRobotDataset(
            prev_action_size=6,
            chunk_size=config.training.chunk_size,
            vocab_offset=vocab_offset,
            action_vocab_size=action_vocab_size,
            dataset_fps=10,
            third_image_name="observation.images.image",
            wrist_image_name=["observation.images.wrist_image"],
            use_prev_action=config.training.use_prev_action,
            action_dim=7,
            use_norm=False,
            libero_dataset=True,
            repo_id=".",
            root=dataset_file_path,
            force_cache_sync=False,
            revision=None,
            download_videos=False,
        )
        for dataset_file_path in dataset_file_paths
    ]

    libero_mix_dataset = ConcatDataset(libero_dataset)
    dataloader = DataLoader(
        libero_mix_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.dataprocess_nums,
        shuffle=True,
        collate_fn=ActionGenerationFromLeRobotDataset.collate_fn,
    )

    def prepare_inputs_and_labels(batch, is_train=True, seed=None):
        images, texts, state_tokens, prev_action_tokens, action_tokens, action_dims = (
            batch
        )
        image_tokens = []
        for imgs in images:
            img_tokens = []
            for img in imgs:
                img = img.to(accelerator.device)
                with torch.no_grad():
                    tokens = vq_model.get_code(img.unsqueeze(0))[0]
                tokens = tokens + len(uni_prompting.text_tokenizer)
                img_tokens.append(tokens)
            image_tokens.append(img_tokens)
        action_tokens = [a.to(accelerator.device) for a in action_tokens]
        masked_action, labels, _, _ = mask_or_random_replace_action_tokens(
            action_tokens,
            mask_id,
            uni_prompting.pad_id,
            config,
            mask_schedule,
            action_vocab_size,
            is_train=is_train,
            seed=seed,
        )
        masked_list, label_list = [], []
        for i, a in enumerate(action_tokens):
            seq_len = a.size(0)
            masked_list.append(masked_action[i, :seq_len].cpu())
            label_list.append(labels[i, :seq_len].cpu())
        input_ids, attention_masks, label_ids = uni_prompting(
            (
                image_tokens,
                texts,
                state_tokens,
                prev_action_tokens,
                masked_list,
                action_dims,
                label_list,
                accelerator.device,
                config.training.chunk_size,
            ),
            "mm2a",
        )
        return (
            input_ids.to(accelerator.device),
            label_ids.to(accelerator.device),
            attention_masks.to(accelerator.device),
        )

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(
            config.training.get("mask_schedule", "cosine")
        )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    try:
        max_train_steps = config.training.max_train_steps
    except:
        print("max train steps not set, use total step")
        updates_per_epoch = math.ceil(
            len(dataloader) / accelerator.gradient_accumulation_steps
        )
        max_train_steps = config.training.num_epochs * updates_per_epoch
    try:
        warmup_steps = config.lr_scheduler.params.warmup_steps
    except:
        warmup_steps = int(max_train_steps * 0.08)
    print("total training step:", max_train_steps)
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    model.train()
    log_interval = config.training.get("log_interval", 50)
    logger = setup_logger(output_dir / "train.log", accelerator.is_main_process)
    if accelerator.is_main_process:
        logger.info("======== Training setup ========")
        logger.info("output_dir: %s", output_dir)
        logger.info("num_epochs: %s", config.training.num_epochs)
        logger.info(
            "grad_accum_steps (from accelerate): %s",
            accelerator.gradient_accumulation_steps,
        )
        logger.info("================================")
    ##################################
    #   Training_script   #
    #################################
    global_update_step = 0
    for epoch in range(config.training.num_epochs):
        updates_per_epoch = math.ceil(
            len(dataloader) / accelerator.gradient_accumulation_steps
        )
        if accelerator.is_main_process:
            pbar = tqdm(
                total=updates_per_epoch,
                dynamic_ncols=True,
                desc=f"Epoch {epoch}",
                disable=False,
            )
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids, labels, attention_masks = prepare_inputs_and_labels(batch)
                logits, loss = model.forward_process_action(
                    input_ids,
                    labels,
                    attention_masks,
                    max_seq_length=max_action_prompt_len,  # important
                    action_loss_type=config.training.action_loss_type
                    # action_err_token_len=config.training.action_err_token_len,
                    # at_value=config.training.at_value,
                )
                accelerator.backward(loss)
            if accelerator.sync_gradients:
                optimizer.step()
                if lr_scheduler is not None and accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update_step += 1
                loss_avg = accelerator.gather_for_metrics(loss.detach()).mean().item()
                try:
                    current_lr = (
                        lr_scheduler.get_last_lr()[0]
                        if lr_scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    )
                except:
                    current_lr = optimizer.param_groups[0]["lr"]
                accelerator.log(
                    {
                        "train/loss": loss_avg,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                        "train/step_in_epoch": step,
                    },
                    step=global_update_step,
                )
                if accelerator.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"loss={loss_avg:.4f} | lr={current_lr:.6g} | step={global_update_step}/{updates_per_epoch*config.training.num_epochs}"
                    )

                    if (global_update_step % log_interval) == 0:
                        logger.info(
                            "[epoch %s] step %s/%s | epoch_step %s/%s | loss %.4f | lr %.6g",
                            epoch,
                            global_update_step,
                            updates_per_epoch * config.training.num_epochs,
                            global_update_step - updates_per_epoch * epoch,
                            updates_per_epoch,
                            loss_avg,
                            current_lr,
                        )
        if accelerator.is_main_process:
            pbar.close()
            logger.info("Epoch %s finished.", epoch)
            if epoch + 1 < config.training.num_epochs:
                epoch_output_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                accelerator.unwrap_model(model).save_pretrained(epoch_output_dir)
                uni_prompting.text_tokenizer.save_pretrained(epoch_output_dir)

    if accelerator.is_main_process:
        pbar.close()
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        uni_prompting.text_tokenizer.save_pretrained(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()

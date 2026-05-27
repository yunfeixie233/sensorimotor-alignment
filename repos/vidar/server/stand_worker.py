import os
import torch
import hashlib
from base64 import b64encode
from fastapi import FastAPI
from pydantic import BaseModel
import json
import torchvision
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from server.idm import IDM
from server.hook_capture import VidarHookCapture
import io
import base64
from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Request(BaseModel):
    prompt: str
    imgs: list
    num_conditional_frames: int = 1
    num_new_frames: int = 16
    seed: int = 1234
    num_sampling_step: int = 5
    guide_scale: float = 5.0
    password: str = ""
    return_imgs: bool = False
    clean_cache: bool = False
    episode_id: int = -1
    chunk_idx: int = -1
    task_name: str = ""


def sha256(text):
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def init():
    global wan_ti2v
    global ulysses_size
    global cfg
    global processor
    global mask_processor
    global idm
    global hook_capture
    global save_hook_features
    global save_pred_frames
    global save_root

    # SAVE_* env vars gate all disk writes. Default off preserves the
    # original no-side-effects behavior of the server.
    save_hook_features = os.getenv("SAVE_HOOK_FEATURES", "0") == "1"
    save_pred_frames = os.getenv("SAVE_PRED_FRAMES", "0") == "1"
    save_root = Path(os.getenv("SAVE_DIR", "./vidar_capture"))
    logger.info(
        f"Capture config: SAVE_HOOK_FEATURES={save_hook_features} "
        f"SAVE_PRED_FRAMES={save_pred_frames} SAVE_DIR={save_root}"
    )
    hook_capture = None

    # 硬编码配置或从环境变量读取
    cfg = WAN_CONFIGS["ti2v-5B"]
    
    logger.info(f"Current PID: {os.getpid()}")
    # 单卡模式下，Rank相关默认为0/1
    rank = int(os.getenv("RANK", 0))
    pt_dir = os.getenv("MODEL", None)
    idm_path = os.getenv("IDM", None)
    
    # 直接使用 CUDA_VISIBLE_DEVICES 里的第一个设备 (即 cuda:0)
    device = 0
    
    # 初始化图像处理
    processor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((518, 518)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_processor = torchvision.transforms.Resize((736, 640))
    
    # 加载 IDM 模型
    if idm_path and idm_path.endswith("_out.pt"):
        output_dim = int(idm_path.split("_out.pt")[0].split("_")[-1])
        idm = IDM(model_name="mask", output_dim=output_dim).to(device)
    else:
        idm = IDM(model_name="mask", output_dim=14).to(device)
        
    if idm_path and os.path.isfile(idm_path):
        loaded_dict = torch.load(idm_path, map_location=f'cuda:{device}', weights_only=False)
        idm.load_state_dict(loaded_dict["model_state_dict"])
        logger.info(f"IDM loaded from {idm_path}")
    idm.eval()

    # 加载 WanTI2V 模型
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir="./Wan2.2-TI2V-5B",
        pt_dir=pt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )

    if save_hook_features:
        hook_capture = VidarHookCapture()
        hook_capture.register(wan_ti2v.model)
        logger.info(
            "Registered VidarHookCapture on WAN transformer blocks "
            f"{VidarHookCapture.VIDEO_BLOCKS} + head.norm + text_embedding."
        )


def batch_tensor_to_jpeg_message(tensor):
    tensor = (tensor * 255).to(torch.uint8).cpu()
    jpeg_message_list = []
    for i in range(tensor.shape[0]):
        jpeg_tensor = torchvision.io.encode_jpeg(tensor[i])
        jpeg_message_list.append(b64encode(jpeg_tensor.numpy().tobytes()).decode("utf-8"))
    return jpeg_message_list


def idm_pred(request, imgs):
    global processor
    global mask_processor
    global idm
    return_imgs = request.return_imgs
    imgs = imgs.to(next(idm.parameters()).device)
    
    with torch.no_grad():
        actions, masks = idm(processor(imgs), return_mask=return_imgs)
    actions = json.dumps(actions.cpu().numpy().tolist())
    pred = {"actions": actions}
    if return_imgs:
        pred['imgs'] = batch_tensor_to_jpeg_message(imgs)
        masks = mask_processor(masks)
        pred['masks'] = batch_tensor_to_jpeg_message(torch.where(masks >= 0.5, imgs, 1))
    return pred


def _save_capture_artifacts(request, imgs):
    """Disk writes for the current chunk. imgs: [F, 3, H, W] in [0, 1] float.

    Layout:
      {SAVE_DIR}/{task_name}/episode{ep:04d}/
        hook_features_chunk{c:02d}.pt   # dict-of-dicts {name: {step_idx: [D]}}
        pred_frames/chunk{c:02d}/f{i:03d}.png
    """
    task = request.task_name or "unknown"
    ep = max(request.episode_id, 0)
    ci = max(request.chunk_idx, 0)
    ep_dir = save_root / task / f"episode{ep:04d}"

    if save_hook_features and hook_capture is not None:
        feats = hook_capture.features()
        feat_path = ep_dir / f"hook_features_chunk{ci:02d}.pt"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": feats,
                "num_cond_steps": hook_capture.num_cond_steps(),
                "episode_id": ep,
                "chunk_idx": ci,
                "task_name": task,
                "num_sampling_step": request.num_sampling_step,
            },
            feat_path,
        )
        logger.info(
            f"[capture] saved {feat_path.name} "
            f"({hook_capture.num_cond_steps()} cond steps, "
            f"{len(feats)} feature hooks)"
        )

    if save_pred_frames:
        frames_dir = ep_dir / "pred_frames" / f"chunk{ci:02d}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        # imgs shape here is [F, 3, H, W] float in [0,1] from make_grid(normalize=True)
        uint8 = (imgs.clamp(0, 1) * 255).to(torch.uint8).cpu()
        for i in range(uint8.shape[0]):
            png = torchvision.io.encode_png(uint8[i])
            (frames_dir / f"f{i:03d}.png").write_bytes(bytes(png.numpy().tobytes()))
        logger.info(f"[capture] saved {uint8.shape[0]} pred PNGs to {frames_dir}")


def get_pred(request):
    global cfg

    if hook_capture is not None:
        hook_capture.reset()

    frame_num = request.num_conditional_frames + request.num_new_frames
    img = request.imgs[-1]
    img = Image.open(io.BytesIO(base64.b64decode(img)))
    img = img.resize(SIZE_CONFIGS["640*736"])

    # 生成视频/图像
    imgs = wan_ti2v.generate(
        request.prompt,
        img=img,
        size=SIZE_CONFIGS["640*736"],
        max_area=MAX_AREA_CONFIGS["640*736"],
        frame_num=frame_num,
        shift=cfg.sample_shift,
        sample_solver='unipc',
        sampling_steps=request.num_sampling_step,
        guide_scale=request.guide_scale,
        seed=request.seed,
    )
    imgs = imgs[None].clamp(-1, 1)
    imgs = torch.stack([torchvision.utils.make_grid(u, nrow=8, normalize=True, value_range=(-1, 1)) for u in imgs.unbind(2)], dim=1).permute(1, 0, 2, 3) # [B, C, H, W]

    # Capture goes BEFORE empty_cache so the CPU tensors are already detached.
    if (save_hook_features or save_pred_frames) and request.episode_id >= 0:
        try:
            _save_capture_artifacts(request, imgs)
        except Exception as e:
            logger.exception(f"[capture] save failed: {e}")

    torch.cuda.empty_cache()
    pred = idm_pred(request, imgs)
    return pred


api = FastAPI()
wan_ti2v = None
ulysses_size = None
cfg = None
idm = None
processor = None
mask_processor = None
hook_capture = None
save_hook_features = False
save_pred_frames = False
save_root = Path("./vidar_capture")
init()


@api.post("/")
async def predict(request: Request):
    print("Request:", request.prompt, request.num_conditional_frames, request.num_new_frames, request.seed)
    # 简单的鉴权
    if sha256(request.password) == "d43e76d9cad30d53805246aa72cc25b04ce2cbe6c7086b53ac6fb5028c48d307":
        pred = get_pred(request)
        if pred is not None:
            return pred
    else:
        return {}

@api.get("/")
async def test():
    return {"message": "Hello, this is vidar server!"}

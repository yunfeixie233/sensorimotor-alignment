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
    cfg = WAN_CONFIGS["ti2v-5B"]
    logger.info(f"Current PID: {os.getpid()}")
    rank = int(os.getenv("RANK", 0))
    pt_dir = os.getenv("MODEL", None)
    idm_path = os.getenv("IDM", None)
    device = 0
    processor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((518, 518)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_processor = torchvision.transforms.Resize((736, 640))
    if idm_path.endswith("_out.pt"):
        output_dim = int(idm_path.split("_out.pt")[0].split("_")[-1])
        idm = IDM(model_name="mask", output_dim=output_dim).to(device)
    else:
        idm = IDM(model_name="mask", output_dim=14).to(device)
    if os.path.isfile(idm_path):
        loaded_dict = torch.load(idm_path, weights_only=False)
        idm.load_state_dict(loaded_dict["model_state_dict"])
        logger.info(f"IDM loaded from {idm_path}")
    idm.eval()

    wan_ti2v = wan.WanTI2VCausalServer(
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


def get_pred(request):
    global cfg
    
    frame_num = request.num_conditional_frames + request.num_new_frames

    # torch.cuda.memory._record_memory_history(max_entries=100000)
    print(f'{request.prompt=}', f'{len(request.imgs)=}', f'{request.num_conditional_frames=}', f'{request.num_new_frames=}', f'{request.seed=}')
    imgs = wan_ti2v.generate(
        request.prompt,
        img=request.imgs,
        size=SIZE_CONFIGS["640*736"],
        max_area=MAX_AREA_CONFIGS["640*736"],
        frame_num=frame_num,
        num_conditional_frames=request.num_conditional_frames,
        clean_cache=request.clean_cache,
        shift=cfg.sample_shift,
        sample_solver='unipc',
        sampling_steps=request.num_sampling_step,
        guide_scale=request.guide_scale,
        seed=request.seed,
        offload_model=False
    )
    imgs = imgs[None].clamp(-1, 1)
    imgs = torch.stack([torchvision.utils.make_grid(u, nrow=8, normalize=True, value_range=(-1, 1)) for u in imgs.unbind(2)], dim=1).permute(1, 0, 2, 3) # [B, C, H, W]
    pred = idm_pred(request, imgs)
    return pred


api = FastAPI()
wan_ti2v = None
ulysses_size = None
cfg = None
idm = None
processor = None
mask_processor = None
init()


@api.post("/")
async def predict(request: Request):
    print("Request:", request.prompt, request.num_conditional_frames, request.num_new_frames, request.seed)
    if sha256(request.password) == "d43e76d9cad30d53805246aa72cc25b04ce2cbe6c7086b53ac6fb5028c48d307":
        pred = get_pred(request)
        if pred is not None:
            return pred
    else:
        return {}

@api.get("/")
async def test():
    return {"message": "Hello, this is vidarc server"}

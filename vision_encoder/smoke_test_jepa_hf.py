"""
Stage-0 smoke test — HF JEPA encoders (I-JEPA + V-JEPA 2 L/H/g).
Verifies each model loads, runs a forward pass, and reports the
last_hidden_state shape (seq length tells us CLS / no-CLS + token count).

Run:
  /lambda/nfs/vla/conda/envs/starVLA/bin/python \
      /lambda/nfs/vla/cknna_project/vision_encoder/smoke_test_jepa_hf.py
"""
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

torch.backends.cudnn.enabled = False
DEVICE = "cuda"
results = {}


def banner(s):
    print(f"\n{'=' * 64}\n{s}\n{'=' * 64}", flush=True)


def test_ijepa():
    from transformers import AutoImageProcessor, IJepaModel
    hf = "facebook/ijepa_vith14_1k"
    banner(f"I-JEPA  {hf}")
    t0 = time.time()
    model = IJepaModel.from_pretrained(hf, torch_dtype=torch.float16).to(DEVICE).eval()
    proc = AutoImageProcessor.from_pretrained(hf, use_fast=True)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    # DROID-shaped dummy image (320x180) -> processor handles resize/crop
    img = Image.fromarray(np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8))
    px = proc(images=img, return_tensors="pt")["pixel_values"].to(DEVICE).half()
    with torch.no_grad():
        h = model(pixel_values=px).last_hidden_state
    print(f"pixel_values {tuple(px.shape)} -> last_hidden_state {tuple(h.shape)}", flush=True)
    print(f"  seq={h.shape[1]} dim={h.shape[2]}  (256 patches => no CLS)", flush=True)
    results["ijepa_vith14_1k"] = ("PASS", tuple(h.shape))
    del model
    torch.cuda.empty_cache()


def test_vjepa2(hf):
    from transformers import VJEPA2Model
    banner(f"V-JEPA 2  {hf}")
    t0 = time.time()
    model = VJEPA2Model.from_pretrained(hf, torch_dtype=torch.float16).to(DEVICE).eval()
    print(f"loaded in {time.time() - t0:.0f}s  "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)
    # 16-frame clip, (B, T, C, H, W), 256px
    vid = torch.randn(1, 16, 3, 256, 256, dtype=torch.float16, device=DEVICE)
    with torch.no_grad():
        out = model(pixel_values_videos=vid)
    h = out.last_hidden_state
    print(f"pixel_values_videos {tuple(vid.shape)} -> last_hidden_state {tuple(h.shape)}", flush=True)
    print(f"  seq={h.shape[1]} dim={h.shape[2]}  (expect 8 temporal x 256 spatial = 2048)", flush=True)
    results[hf.split('/')[-1]] = ("PASS", tuple(h.shape))
    del model
    torch.cuda.empty_cache()


def test_vjepa2_processor():
    """Probe the V-JEPA 2 video processor so Stage 1 knows the exact input format."""
    banner("V-JEPA 2 video processor probe")
    from transformers import AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    print(f"processor class: {type(proc).__name__}", flush=True)
    frames = [np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(16)]
    try:
        out = proc(videos=[frames], return_tensors="pt")
    except Exception:
        out = proc([frames], return_tensors="pt")
    for k, v in out.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else v
        print(f"  out[{k!r}] = {shape}", flush=True)
    results["vjepa2_processor"] = ("PASS", "ok")


def main():
    print(f"torch {torch.__version__}  GPU {torch.cuda.get_device_name(0)}", flush=True)
    import transformers
    print(f"transformers {transformers.__version__}", flush=True)

    tests = [
        ("ijepa_vith14_1k", test_ijepa),
        ("vjepa2-vitl-fpc64-256", lambda: test_vjepa2("facebook/vjepa2-vitl-fpc64-256")),
        ("vjepa2-vith-fpc64-256", lambda: test_vjepa2("facebook/vjepa2-vith-fpc64-256")),
        ("vjepa2-vitg-fpc64-256", lambda: test_vjepa2("facebook/vjepa2-vitg-fpc64-256")),
        ("vjepa2_processor", test_vjepa2_processor),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}", flush=True)
            traceback.print_exc()
            results[name] = ("FAIL", str(e))

    banner("SUMMARY")
    n_pass = 0
    for name, (status, info) in results.items():
        print(f"  {status:5s}  {name:28s}  {info}", flush=True)
        n_pass += status == "PASS"
    print(f"\n{n_pass}/{len(results)} passed", flush=True)
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()

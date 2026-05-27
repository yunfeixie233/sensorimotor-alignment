"""Audit the JEPA §F extension for extraction / alignment bugs.
CHECK 1: feature collapse — mean off-diagonal cosine of each JEPA feature set
         vs known-good reference encoders.
CHECK 2: multiframe anchor alignment — frame 064.png of each multiframe sample
         must equal the Setting-1 anchor image at index_map[new].
CHECK 4: V-JEPA 2 token order — perturb only the last clip frame; if tokens are
         temporal-major the changed embedding tokens fall in the last 1/8 block.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

D1 = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid"
MF = "/lambda/nfs/vla/cache/cknna_data_store/cknna_data_droid_multiframe"


def banner(s):
    print(f"\n{'=' * 64}\n{s}\n{'=' * 64}", flush=True)


# ---- CHECK 1: feature collapse ------------------------------------------------
banner("CHECK 1: feature collapse (mean off-diagonal cosine, 2000-sample subset)")


def cos_stats(path, n=2000):
    f = torch.load(path, weights_only=True).float()
    g = F.normalize(f[:n], dim=1)
    sim = g @ g.T
    od = sim[~torch.eye(len(g), dtype=bool)]
    return tuple(f.shape), od.mean().item(), od.std().item(), od.amax().item()


jepa_s1 = ["ijepa-vith14", "vjepa2-vitl", "vjepa2-vith", "vjepa2-vitg",
           "vjepa-v1-vitl", "vjepa-v1-vith", "vjepa2-1-vitl", "vjepa2-1-vitg"]
refs = ["dinov2-base", "mae-base", "clip-vit-l14", "siglip2-large"]
for k in jepa_s1:
    sh, m, s, mx = cos_stats(f"{D1}/{k}-standalone/feats_A_vit.pt")
    print(f"  JEPA-S1  {k:16s} {str(sh):14s} cos mean={m:+.3f} std={s:.3f} max={mx:.3f}", flush=True)
print()
for k in refs:
    sh, m, s, mx = cos_stats(f"{D1}/{k}-standalone/feats_A_vit.pt")
    print(f"  ref      {k:16s} {str(sh):14s} cos mean={m:+.3f} std={s:.3f} max={mx:.3f}", flush=True)
print()
for k in ["vjepa2-vitg-realclip", "vjepa-v1-vith-realclip", "vjepa2-1-vitg-realclip"]:
    sh, m, s, mx = cos_stats(f"{MF}/{k}/feats_A_vit.pt")
    print(f"  JEPA-S2  {k:24s} {str(sh):14s} cos mean={m:+.3f} std={s:.3f} max={mx:.3f}", flush=True)

# ---- CHECK 2: multiframe anchor alignment ------------------------------------
banner("CHECK 2: multiframe frame 064.png == Setting-1 anchor at index_map[new]")
imap = json.load(open(f"{MF}/index_map.json"))["mapping"]
all_ok = True
for new in [0, 1, 1000, 2444, 4889]:
    orig = imap[str(new)]
    a = np.array(Image.open(f"{MF}/images/{new:06d}/064.png").convert("RGB"))
    b = np.array(Image.open(f"{D1}/images/{orig:06d}.png").convert("RGB"))
    same = a.shape == b.shape and np.array_equal(a, b)
    all_ok &= same
    print(f"  new={new:5d} -> orig={orig:5d}: frame064==anchor? {same}", flush=True)
# also: first multiframe frame should be orig anchor minus 64 (sanity on past-clip)
print(f"  ALL anchor frames match: {all_ok}", flush=True)

# ---- CHECK 4: V-JEPA 2 token order (temporal-major?) -------------------------
banner("CHECK 4: V-JEPA 2 token order — does lasttube == last temporal block?")
try:
    from transformers import VJEPA2Model
    m = VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256").cuda().eval()
    torch.manual_seed(0)
    base = torch.rand(1, 16, 3, 256, 256, device="cuda")
    pert = base.clone()
    pert[:, 15] = torch.rand(1, 3, 256, 256, device="cuda")  # perturb ONLY last frame
    with torch.no_grad():
        h_base = m(pixel_values_videos=base, output_hidden_states=True).hidden_states[0]
        h_pert = m(pixel_values_videos=pert, output_hidden_states=True).hidden_states[0]
    diff = (h_base - h_pert).abs().sum(-1)[0]          # (seq,)
    changed = (diff > 1e-4).nonzero().flatten()
    seq = diff.shape[0]
    block = seq // 8                                    # 8 temporal groups (16 frames / tubelet 2)
    print(f"  seq={seq}  changed tokens: {len(changed)}  "
          f"range=[{changed.min().item()},{changed.max().item()}]", flush=True)
    print(f"  last temporal block = indices [{seq - block},{seq - 1}] (what pool='lasttube' takes)", flush=True)
    in_last = ((changed >= seq - block).float().mean().item())
    print(f"  fraction of changed tokens inside the last block: {in_last:.2f}", flush=True)
    print(f"  -> temporal-major (lasttube correct): {changed.min().item() >= seq - block}", flush=True)
except Exception as e:
    import traceback
    print(f"  CHECK 4 failed: {e}")
    traceback.print_exc()

print("\nDONE", flush=True)

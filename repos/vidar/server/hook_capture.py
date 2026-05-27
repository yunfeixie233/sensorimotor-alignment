"""Server-side hook capture for VIDAR WAN2.2-TI2V-5B.

Per-cond-step accumulation: `self._buf[name]` is a dict keyed by denoising
step index, so the full schedule is available after `generate()` returns.
Distinct from the older `extractors/extract_va_features_vidar_droid.py:161`
class which overwrote `self._buf[name]` on every cond pass (last-step only).

CFG parity: even `_call_idx` values are cond passes, odd are uncond. The
parity counter is driven by `wan_model.head` forward hook (fires once per
transformer forward, which is called twice per sampling iteration).
"""
import torch


class VidarHookCapture:
    VIDEO_BLOCKS = [0, 7, 14, 21, 29]

    def __init__(self):
        self._buf = {}          # {name: {step_idx: Tensor}}
        self._call_idx = 0      # total forwards (cond + uncond)
        self._handles = []

    def reset(self):
        self._buf = {}
        self._call_idx = 0

    def _feature_hook(self, name, kind="out"):
        def hook(module, inp, out=None):
            if self._call_idx % 2 != 0:
                return
            step_idx = self._call_idx // 2
            if kind == "pre":
                t = inp[0] if isinstance(inp, tuple) else inp
            else:
                t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            # Mean-pool on GPU in native dtype BEFORE dtype promotion or host
            # transfer. WAN2.2-TI2V-5B full-resolution tokens can be L ≈ 29k,
            # so a naive `.float().cpu()` briefly allocates ~360 MB fp32 per
            # hook call on GPU and OOMs at IDM time. Mean-pool to [D] first.
            with torch.no_grad():
                if t.dim() >= 3:
                    pooled = t[0].mean(dim=0)
                elif t.dim() == 2:
                    pooled = t[0]
                else:
                    pooled = t
                self._buf.setdefault(name, {})[step_idx] = (
                    pooled.detach().float().cpu()
                )

        return hook

    def _counter_hook(self):
        def hook(module, inp, out):
            self._call_idx += 1

        return hook

    def register(self, wan_model):
        self._handles.append(
            wan_model.blocks[0].register_forward_pre_hook(
                self._feature_hook("V-D1", kind="pre")
            )
        )
        for bi in self.VIDEO_BLOCKS:
            self._handles.append(
                wan_model.blocks[bi].register_forward_hook(
                    self._feature_hook(f"V-B{bi}")
                )
            )
        self._handles.append(
            wan_model.head.norm.register_forward_hook(self._feature_hook("V-Norm"))
        )
        self._handles.append(
            wan_model.text_embedding.register_forward_hook(self._feature_hook("T-D1"))
        )
        self._handles.append(
            wan_model.head.register_forward_hook(self._counter_hook())
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def features(self):
        """All per-step features mean-pooled. {name: {step_idx: Tensor[D]}}."""
        out = {}
        for name, step_dict in self._buf.items():
            out[name] = {}
            for step_idx, t in step_dict.items():
                out[name][step_idx] = t[0].mean(dim=0) if t.dim() >= 2 else t
        return out

    def features_last_step(self):
        """Last cond-step only. {name: Tensor[D]}. Matches the old API."""
        out = {}
        for name, step_dict in self._buf.items():
            last_step = max(step_dict.keys())
            t = step_dict[last_step]
            out[name] = t[0].mean(dim=0) if t.dim() >= 2 else t
        return out

    def num_cond_steps(self):
        if not self._buf:
            return 0
        return max(len(sd) for sd in self._buf.values())

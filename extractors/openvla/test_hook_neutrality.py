"""
test_hook_neutrality.py

Unit test: verify that registering the hidden-state hooks on model.language_model
does NOT change the output of the fine-tuned OpenVLA forward pass.

The ft model's custom forward() hardcodes output_hidden_states=False on its
internal self.language_model() call (line 1339 of modeling_prismatic.py).
Our pre-hook overrides that to True. This test checks that the final return
value -- the action logits tensor -- is bit-for-bit identical with and
without the hooks.

Usage:
    cd /home/ubuntu/verl/SimplerEnv-OpenVLA
    conda activate openvla_env
    python cknna/test_hook_neutrality.py
"""

import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "openvla-bridge-sft")
DATA_DIR = "/home/ubuntu/verl/starVLA/cknna/cknna_data_50k"
N_SAMPLES = 4        # small -- we only need to check a few samples
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 0.0           # we expect bit-for-bit equality (same dtype, same path)


def load_model():
    print(f"Loading model from {CKPT} ...")
    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(DEVICE)
    print(f"  Loaded. dtype={next(model.parameters()).dtype}  device={DEVICE}")
    return model, processor


def build_inputs(processor, n_samples):
    import json
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    images_dir = os.path.join(DATA_DIR, "images")
    samples = []
    for i in range(n_samples):
        img = Image.open(os.path.join(images_dir, f"{i:06d}.png")).convert("RGB")
        task = meta["task_descriptions"][i] or "pick the object"
        prompt = f"In: What action should the robot take to {task}?\nOut:"
        inp = processor(prompt, img)
        samples.append({
            "input_ids":      inp["input_ids"].to(DEVICE),
            "attention_mask": inp["attention_mask"].to(DEVICE),
            "pixel_values":   inp["pixel_values"].to(dtype=torch.bfloat16, device=DEVICE),
        })
    return samples


def forward_no_hooks(model, inputs):
    """Run model.forward() with no hooks. Returns list of output tensors."""
    outputs = []
    with torch.inference_mode():
        for inp in inputs:
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                pixel_values=inp["pixel_values"],
            )
            # ft model returns a plain Tensor (action logits)
            outputs.append(out.detach().cpu())
    return outputs


def forward_with_hooks(model, inputs):
    """Run model.forward() with hidden-state hooks active. Returns (outputs, hidden_states)."""
    _hook_output = {}

    def _lm_pre_hook(_module, _args, kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return _args, kwargs

    def _lm_post_hook(_module, _input, output):
        _hook_output["hidden_states"] = output.hidden_states

    h1 = model.language_model.register_forward_pre_hook(_lm_pre_hook, with_kwargs=True)
    h2 = model.language_model.register_forward_hook(_lm_post_hook)

    outputs = []
    captured_hidden = []
    with torch.inference_mode():
        for inp in inputs:
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                pixel_values=inp["pixel_values"],
            )
            outputs.append(out.detach().cpu())
            # Record shape of last-layer hidden state (confirm it was captured)
            captured_hidden.append(_hook_output["hidden_states"][-1].shape)

    h1.remove()
    h2.remove()
    return outputs, captured_hidden


def run_tests():
    print("=" * 60)
    print("Hook Neutrality Test for Fine-tuned OpenVLA")
    print("=" * 60)

    model, processor = load_model()
    inputs = build_inputs(processor, N_SAMPLES)

    print(f"\nRunning forward WITHOUT hooks  (N={N_SAMPLES}) ...")
    t0 = time.perf_counter()
    out_no_hook = forward_no_hooks(model, inputs)
    dt_no_hook = time.perf_counter() - t0
    print(f"  Done in {dt_no_hook:.2f}s")

    print(f"\nRunning forward WITH hooks     (N={N_SAMPLES}) ...")
    t0 = time.perf_counter()
    out_with_hook, captured_shapes = forward_with_hooks(model, inputs)
    dt_with_hook = time.perf_counter() - t0
    print(f"  Done in {dt_with_hook:.2f}s")

    # -----------------------------------------------------------------------
    # Test 1: output logits are bit-for-bit identical
    # -----------------------------------------------------------------------
    print("\n--- Test 1: action logits identical (no-hook vs with-hook) ---")
    all_pass = True
    for i, (a, b) in enumerate(zip(out_no_hook, out_with_hook)):
        equal = torch.equal(a, b)
        close = torch.allclose(a.float(), b.float(), atol=ATOL)
        max_diff = (a.float() - b.float()).abs().max().item()
        status = "PASS" if equal else "FAIL"
        print(f"  sample {i}: shape={tuple(a.shape)}  equal={equal}  max_diff={max_diff:.2e}  [{status}]")
        if not equal:
            all_pass = False

    if all_pass:
        print("  => Test 1 PASSED: hooks do not alter the action logits.")
    else:
        print("  => Test 1 FAILED: hooks changed the output!")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Test 2: hooks actually captured hidden states
    # -----------------------------------------------------------------------
    print("\n--- Test 2: hook captured hidden_states from language model ---")
    for i, shape in enumerate(captured_shapes):
        print(f"  sample {i}: last-layer hidden_states shape = {tuple(shape)}")
        assert len(shape) == 3, f"Expected 3D tensor, got shape {shape}"
        assert shape[-1] > 0, "hidden_size must be > 0"
    print("  => Test 2 PASSED: hidden states captured correctly.")

    # -----------------------------------------------------------------------
    # Test 3: hooks are removed after the call (no side effects)
    # -----------------------------------------------------------------------
    print("\n--- Test 3: hooks removed -- subsequent forward is clean ---")
    out_post = forward_no_hooks(model, inputs[:1])
    equal = torch.equal(out_post[0], out_no_hook[0])
    print(f"  post-hook forward matches original: {equal}")
    assert equal, "Forward changed after hook removal -- hooks left state behind!"
    print("  => Test 3 PASSED: hooks leave no side effects after removal.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    overhead_pct = (dt_with_hook - dt_no_hook) / dt_no_hook * 100
    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED")
    print(f"  Time without hooks: {dt_no_hook:.2f}s")
    print(f"  Time with hooks:    {dt_with_hook:.2f}s")
    print(f"  Overhead:           {overhead_pct:+.1f}%  (extra cost of output_hidden_states=True)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_tests()

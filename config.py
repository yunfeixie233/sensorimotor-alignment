"""
Central path configuration for the CKNNA project.

Loads paths from paths.env (shell-compatible key=value file) with
fallback to environment variables, then to hardcoded defaults.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_THIS_DIR, "paths.env")


def _load_env_file(path):
    """Parse a shell-style KEY=VALUE file, resolving ${VAR} references."""
    vals = {}
    if not os.path.isfile(path):
        return vals
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            for ref_key, ref_val in vals.items():
                value = value.replace("${%s}" % ref_key, ref_val)
            vals[key] = value
    return vals


_FILE_VALS = _load_env_file(_ENV_FILE)


def get(key, default=None):
    """Look up a path: env var > paths.env > default."""
    return os.environ.get(key, _FILE_VALS.get(key, default))


PROJECT_ROOT = _THIS_DIR
REPOS_DIR = os.path.join(_THIS_DIR, "repos")
WORK = get("WORK", "/home/ubuntu/verl")
CONDA_ROOT = get("CONDA_ROOT", os.path.join(WORK, "conda"))

BRIDGE_RLDS_DIR = get("BRIDGE_RLDS_DIR",
                       os.path.join(WORK, "openvla/datasets/bridge_orig/1.0.0"))
DROID_RLDS_DIR = get("DROID_RLDS_DIR",
                      os.path.join(WORK, "droid/data/droid/1.0.1"))

STARVLA_ROOT = get("STARVLA_ROOT", os.path.join(REPOS_DIR, "starVLA"))
MOTUS_ROOT = get("MOTUS_ROOT", os.path.join(REPOS_DIR, "Motus"))
XVLA_ROOT = get("XVLA_ROOT", os.path.join(REPOS_DIR, "X-VLA"))
LINGBOT_VA_ROOT = get("LINGBOT_VA_ROOT", os.path.join(REPOS_DIR, "lingbot-va"))
SOFTDTW_DIR = get("SOFTDTW_DIR", os.path.join(REPOS_DIR, "pytorch-softdtw-cuda"))
INTERNVLA_ROOT = get("INTERNVLA_ROOT", os.path.join(REPOS_DIR, "InternVLA-A1"))
ISAAC_GROOT_DIR = get("ISAAC_GROOT_DIR", os.path.join(REPOS_DIR, "Isaac-GR00T-N15"))
ISAAC_GROOT_N16_DIR = get("ISAAC_GROOT_N16_DIR",
                           os.path.join(REPOS_DIR, "Isaac-GR00T-N16"))
COGACT_ROOT = get("COGACT_ROOT", os.path.join(REPOS_DIR, "CogACT"))
OCTO_ROOT = get("OCTO_ROOT", os.path.join(REPOS_DIR, "octo"))

SIMPLER_ROOT = get("SIMPLER_ROOT", os.path.join(WORK, "SimplerEnv-OpenVLA"))

DATA_STORE = get("DATA_STORE", os.path.join(WORK, "starVLA/cknna"))
RECORD_DIR = get("RECORD_DIR", os.path.join(DATA_STORE, "record"))
META_CSV = get("META_CSV",
                os.path.join(PROJECT_ROOT, "data/performance/model_metadata.csv"))

CKPT_OPENVLA_7B = get("CKPT_OPENVLA_7B",
                        os.path.join(SIMPLER_ROOT, "checkpoints/openvla-7b"))
CKPT_OPENVLA_FT = get("CKPT_OPENVLA_FT",
                        os.path.join(SIMPLER_ROOT, "checkpoints/openvla-bridge-sft"))
CKPT_RT1X = get("CKPT_RT1X",
                  os.path.join(SIMPLER_ROOT,
                               "rt_1_x_tf_trained_for_002272480_step"))
CKPT_GROOT_N15 = get("CKPT_GROOT_N15",
                       os.path.join(WORK, "GR00T-N1.5-Lerobot-SimplerEnv-BridgeV2"))
CKPT_GROOT_N16 = get("CKPT_GROOT_N16",
                       os.path.join(WORK, "GR00T-N1.6-bridge"))
CKPT_PRISMATIC_RAW = get("CKPT_PRISMATIC_RAW",
                           os.path.join(WORK,
                                        "prismatic-vlms-hf/prism-dinosiglip-224px+7b"))
CKPT_STARVLA_DIR = get("CKPT_STARVLA_DIR",
                         os.path.join(WORK, "starVLA/playground/Pretrained_models"))

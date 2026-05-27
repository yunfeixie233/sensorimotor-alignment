"""
Compatibility shim: map old lerobot.common.* import paths to current lerobot.*.

The HaomingSong/lerobot-pi0-bridge checkpoint was built with an older lerobot
that used lerobot.common.{constants,optim,policies,utils}. The current lerobot
moved everything up one level (lerobot.{utils,optim,policies}).

Call install_shim() BEFORE importing the checkpoint's modeling_pi0.py.
"""

import importlib
import sys
import types


def install_shim():
    if "lerobot.common" in sys.modules:
        return

    import lerobot

    # -- lerobot.common (package) --
    common = types.ModuleType("lerobot.common")
    common.__path__ = []
    common.__package__ = "lerobot.common"
    sys.modules["lerobot.common"] = common
    lerobot.common = common

    # -- lerobot.common.constants --
    real_constants = importlib.import_module("lerobot.utils.constants")
    compat_constants = types.ModuleType("lerobot.common.constants")
    for attr in dir(real_constants):
        if not attr.startswith("_"):
            setattr(compat_constants, attr, getattr(real_constants, attr))
    # Old name OBS_ROBOT = "observation.state" (renamed to OBS_STATE in new lerobot)
    if not hasattr(compat_constants, "OBS_ROBOT"):
        compat_constants.OBS_ROBOT = getattr(real_constants, "OBS_STATE", "observation.state")
    sys.modules["lerobot.common.constants"] = compat_constants
    common.constants = compat_constants

    # -- lerobot.common.optim (package) --
    common_optim = types.ModuleType("lerobot.common.optim")
    common_optim.__path__ = []
    common_optim.__package__ = "lerobot.common.optim"
    sys.modules["lerobot.common.optim"] = common_optim
    common.optim = common_optim

    # -- lerobot.common.optim.optimizers --
    real_optimizers = importlib.import_module("lerobot.optim.optimizers")
    sys.modules["lerobot.common.optim.optimizers"] = real_optimizers
    common_optim.optimizers = real_optimizers

    # -- lerobot.common.optim.schedulers --
    real_schedulers = importlib.import_module("lerobot.optim.schedulers")
    sys.modules["lerobot.common.optim.schedulers"] = real_schedulers
    common_optim.schedulers = real_schedulers

    # -- lerobot.common.policies (package) --
    common_policies = types.ModuleType("lerobot.common.policies")
    common_policies.__path__ = []
    common_policies.__package__ = "lerobot.common.policies"
    sys.modules["lerobot.common.policies"] = common_policies
    common.policies = common_policies

    # -- lerobot.common.policies.pretrained --
    real_pretrained = importlib.import_module("lerobot.policies.pretrained")
    sys.modules["lerobot.common.policies.pretrained"] = real_pretrained
    common_policies.pretrained = real_pretrained

    # -- lerobot.common.policies.normalize --
    # Normalize/Unnormalize were removed from lerobot. Load from local compat file.
    import os
    compat_normalize_path = os.path.join(os.path.dirname(__file__), "lerobot_compat_normalize.py")
    spec = importlib.util.spec_from_file_location("lerobot.common.policies.normalize", compat_normalize_path)
    compat_normalize = importlib.util.module_from_spec(spec)
    sys.modules["lerobot.common.policies.normalize"] = compat_normalize
    common_policies.normalize = compat_normalize
    spec.loader.exec_module(compat_normalize)

    # -- lerobot.common.utils (package) --
    common_utils = types.ModuleType("lerobot.common.utils")
    common_utils.__path__ = []
    common_utils.__package__ = "lerobot.common.utils"
    sys.modules["lerobot.common.utils"] = common_utils
    common.utils = common_utils

    # -- lerobot.common.utils.utils --
    real_utils = importlib.import_module("lerobot.utils.utils")
    sys.modules["lerobot.common.utils.utils"] = real_utils
    common_utils.utils = real_utils

    # -- lerobot.common.policies.pi0 (package, needed by paligemma_with_expert) --
    common_policies_pi0 = types.ModuleType("lerobot.common.policies.pi0")
    common_policies_pi0.__path__ = []
    common_policies_pi0.__package__ = "lerobot.common.policies.pi0"
    sys.modules["lerobot.common.policies.pi0"] = common_policies_pi0
    common_policies.pi0 = common_policies_pi0

    # -- lerobot.common.policies.pi0.flex_attention --
    compat_dir = os.path.dirname(__file__)
    flex_path = os.path.join(compat_dir, "flex_attention.py")
    if os.path.exists(flex_path):
        spec = importlib.util.spec_from_file_location(
            "lerobot.common.policies.pi0.flex_attention", flex_path)
        flex_mod = importlib.util.module_from_spec(spec)
        sys.modules["lerobot.common.policies.pi0.flex_attention"] = flex_mod
        common_policies_pi0.flex_attention = flex_mod
        spec.loader.exec_module(flex_mod)

    # -- paligemma_with_expert (standalone module, old lerobot bundled it alongside pi0) --
    if "paligemma_with_expert" not in sys.modules:
        pwe_path = os.path.join(compat_dir, "paligemma_with_expert.py")
        if os.path.exists(pwe_path):
            spec = importlib.util.spec_from_file_location("paligemma_with_expert", pwe_path)
            pwe = importlib.util.module_from_spec(spec)
            sys.modules["paligemma_with_expert"] = pwe
            spec.loader.exec_module(pwe)

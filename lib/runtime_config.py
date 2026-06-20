"""Shared runtime controls for the PA-SfM pipeline.

Import this module before importing torch so process-level CUDA/Python
environment switches are fixed before CUDA libraries are initialized.
"""
import os
import random

DEFAULT_SEED = int(os.environ.get("REPRO_SEED", "1013"))

os.environ.setdefault("REPRO_SEED", str(DEFAULT_SEED))
os.environ.setdefault("PYTHONHASHSEED", str(DEFAULT_SEED))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def reproducible_run_tag(seed=None):
    tag = os.environ.get("REPRO_RUN_ID") or os.environ.get("RUN_ID") or "stable"
    tag_l = tag.lower()
    if "seed" in tag_l or "1013" in tag_l:
        return "stable"
    return tag


def seed_everything(seed=None):
    seed = DEFAULT_SEED if seed is None else int(seed)

    random.seed(seed)

    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return seed

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    deterministic = _env_flag("REPRO_DETERMINISTIC", True)
    warn_only = _env_flag("REPRO_DETERMINISTIC_WARN_ONLY", True)
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    allow_tf32 = _env_flag("REPRO_ALLOW_TF32", True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32

    return seed


def torch_generator(device=None, seed=None):
    seed = DEFAULT_SEED if seed is None else int(seed)

    import torch

    if device is None:
        generator = torch.Generator()
    else:
        generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator

"""Reproducibility utilities: seeding + run metadata capture.

Import pattern (scripts): add project `code/` dir to sys.path, then
    from utils.seeds import set_seed, run_metadata
"""
from __future__ import annotations
import os
import sys
import json
import time
import random
import hashlib
import platform
import subprocess

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed python/numpy/torch and (best-effort) enable deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass
    return seed


def _git(args):
    try:
        return (
            subprocess.check_output(["git"] + args, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def run_metadata(config: dict | None = None) -> dict:
    """Capture reproducibility metadata for a run (host, versions, git, config hash)."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": _git(["rev-parse", "HEAD"]),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        meta["n_gpu"] = torch.cuda.device_count()
    except Exception:
        pass
    try:
        import rdkit

        meta["rdkit"] = rdkit.__version__
    except Exception:
        pass
    if config is not None:
        blob = json.dumps(config, sort_keys=True, default=str).encode()
        meta["config_hash"] = hashlib.md5(blob).hexdigest()[:12]
    return meta


if __name__ == "__main__":
    set_seed(42)
    print(json.dumps(run_metadata(), indent=2))

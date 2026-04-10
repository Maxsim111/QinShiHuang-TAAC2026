from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def resolve_repo_root(current_file: Path) -> Path:
    return current_file.resolve().parents[3]


def resolve_workspace_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def set_runtime_environment(config: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device"]["cuda_visible_devices"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: dict[str, Any]) -> "torch.device":
    import torch

    required_keyword = str(config["device"].get("required_device_keyword", "")).strip()
    if required_keyword:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable but the config requires a specific GPU.")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"Expected exactly one visible GPU after masking, found {torch.cuda.device_count()}.")
        device_name = torch.cuda.get_device_name(0)
        if required_keyword not in device_name:
            raise RuntimeError(f"Visible GPU is '{device_name}', expected it to contain '{required_keyword}'.")
        print(f"Using device: cuda:0 ({device_name})")
        return torch.device("cuda:0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def enable_fast_torch_runtime() -> None:
    import torch

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

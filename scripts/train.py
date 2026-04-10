from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeepFM experiment from a tracked config.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config under configs/.")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare cached features.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild cached features before training.")
    parser.add_argument("--raw-data", type=str, default=None, help="Optional override for paths.raw_data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from deepfm.common.config import load_config
    from deepfm.common.runtime import enable_fast_torch_runtime, resolve_device, seed_everything, set_runtime_environment
    from deepfm.train import criteo as criteo_train
    from deepfm.train import taac2026 as taac_train

    config = load_config((repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    if args.raw_data is not None:
        config["paths"]["raw_data"] = args.raw_data

    print(f"Rerun: python scripts/train.py --config {args.config}")
    set_runtime_environment(config)
    seed_everything(int(config["training"]["seed"]))
    enable_fast_torch_runtime()
    device = resolve_device(config)

    task = config["task"]
    if task == "taac2026":
        taac_train.run_training(config, repo_root, device, prepare_only=args.prepare_only, force_prepare=args.force_prepare)
    elif task == "criteo":
        criteo_train.run_training(config, repo_root, device, prepare_only=args.prepare_only, force_prepare=args.force_prepare)
    else:
        raise ValueError(f"Unsupported task={task!r}")


if __name__ == "__main__":
    main()

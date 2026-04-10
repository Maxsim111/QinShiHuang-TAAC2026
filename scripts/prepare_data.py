from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached features for a DeepFM experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config under configs/.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of prepared artifacts.")
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

    print(f"Rerun: python scripts/prepare_data.py --config {args.config}")
    set_runtime_environment(config)
    seed_everything(int(config["training"]["seed"]))
    enable_fast_torch_runtime()
    device = resolve_device(config)

    if config["task"] == "taac2026":
        taac_train.run_training(config, repo_root, device, prepare_only=True, force_prepare=args.force)
    elif config["task"] == "criteo":
        criteo_train.run_training(config, repo_root, device, prepare_only=True, force_prepare=args.force)
    else:
        raise ValueError(f"Unsupported task={config['task']!r}")


if __name__ == "__main__":
    main()

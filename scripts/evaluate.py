from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepFM checkpoint on the validation split.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config under configs/.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--split", type=str, default="valid", help="Dataset split to evaluate. Default: valid.")
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
    checkpoint_path = None if args.checkpoint is None else Path(args.checkpoint).resolve()

    print(f"Rerun: python scripts/evaluate.py --config {args.config}")
    set_runtime_environment(config)
    seed_everything(int(config["training"]["seed"]))
    enable_fast_torch_runtime()
    device = resolve_device(config)

    if config["task"] == "taac2026":
        taac_train.run_evaluation(config, repo_root, device, checkpoint_path=checkpoint_path, split=args.split)
    elif config["task"] == "criteo":
        criteo_train.run_evaluation(config, repo_root, device, checkpoint_path=checkpoint_path, split=args.split)
    else:
        raise ValueError(f"Unsupported task={config['task']!r}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from deepfm.common.config import load_config
    from deepfm.common.runtime import enable_fast_torch_runtime, resolve_device, seed_everything, set_runtime_environment
    from deepfm.train import criteo as criteo_train

    config_path = repo_root / "configs" / "debug" / "criteo_smoke.yaml"
    config = load_config(config_path)

    print(f"Rerun: python scripts/smoke_test.py")
    set_runtime_environment(config)
    seed_everything(int(config["training"]["seed"]))
    enable_fast_torch_runtime()
    device = resolve_device(config)
    criteo_train.run_training(config, repo_root, device, prepare_only=False, force_prepare=True)


if __name__ == "__main__":
    main()

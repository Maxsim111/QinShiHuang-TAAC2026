from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch DeepFM on the TAAC2026 sample dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
        help="Path to the training config yaml.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only generate intermediate features.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild cached features before training.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_workspace_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root.parent / path).resolve()


def set_runtime_environment(config: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device"]["cuda_visible_devices"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_device(config: dict[str, Any]) -> "torch.device":
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This pipeline is configured to run on the RTX 5090D only.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU after masking, found {torch.cuda.device_count()}.")
    device_name = torch.cuda.get_device_name(0)
    expected_keyword = str(config["device"]["required_device_keyword"])
    if expected_keyword not in device_name:
        raise RuntimeError(f"Visible GPU is '{device_name}', expected it to contain '{expected_keyword}'.")
    print(f"Using device: cuda:0 ({device_name})")
    return torch.device("cuda:0")


def maybe_prepare_features(config: dict[str, Any], repo_root: Path, force_prepare: bool) -> dict[str, Any]:
    from DeepFM.utils import load_feature_schema, prepare_taac2026_dataset

    feature_dir = resolve_workspace_path(repo_root, config["paths"]["feature_dir"])
    schema_path = feature_dir / "feature_schema.yaml"
    if force_prepare or not schema_path.exists():
        print(f"Preparing features into: {feature_dir}")
        return prepare_taac2026_dataset(config, repo_root)
    print(f"Reusing cached features from: {feature_dir}")
    return load_feature_schema(feature_dir)


def create_dataloaders(schema: dict[str, Any], config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from DeepFM.utils import ParquetDataset

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    train_path = Path(schema["paths"]["train_split"])
    valid_path = Path(schema["paths"]["valid_split"])

    datasets = {
        "train": ParquetDataset(train_path, schema),
        "valid": ParquetDataset(valid_path, schema),
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    print(
        "Dataset summary:",
        json.dumps(
            {
                split: len(dataset)
                for split, dataset in datasets.items()
            },
            ensure_ascii=False,
        ),
    )
    return {"datasets": datasets, "loaders": loaders}


def instantiate_model(schema: dict[str, Any], config: dict[str, Any], device: "torch.device") -> "torch.nn.Module":
    from DeepFM.model import DeepFM

    sparse_cardinalities = [schema["sparse_cardinalities"][feature] for feature in schema["sparse_features"]]
    model = DeepFM(
        dense_feature_dim=len(schema["dense_features"]),
        sparse_cardinalities=sparse_cardinalities,
        embed_dim=int(config["training"]["embed_dim"]),
        hidden_units=list(config["training"]["hidden_units"]),
        dropout=float(config["training"].get("dropout", 0.0)),
    )
    return model.to(device)


def evaluate(model: "torch.nn.Module", data_loader: Any, device: "torch.device") -> tuple[float, dict[str, float]]:
    import torch
    from DeepFM.utils import compute_classification_metrics

    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for dense_x, sparse_x, labels in data_loader:
            dense_x = dense_x.to(device, non_blocking=True)
            sparse_x = sparse_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            logits = model(dense_x, sparse_x)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            losses.append(loss.item())
            all_labels.append(labels.squeeze(1).cpu().numpy())
            all_probs.append(probabilities.squeeze(1).cpu().numpy())
    metrics = compute_classification_metrics(np.concatenate(all_labels), np.concatenate(all_probs))
    return float(np.mean(losses)), metrics


def build_training_criterion(training_config: dict[str, Any], labels: np.ndarray, device: "torch.device") -> "torch.nn.Module":
    import torch

    if not training_config.get("use_pos_weight", False):
        return torch.nn.BCEWithLogitsLoss()
    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    if positive_count <= 0:
        return torch.nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.6f}")
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_model(
    model: "torch.nn.Module",
    schema: dict[str, Any],
    config: dict[str, Any],
    data_objects: dict[str, Any],
    device: "torch.device",
    repo_root: Path,
) -> None:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from DeepFM.utils import format_metrics

    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_config.get("lr_decay_factor", 0.5)),
        patience=int(training_config.get("lr_decay_patience", 2)),
    )
    train_labels = data_objects["datasets"]["train"].labels
    criterion = build_training_criterion(training_config, train_labels, device)

    log_dir = resolve_workspace_path(repo_root, config["paths"]["log_dir"])
    checkpoint_dir = resolve_workspace_path(repo_root, config["paths"]["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_auc = -float("inf")
    best_state = None
    patience = int(training_config["early_stopping_patience"])
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for dense_x, sparse_x, labels in data_objects["loaders"]["train"]:
            dense_x = dense_x.to(device, non_blocking=True)
            sparse_x = sparse_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(dense_x, sparse_x)
            loss = criterion(logits, labels)
            loss.backward()
            max_grad_norm = float(training_config.get("max_grad_norm", 0.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        valid_loss, valid_metrics = evaluate(model, data_objects["loaders"]["valid"], device)
        scheduler.step(valid_metrics["auc"])

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/valid", valid_loss, epoch)
        writer.add_scalar("metrics/valid_auc", valid_metrics["auc"], epoch)
        writer.add_scalar("metrics/valid_logloss", valid_metrics["logloss"], epoch)
        writer.add_scalar("metrics/valid_accuracy", valid_metrics["accuracy"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:02d} train_loss={train_loss:.6f}")
        print(format_metrics("valid", valid_metrics))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                **{f"valid_{key}": float(value) for key, value in valid_metrics.items()},
            }
        )

        if valid_metrics["auc"] > best_auc:
            best_auc = valid_metrics["auc"]
            epochs_without_improvement = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_metrics": valid_metrics,
                "schema": schema,
                "config": config,
            }
            torch.save(best_state, checkpoint_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    final_state = {
        "epoch": history[-1]["epoch"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_auc": best_auc,
        "schema": schema,
        "config": config,
    }
    torch.save(final_state, checkpoint_dir / "final_model.pt")
    writer.add_hparams(
        {
            "embed_dim": int(training_config["embed_dim"]),
            "batch_size": int(training_config["batch_size"]),
            "learning_rate": float(training_config["learning_rate"]),
            "weight_decay": float(training_config["weight_decay"]),
        },
        {
            "hparam/best_valid_auc": best_auc,
            "hparam/final_train_loss": history[-1]["train_loss"],
            "hparam/final_valid_logloss": history[-1]["valid_logloss"],
        },
    )
    writer.close()
    print(f"Saved checkpoints to: {checkpoint_dir / 'best_model.pt'} and {checkpoint_dir / 'final_model.pt'}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    config = load_config(config_path)
    set_runtime_environment(config)

    import torch

    seed_everything(int(config["training"]["seed"]))
    device = validate_device(config)

    schema = maybe_prepare_features(config, repo_root, args.force_prepare)
    print(
        json.dumps(
            {
                "raw_data": config["paths"]["raw_data"],
                "target_action_type": config["data"]["target_action_type"],
                "num_dense_features": len(schema["dense_features"]),
                "num_sparse_features": len(schema["sparse_features"]),
                "embed_dim": config["training"]["embed_dim"],
                "hidden_units": config["training"]["hidden_units"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.prepare_only:
        print("Feature preparation completed.")
        return

    data_objects = create_dataloaders(schema, config, repo_root)
    model = instantiate_model(schema, config, device)
    train_model(model, schema, config, data_objects, device, repo_root)


if __name__ == "__main__":
    main()

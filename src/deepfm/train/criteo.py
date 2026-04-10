"""Training pipeline for the full Criteo dataset."""
from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepfm.common.runtime import resolve_workspace_path
from deepfm.data.criteo import prepare_criteo_dataset
from deepfm.models import DeepFM


def resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported amp_dtype={dtype_name!r}. Use one of: {sorted(mapping)}")
    return mapping[key]


def to_device(batch: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_np, sparse_np, labels_np = batch
    dense_x = torch.from_numpy(dense_np).to(device, non_blocking=True)
    sparse_x = torch.from_numpy(sparse_np).to(device, non_blocking=True)
    labels = torch.from_numpy(labels_np).to(device, non_blocking=True).unsqueeze(1)
    return dense_x, sparse_x, labels


def evaluate_model(model, dataset, device, config, amp_ctx, criterion):
    from deepfm.data.criteo import compute_metrics

    batch_size = int(config["training"].get("eval_batch_size", config["training"]["batch_size"] * 2))
    shuffle_block_rows = int(config["training"].get("shuffle_block_rows", 262_144))
    model.eval()
    losses, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataset.iter_batches(
            batch_size,
            shuffle=False,
            seed=0,
            shuffle_block_rows=shuffle_block_rows,
        ):
            dense_x, sparse_x, labels = to_device(batch, device)
            with amp_ctx():
                logits = model(dense_x, sparse_x)
                loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            losses.append(float(loss.item()))
            all_labels.append(labels.squeeze(1).cpu().numpy())
            all_probs.append(probs.squeeze(1).cpu().numpy())

    metrics = compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))
    return float(np.mean(losses)), metrics


def instantiate_model(schema: dict[str, Any], config: dict[str, Any], device: "torch.device") -> "torch.nn.Module":
    sparse_cards = [schema["sparse_cardinalities"][col] for col in schema["sparse_features"]]
    model = DeepFM(
        dense_feature_dim=len(schema["dense_features"]),
        sparse_cardinalities=sparse_cards,
        embed_dim=int(config["training"]["embed_dim"]),
        hidden_units=list(config["training"]["hidden_units"]),
        dropout=float(config["training"].get("dropout", 0.0)),
    )
    return model.to(device)


def run_training(
    config: dict[str, Any],
    repo_root: Path,
    device: "torch.device",
    *,
    prepare_only: bool = False,
    force_prepare: bool = False,
) -> dict[str, Any]:
    tc = config["training"]
    result = prepare_criteo_dataset(config, repo_root, force_prepare=force_prepare)
    if prepare_only:
        print("Preparation finished. Training was skipped because prepare_only=True.")
        return {"schema": result["schema"], "prepared_only": True}

    schema = result["schema"]
    datasets = result["datasets"]
    stats = result["stats"]
    model = instantiate_model(schema, config, device)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {total_params:,}")

    amp_enabled = device.type == "cuda" and bool(tc.get("mixed_precision", True))
    amp_dtype = resolve_amp_dtype(str(tc.get("amp_dtype", "bfloat16")))
    use_grad_scaler = amp_enabled and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    def amp_ctx():
        if not amp_enabled:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    train_stats = stats["train"]
    if tc.get("use_pos_weight", False):
        pos_count = max(int(train_stats["positive_count"]), 1)
        neg_count = max(int(train_stats["negative_count"]), 1)
        pos_weight = torch.tensor([neg_count / pos_count], device=device, dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using positive class weight: {pos_weight.item():.4f}")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tc["learning_rate"]),
        weight_decay=float(tc["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(tc.get("lr_decay_factor", 0.5)),
        patience=int(tc.get("lr_decay_patience", 1)),
    )

    checkpoint_dir = resolve_workspace_path(repo_root, config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -math.inf
    patience_cnt = 0
    patience = int(tc["early_stopping_patience"])
    batch_size = int(tc["batch_size"])
    shuffle_block_rows = int(tc.get("shuffle_block_rows", 262_144))
    log_interval = int(tc.get("log_interval", 500))
    train_split = datasets["train"]
    valid_split = datasets["valid"]
    train_steps = math.ceil(train_split.num_rows / batch_size)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(tc["epochs"]) + 1):
        model.train()
        train_losses = []

        for step, batch in enumerate(
            train_split.iter_batches(
                batch_size,
                shuffle=True,
                seed=int(tc["seed"]) + epoch,
                shuffle_block_rows=shuffle_block_rows,
            ),
            start=1,
        ):
            dense_x, sparse_x, labels = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                logits = model(dense_x, sparse_x)
                loss = criterion(logits, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if float(tc.get("max_grad_norm", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(tc.get("max_grad_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["max_grad_norm"]))
                optimizer.step()

            train_losses.append(float(loss.item()))
            if step % log_interval == 0 or step == train_steps:
                avg_loss = float(np.mean(train_losses[-log_interval:]))
                print(f"Epoch {epoch:02d} Step {step:05d}/{train_steps:05d} train_loss={avg_loss:.6f}")

        train_loss = float(np.mean(train_losses))
        val_loss, val_metrics = evaluate_model(model, valid_split, device, config, amp_ctx, criterion)
        scheduler.step(val_metrics["auc"])
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} AUC={val_metrics['auc']:.6f} "
            f"LogLoss={val_metrics['logloss']:.6f} Acc={val_metrics['accuracy']:.4f} lr={lr:.2e}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": val_loss,
                **{f"valid_{key}": float(value) for key, value in val_metrics.items()},
            }
        )

        if val_metrics["auc"] > best_auc:
            best_auc = float(val_metrics["auc"])
            patience_cnt = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "metrics": val_metrics,
                    "schema": schema,
                    "config": config,
                },
                checkpoint_dir / "best_model.pt",
            )
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Best valid AUC: {best_auc:.6f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")
    return {
        "schema": schema,
        "checkpoint_path": checkpoint_dir / "best_model.pt",
        "best_valid_auc": best_auc,
        "history": history,
    }


def run_evaluation(
    config: dict[str, Any],
    repo_root: Path,
    device: "torch.device",
    *,
    checkpoint_path: Path | None = None,
    split: str = "valid",
) -> dict[str, Any]:
    import torch

    if split != "valid":
        raise ValueError("Criteo evaluation currently supports split='valid' only.")

    result = prepare_criteo_dataset(config, repo_root, force_prepare=False)
    schema = result["schema"]
    datasets = result["datasets"]
    model = instantiate_model(schema, config, device)

    tc = config["training"]
    amp_enabled = device.type == "cuda" and bool(tc.get("mixed_precision", True))
    amp_dtype = resolve_amp_dtype(str(tc.get("amp_dtype", "bfloat16")))

    def amp_ctx():
        if not amp_enabled:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    criterion = torch.nn.BCEWithLogitsLoss()
    if checkpoint_path is None:
        checkpoint_path = resolve_workspace_path(repo_root, config["paths"]["checkpoint_dir"]) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss, metrics = evaluate_model(model, datasets[split], device, config, amp_ctx, criterion)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{split}_loss={loss:.6f}")
    print(
        f"{split} "
        f"AUC={metrics['auc']:.6f} "
        f"LogLoss={metrics['logloss']:.6f} "
        f"Accuracy={metrics['accuracy']:.6f}"
    )
    return {"loss": loss, "metrics": metrics, "checkpoint_path": checkpoint_path}

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def reset_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def build_training_summary(
    *,
    task: str,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    checkpoint_path: Path,
    best_epoch: int | None,
    best_metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "task": task,
        "timestamp_utc": now_utc_iso(),
        "config_path": config.get("meta", {}).get("config_path"),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "num_epochs_recorded": len(history),
        "history_files": {
            "jsonl": "train_history.jsonl",
            "csv": "train_history.csv",
        },
    }
    if history:
        summary["final_epoch"] = history[-1]["epoch"]
        summary["final_metrics"] = history[-1]
    if extra:
        summary.update(extra)
    return summary


def build_evaluation_report(
    *,
    task: str,
    split: str,
    config: dict[str, Any],
    checkpoint_path: Path,
    loss: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "split": split,
        "timestamp_utc": now_utc_iso(),
        "config_path": config.get("meta", {}).get("config_path"),
        "checkpoint_path": str(checkpoint_path),
        "loss": float(loss),
        "metrics": metrics,
    }

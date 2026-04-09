from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm


FEATURE_KIND_DENSE = "dense"
FEATURE_KIND_SPARSE = "sparse"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_workspace_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root.parent / path).resolve()


def normalize_sparse_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return str(int(value)) if float(value).is_integer() else f"{float(value):.6f}"
    return str(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def register_feature(specs: dict[str, dict[str, Any]], name: str, kind: str, source: str, transform: str) -> None:
    if name not in specs:
        specs[name] = {"kind": kind, "source": source, "transform": transform}


def compute_array_statistics(values: Any) -> dict[str, float]:
    if values is None:
        array = np.asarray([], dtype=np.float32)
    else:
        array = np.asarray(values).astype(np.float32).reshape(-1)
    if array.size == 0:
        return {
            "len": 0.0,
            "sum": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "last": 0.0,
            "std": 0.0,
            "nunique": 0.0,
        }
    return {
        "len": float(array.size),
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "last": float(array[-1]),
        "std": float(array.std()),
        "nunique": float(np.unique(array).size),
    }


def add_dense_feature(
    row_features: dict[str, Any],
    feature_specs: dict[str, dict[str, Any]],
    name: str,
    value: float,
    source: str,
    transform: str,
) -> None:
    row_features[name] = float(value)
    register_feature(feature_specs, name, FEATURE_KIND_DENSE, source, transform)


def add_sparse_feature(
    row_features: dict[str, Any],
    feature_specs: dict[str, dict[str, Any]],
    name: str,
    value: Any,
    source: str,
    transform: str,
) -> None:
    row_features[name] = normalize_sparse_value(value)
    register_feature(feature_specs, name, FEATURE_KIND_SPARSE, source, transform)


def extract_entity_features(
    feature_list: Any,
    entity_name: str,
    row_features: dict[str, Any],
    feature_specs: dict[str, dict[str, Any]],
) -> None:
    iterable = list(feature_list) if feature_list is not None else []
    add_dense_feature(
        row_features,
        feature_specs,
        f"{entity_name}_feature_count",
        float(len(iterable)),
        entity_name,
        "count_entries",
    )
    for feature in iterable:
        feature_id = int(feature.get("feature_id"))
        base_name = f"{entity_name}_f{feature_id}"
        if feature.get("int_value") is not None:
            add_sparse_feature(
                row_features,
                feature_specs,
                f"{base_name}_int",
                feature["int_value"],
                entity_name,
                "categorical_int_value",
            )
        if feature.get("float_value") is not None:
            add_dense_feature(
                row_features,
                feature_specs,
                f"{base_name}_float",
                safe_float(feature["float_value"]),
                entity_name,
                "numeric_float_value",
            )
        if feature.get("int_array") is not None:
            stats = compute_array_statistics(feature.get("int_array"))
            for stat_name, stat_value in stats.items():
                add_dense_feature(
                    row_features,
                    feature_specs,
                    f"{base_name}_int_array_{stat_name}",
                    stat_value,
                    entity_name,
                    "int_array_stats",
                )
        if feature.get("float_array") is not None:
            stats = compute_array_statistics(feature.get("float_array"))
            for stat_name, stat_value in stats.items():
                add_dense_feature(
                    row_features,
                    feature_specs,
                    f"{base_name}_float_array_{stat_name}",
                    stat_value,
                    entity_name,
                    "float_array_stats",
                )


def extract_sequence_features(
    seq_feature: dict[str, Any],
    item_id: Any,
    timestamp: Any,
    row_features: dict[str, Any],
    feature_specs: dict[str, dict[str, Any]],
) -> None:
    current_item_id = safe_float(item_id)
    current_timestamp = safe_float(timestamp)
    for sequence_name, raw_entries in seq_feature.items():
        entries = list(raw_entries) if raw_entries is not None else []
        prefix = f"seq_{sequence_name}"
        add_dense_feature(
            row_features,
            feature_specs,
            f"{prefix}_feature_count",
            float(len(entries)),
            sequence_name,
            "count_entries",
        )
        max_historical_timestamp = None
        overlap_count = 0.0
        overlap_flag = 0.0
        for entry in entries:
            feature_id = int(entry.get("feature_id"))
            base_name = f"{prefix}_f{feature_id}"
            if entry.get("int_value") is not None:
                add_sparse_feature(
                    row_features,
                    feature_specs,
                    f"{base_name}_int",
                    entry["int_value"],
                    sequence_name,
                    "categorical_int_value",
                )
            if entry.get("float_value") is not None:
                add_dense_feature(
                    row_features,
                    feature_specs,
                    f"{base_name}_float",
                    safe_float(entry["float_value"]),
                    sequence_name,
                    "numeric_float_value",
                )
            for array_key in ("int_array", "float_array"):
                if entry.get(array_key) is None:
                    continue
                values = np.asarray(entry[array_key]).reshape(-1)
                stats = compute_array_statistics(values)
                for stat_name, stat_value in stats.items():
                    add_dense_feature(
                        row_features,
                        feature_specs,
                        f"{base_name}_{array_key}_{stat_name}",
                        stat_value,
                        sequence_name,
                        f"{array_key}_stats",
                    )
                if array_key == "int_array" and values.size > 0:
                    numeric_values = values.astype(np.float64)
                    if sequence_name == "item_seq":
                        matches = np.isclose(numeric_values, current_item_id)
                        match_count = float(matches.sum())
                        overlap_count += match_count
                        overlap_flag = max(overlap_flag, 1.0 if match_count > 0 else 0.0)
                    plausible_timestamps = numeric_values[(numeric_values > 1_500_000_000) & (numeric_values <= current_timestamp)]
                    if plausible_timestamps.size > 0:
                        candidate_ts = float(plausible_timestamps.max())
                        if max_historical_timestamp is None or candidate_ts > max_historical_timestamp:
                            max_historical_timestamp = candidate_ts
        add_dense_feature(
            row_features,
            feature_specs,
            f"{prefix}_contains_current_item",
            overlap_flag,
            sequence_name,
            "current_item_overlap_flag",
        )
        add_dense_feature(
            row_features,
            feature_specs,
            f"{prefix}_current_item_overlap_count",
            overlap_count,
            sequence_name,
            "current_item_overlap_count",
        )
        recency_gap = 0.0 if max_historical_timestamp is None else max(current_timestamp - max_historical_timestamp, 0.0)
        add_dense_feature(
            row_features,
            feature_specs,
            f"{prefix}_recent_gap_seconds",
            recency_gap,
            sequence_name,
            "recent_timestamp_gap",
        )


def extract_time_features(
    timestamp: Any,
    row_features: dict[str, Any],
    feature_specs: dict[str, dict[str, Any]],
) -> None:
    ts = int(safe_float(timestamp))
    dt = pd.to_datetime(ts, unit="s", utc=True)
    add_sparse_feature(row_features, feature_specs, "event_hour", dt.hour, "timestamp", "hour_of_day")
    add_sparse_feature(row_features, feature_specs, "event_weekday", dt.weekday(), "timestamp", "weekday")
    add_sparse_feature(row_features, feature_specs, "event_month", dt.month, "timestamp", "month")
    add_sparse_feature(row_features, feature_specs, "event_daypart", dt.hour // 6, "timestamp", "daypart_bucket")
    add_dense_feature(
        row_features,
        feature_specs,
        "event_timestamp",
        float(ts),
        "timestamp",
        "raw_timestamp_dense",
    )


def build_label(labels: Any, target_action_type: int) -> int:
    for label in labels:
        if int(label["action_type"]) == target_action_type:
            return 1
    return 0


def build_records(
    raw_df: pd.DataFrame,
    target_action_type: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], Counter]:
    feature_specs: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    action_counter: Counter = Counter()
    for row in tqdm(raw_df.itertuples(index=False), total=len(raw_df), desc="Extracting features"):
        row_features: dict[str, Any] = {}
        row_dict = row._asdict()
        for label in row_dict["label"]:
            action_counter[int(label["action_type"])] += 1

        add_sparse_feature(row_features, feature_specs, "user_id", row_dict["user_id"], "user_id", "identity")
        add_sparse_feature(row_features, feature_specs, "item_id", row_dict["item_id"], "item_id", "identity")
        extract_time_features(row_dict["timestamp"], row_features, feature_specs)
        extract_entity_features(row_dict["user_feature"], "user", row_features, feature_specs)
        extract_entity_features(row_dict["item_feature"], "item", row_features, feature_specs)
        extract_sequence_features(row_dict["seq_feature"], row_dict["item_id"], row_dict["timestamp"], row_features, feature_specs)

        row_features["label"] = build_label(row_dict["label"], target_action_type)
        row_features["timestamp"] = int(safe_float(row_dict["timestamp"]))
        records.append(row_features)
    feature_frame = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return feature_frame, feature_specs, action_counter


def split_by_time(feature_frame: pd.DataFrame, train_ratio: float, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 or not np.isclose(train_ratio + valid_ratio, 1.0):
        raise ValueError("train_ratio and valid_ratio must be within (0,1) and sum to 1.0")
    total = len(feature_frame)
    train_end = max(1, int(total * train_ratio))
    train_end = min(train_end, total - 1)
    train_df = feature_frame.iloc[:train_end].copy()
    valid_df = feature_frame.iloc[train_end:].copy()
    return train_df, valid_df


def infer_feature_lists(feature_specs: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    dense_features = sorted(name for name, spec in feature_specs.items() if spec["kind"] == FEATURE_KIND_DENSE)
    sparse_features = sorted(name for name, spec in feature_specs.items() if spec["kind"] == FEATURE_KIND_SPARSE)
    return dense_features, sparse_features


def fit_dense_stats(train_df: pd.DataFrame, dense_features: list[str]) -> dict[str, dict[str, float]]:
    dense_stats: dict[str, dict[str, float]] = {}
    for feature in dense_features:
        values = pd.to_numeric(train_df[feature], errors="coerce").fillna(0.0).astype(np.float32)
        mean = float(values.mean())
        std = float(values.std())
        dense_stats[feature] = {"mean": mean, "std": std if std > 1e-6 else 1.0}
    return dense_stats


def fit_sparse_vocabs(train_df: pd.DataFrame, sparse_features: list[str]) -> dict[str, dict[str, int]]:
    sparse_vocabs: dict[str, dict[str, int]] = {}
    for feature in sparse_features:
        series = train_df[feature].fillna("__MISSING__").map(normalize_sparse_value)
        unique_values = list(dict.fromkeys(series.tolist()))
        vocab = {"__UNK__": 0}
        for value in unique_values:
            if value == "__UNK__":
                continue
            vocab[value] = len(vocab)
        sparse_vocabs[feature] = vocab
    return sparse_vocabs


def transform_split(
    df: pd.DataFrame,
    dense_features: list[str],
    sparse_features: list[str],
    dense_stats: dict[str, dict[str, float]],
    sparse_vocabs: dict[str, dict[str, int]],
) -> pd.DataFrame:
    transformed_columns: dict[str, Any] = {}
    for feature in dense_features:
        values = pd.to_numeric(df.get(feature, 0.0), errors="coerce").fillna(0.0).astype(np.float32)
        stats = dense_stats[feature]
        transformed_columns[feature] = ((values - stats["mean"]) / stats["std"]).astype(np.float32)
    for feature in sparse_features:
        vocab = sparse_vocabs[feature]
        series = df.get(feature, "__MISSING__")
        if not isinstance(series, pd.Series):
            series = pd.Series([series] * len(df), index=df.index)
        transformed_columns[feature] = series.fillna("__MISSING__").map(
            lambda item: vocab.get(normalize_sparse_value(item), 0)
        ).astype(np.int64)
    transformed_columns["label"] = df["label"].astype(np.float32)
    transformed_columns["timestamp"] = df["timestamp"].astype(np.int64)
    return pd.DataFrame(transformed_columns, index=df.index)


def build_feature_schema(
    dense_features: list[str],
    sparse_features: list[str],
    feature_specs: dict[str, dict[str, Any]],
    sparse_vocabs: dict[str, dict[str, int]],
    config: dict[str, Any],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "label_column": "label",
        "timestamp_column": "timestamp",
        "target_action_type": int(config["data"]["target_action_type"]),
        "dense_features": dense_features,
        "sparse_features": sparse_features,
        "feature_specs": feature_specs,
        "sparse_cardinalities": {feature: len(vocab) for feature, vocab in sparse_vocabs.items()},
        "training": {
            "embed_dim": int(config["training"]["embed_dim"]),
            "hidden_units": list(config["training"]["hidden_units"]),
            "dropout": float(config["training"].get("dropout", 0.0)),
        },
        "paths": output_paths,
    }


def compute_dataset_stats(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    dense_features: list[str],
    sparse_features: list[str],
    action_counter: Counter,
    target_action_type: int,
) -> dict[str, Any]:
    def split_stats(df: pd.DataFrame) -> dict[str, float]:
        return {
            "num_rows": int(len(df)),
            "positive_ratio": float(df["label"].mean()),
            "positive_count": int(df["label"].sum()),
            "negative_count": int((1 - df["label"]).sum()),
        }

    return {
        "target_action_type": target_action_type,
        "action_type_distribution": {str(key): int(value) for key, value in sorted(action_counter.items())},
        "num_dense_features": len(dense_features),
        "num_sparse_features": len(sparse_features),
        "train": split_stats(train_df),
        "valid": split_stats(valid_df),
    }


def prepare_taac2026_dataset(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    paths = config["paths"]
    raw_path = resolve_workspace_path(repo_root, paths["raw_data"])
    feature_dir = ensure_directory(resolve_workspace_path(repo_root, paths["feature_dir"]))
    encoder_dir = ensure_directory(feature_dir / "encoders")

    raw_df = pd.read_parquet(raw_path).sort_values("timestamp").reset_index(drop=True)
    feature_frame, feature_specs, action_counter = build_records(raw_df, int(config["data"]["target_action_type"]))
    train_df, valid_df = split_by_time(
        feature_frame,
        float(config["data"]["train_ratio"]),
        float(config["data"]["valid_ratio"]),
    )

    dense_features, sparse_features = infer_feature_lists(feature_specs)
    dense_stats = fit_dense_stats(train_df, dense_features)
    sparse_vocabs = fit_sparse_vocabs(train_df, sparse_features)

    transformed_train = transform_split(train_df, dense_features, sparse_features, dense_stats, sparse_vocabs)
    transformed_valid = transform_split(valid_df, dense_features, sparse_features, dense_stats, sparse_vocabs)

    train_path = feature_dir / "train.parquet"
    valid_path = feature_dir / "valid.parquet"
    transformed_train.to_parquet(train_path, index=False)
    transformed_valid.to_parquet(valid_path, index=False)

    dump_json(encoder_dir / "dense_stats.json", dense_stats)
    dump_json(encoder_dir / "sparse_vocabs.json", sparse_vocabs)

    output_paths = {
        "raw_data": str(raw_path),
        "feature_dir": str(feature_dir),
        "train_split": str(train_path),
        "valid_split": str(valid_path),
        "encoder_dir": str(encoder_dir),
    }
    schema = build_feature_schema(dense_features, sparse_features, feature_specs, sparse_vocabs, config, output_paths)
    dump_yaml(feature_dir / "feature_schema.yaml", schema)

    stats = compute_dataset_stats(
        transformed_train,
        transformed_valid,
        dense_features,
        sparse_features,
        action_counter,
        int(config["data"]["target_action_type"]),
    )
    dump_json(feature_dir / "stats.json", stats)
    return schema


def load_feature_schema(feature_dir: Path) -> dict[str, Any]:
    return load_yaml(feature_dir / "feature_schema.yaml")


class ParquetDataset(Dataset):
    def __init__(self, parquet_path: Path, schema: dict[str, Any]) -> None:
        frame = pd.read_parquet(parquet_path)
        self.dense_features = schema["dense_features"]
        self.sparse_features = schema["sparse_features"]
        self.labels = frame[schema["label_column"]].to_numpy(dtype=np.float32)
        self.timestamps = frame[schema["timestamp_column"]].to_numpy(dtype=np.int64)
        self.dense = (
            frame[self.dense_features].to_numpy(dtype=np.float32)
            if self.dense_features
            else np.zeros((len(frame), 0), dtype=np.float32)
        )
        self.sparse = (
            frame[self.sparse_features].to_numpy(dtype=np.int64)
            if self.sparse_features
            else np.zeros((len(frame), 0), dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dense_x = torch.from_numpy(self.dense[index])
        sparse_x = torch.from_numpy(self.sparse[index])
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return dense_x, sparse_x, label


def compute_classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.float32)
    probabilities = probabilities.astype(np.float32)
    predictions = (probabilities >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    metrics["auc"] = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size > 1 else float("nan")
    return metrics


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} "
        f"AUC={metrics['auc']:.6f} "
        f"LogLoss={metrics['logloss']:.6f} "
        f"Accuracy={metrics['accuracy']:.6f}"
    )

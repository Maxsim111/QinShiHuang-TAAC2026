"""Out-of-core Criteo dataset utilities for DeepFM training."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml
from pandas.util import hash_pandas_object

from deepfm.common.runtime import resolve_workspace_path
from deepfm.eval.classification import compute_classification_metrics

NUM_DENSE = 13  # I1 - I13
NUM_SPARSE = 26  # C1 - C26
DENSE_COLS = [f"I{i}" for i in range(1, NUM_DENSE + 1)]
SPARSE_COLS = [f"C{i}" for i in range(1, NUM_SPARSE + 1)]
ALL_COLS = ["label"] + DENSE_COLS + SPARSE_COLS

MEMMAP_DIRNAME = "memmap"
PREPARE_CONFIG_NAME = "prepare_config.json"
DATA_STATS_NAME = "stats.json"


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_split_config(train_ratio: float, valid_ratio: float) -> None:
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1:
        raise ValueError("train_ratio and valid_ratio must both be within (0, 1)")
    if not math.isclose(train_ratio + valid_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_ratio and valid_ratio must sum to 1.0")


def count_criteo_rows(path: Path, num_samples: int | None = None, buffer_size: int = 16 * 1024 * 1024) -> int:
    total = 0
    last_byte = b""
    with path.open("rb") as file:
        while True:
            chunk = file.read(buffer_size)
            if not chunk:
                break
            total += chunk.count(b"\n")
            last_byte = chunk[-1:]
            if num_samples is not None and total >= num_samples:
                return num_samples
    if last_byte and last_byte != b"\n":
        total += 1
    return total


def iter_criteo_chunks(path: Path, num_samples: int | None, chunk_size: int) -> Iterator[pd.DataFrame]:
    dtype_map: dict[str, Any] = {"label": np.int8}
    dtype_map.update({col: np.float32 for col in DENSE_COLS})
    dtype_map.update({col: "string" for col in SPARSE_COLS})

    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=ALL_COLS,
        nrows=num_samples,
        na_values="",
        keep_default_na=True,
        chunksize=chunk_size,
        dtype=dtype_map,
        low_memory=False,
    )
    yield from reader


def fit_dense_stats_streaming(
    path: Path,
    num_samples: int | None,
    chunk_size: int,
    train_rows: int,
) -> dict[str, dict[str, float]]:
    processed = 0
    sums = np.zeros(NUM_DENSE, dtype=np.float64)
    squared_sums = np.zeros(NUM_DENSE, dtype=np.float64)

    for chunk in iter_criteo_chunks(path, num_samples, chunk_size):
        remaining = train_rows - processed
        if remaining <= 0:
            break
        train_chunk = chunk.iloc[:remaining]
        dense = train_chunk[DENSE_COLS].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        sums += dense.sum(axis=0, dtype=np.float64)
        squared_sums += np.square(dense, dtype=np.float64).sum(axis=0, dtype=np.float64)
        processed += len(train_chunk)

    if processed != train_rows:
        raise ValueError(f"Expected {train_rows} training rows but only processed {processed}")

    means = sums / max(train_rows, 1)
    variances = np.maximum(squared_sums / max(train_rows, 1) - np.square(means), 1.0e-6)
    stds = np.sqrt(variances)
    return {
        col: {"mean": float(means[index]), "std": float(stds[index])}
        for index, col in enumerate(DENSE_COLS)
    }


def build_hash_cardinalities(data_cfg: dict[str, Any]) -> dict[str, int]:
    default_bucket_size = int(data_cfg.get("hash_bucket_size", 200_000))
    per_feature_bucket_sizes = data_cfg.get("hash_bucket_sizes", {})
    cardinalities: dict[str, int] = {}
    for col in SPARSE_COLS:
        bucket_size = int(per_feature_bucket_sizes.get(col, default_bucket_size))
        if bucket_size < 2:
            raise ValueError(f"Hash bucket size for {col} must be >= 2, got {bucket_size}")
        cardinalities[col] = bucket_size
    return cardinalities


def build_vocabs_streaming(
    path: Path,
    num_samples: int | None,
    chunk_size: int,
    train_rows: int,
    threshold: int,
    max_vocab_size: int | None,
) -> dict[str, dict[str, int]]:
    counters = {col: Counter() for col in SPARSE_COLS}
    processed = 0

    for chunk in iter_criteo_chunks(path, num_samples, chunk_size):
        remaining = train_rows - processed
        if remaining <= 0:
            break
        train_chunk = chunk.iloc[:remaining]
        for col in SPARSE_COLS:
            values = train_chunk[col].fillna("__MISSING__").astype(str)
            counters[col].update(values.tolist())
        processed += len(train_chunk)

    vocabs: dict[str, dict[str, int]] = {}
    for col in SPARSE_COLS:
        items = [(value, count) for value, count in counters[col].items() if count >= threshold]
        items.sort(key=lambda item: (-item[1], item[0]))
        if max_vocab_size is not None:
            items = items[:max_vocab_size]
        vocab = {"__UNK__": 0}
        for value, _ in items:
            if value == "__UNK__":
                continue
            vocab[value] = len(vocab)
        vocabs[col] = vocab
    return vocabs


def build_prepare_signature(config: dict[str, Any], resolved_paths: dict[str, Path]) -> dict[str, Any]:
    data_cfg = config["data"]
    return {
        "raw_data": str(resolved_paths["raw_data"]),
        "feature_dir": str(resolved_paths["feature_dir"]),
        "num_samples": data_cfg.get("num_samples"),
        "train_ratio": float(data_cfg["train_ratio"]),
        "valid_ratio": float(data_cfg["valid_ratio"]),
        "chunk_size": int(data_cfg.get("chunk_size", 250_000)),
        "sparse_encoding": str(data_cfg.get("sparse_encoding", "hash")),
        "hash_bucket_size": data_cfg.get("hash_bucket_size"),
        "hash_bucket_sizes": data_cfg.get("hash_bucket_sizes", {}),
        "sparse_threshold": data_cfg.get("sparse_threshold"),
        "max_vocab_size": data_cfg.get("max_vocab_size"),
    }


def expected_memmap_paths(memmap_dir: Path) -> dict[str, dict[str, Path]]:
    return {
        "train": {
            "dense": memmap_dir / "train_dense.npy",
            "sparse": memmap_dir / "train_sparse.npy",
            "labels": memmap_dir / "train_labels.npy",
        },
        "valid": {
            "dense": memmap_dir / "valid_dense.npy",
            "sparse": memmap_dir / "valid_sparse.npy",
            "labels": memmap_dir / "valid_labels.npy",
        },
    }


def cache_is_valid(feature_dir: Path, signature: dict[str, Any]) -> bool:
    prepare_config_path = feature_dir / PREPARE_CONFIG_NAME
    schema_path = feature_dir / "feature_schema.yaml"
    stats_path = feature_dir / DATA_STATS_NAME
    memmap_paths = expected_memmap_paths(feature_dir / MEMMAP_DIRNAME)

    if not prepare_config_path.exists() or not schema_path.exists() or not stats_path.exists():
        return False
    if not all(path.exists() for split_paths in memmap_paths.values() for path in split_paths.values()):
        return False

    cached_signature = load_json(prepare_config_path)
    return cached_signature == signature


def dense_stats_to_arrays(dense_stats: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray([dense_stats[col]["mean"] for col in DENSE_COLS], dtype=np.float32)
    stds = np.asarray([dense_stats[col]["std"] for col in DENSE_COLS], dtype=np.float32)
    return means, stds


def transform_dense_chunk(chunk: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    dense = chunk[DENSE_COLS].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    dense = (dense - means) / stds
    return dense.astype(np.float32, copy=False)


def transform_sparse_hash_chunk(chunk: pd.DataFrame, sparse_cardinalities: dict[str, int]) -> np.ndarray:
    result = np.empty((len(chunk), NUM_SPARSE), dtype=np.int32)
    for index, col in enumerate(SPARSE_COLS):
        values = chunk[col].fillna("__MISSING__").astype(str)
        hashed = hash_pandas_object(values, index=False, categorize=False).to_numpy(dtype=np.uint64, copy=False)
        bucket_size = sparse_cardinalities[col]
        result[:, index] = ((hashed % np.uint64(bucket_size - 1)) + 1).astype(np.int32, copy=False)
    return result


def transform_sparse_vocab_chunk(chunk: pd.DataFrame, sparse_vocabs: dict[str, dict[str, int]]) -> np.ndarray:
    result = np.zeros((len(chunk), NUM_SPARSE), dtype=np.int32)
    for index, col in enumerate(SPARSE_COLS):
        vocab = sparse_vocabs[col]
        values = chunk[col].fillna("__MISSING__").astype(str)
        mapped = values.map(lambda item: vocab.get(item, 0))
        result[:, index] = mapped.to_numpy(dtype=np.int32, copy=False)
    return result


def create_memmaps(memmap_dir: Path, train_rows: int, valid_rows: int) -> dict[str, dict[str, np.memmap]]:
    memmap_dir.mkdir(parents=True, exist_ok=True)
    paths = expected_memmap_paths(memmap_dir)
    return {
        "train": {
            "dense": np.lib.format.open_memmap(paths["train"]["dense"], mode="w+", dtype=np.float32, shape=(train_rows, NUM_DENSE)),
            "sparse": np.lib.format.open_memmap(paths["train"]["sparse"], mode="w+", dtype=np.int32, shape=(train_rows, NUM_SPARSE)),
            "labels": np.lib.format.open_memmap(paths["train"]["labels"], mode="w+", dtype=np.uint8, shape=(train_rows,)),
        },
        "valid": {
            "dense": np.lib.format.open_memmap(paths["valid"]["dense"], mode="w+", dtype=np.float32, shape=(valid_rows, NUM_DENSE)),
            "sparse": np.lib.format.open_memmap(paths["valid"]["sparse"], mode="w+", dtype=np.int32, shape=(valid_rows, NUM_SPARSE)),
            "labels": np.lib.format.open_memmap(paths["valid"]["labels"], mode="w+", dtype=np.uint8, shape=(valid_rows,)),
        },
    }


def populate_memmaps(
    path: Path,
    num_samples: int | None,
    chunk_size: int,
    train_rows: int,
    dense_stats: dict[str, dict[str, float]],
    sparse_encoding: str,
    sparse_cardinalities: dict[str, int],
    sparse_vocabs: dict[str, dict[str, int]] | None,
    memmaps: dict[str, dict[str, np.memmap]],
) -> dict[str, Any]:
    means, stds = dense_stats_to_arrays(dense_stats)
    train_offset = 0
    valid_offset = 0
    train_positive = 0
    valid_positive = 0

    for chunk in iter_criteo_chunks(path, num_samples, chunk_size):
        dense = transform_dense_chunk(chunk, means, stds)
        if sparse_encoding == "hash":
            sparse = transform_sparse_hash_chunk(chunk, sparse_cardinalities)
        else:
            if sparse_vocabs is None:
                raise ValueError("sparse_vocabs must be provided when sparse_encoding='vocab'")
            sparse = transform_sparse_vocab_chunk(chunk, sparse_vocabs)
        labels = chunk["label"].fillna(0).to_numpy(dtype=np.uint8, copy=False)

        remaining_train = max(train_rows - train_offset, 0)
        take_train = min(remaining_train, len(chunk))
        if take_train > 0:
            train_slice = slice(train_offset, train_offset + take_train)
            memmaps["train"]["dense"][train_slice] = dense[:take_train]
            memmaps["train"]["sparse"][train_slice] = sparse[:take_train]
            memmaps["train"]["labels"][train_slice] = labels[:take_train]
            train_positive += int(labels[:take_train].sum(dtype=np.int64))
            train_offset += take_train

        take_valid = len(chunk) - take_train
        if take_valid > 0:
            valid_slice = slice(valid_offset, valid_offset + take_valid)
            memmaps["valid"]["dense"][valid_slice] = dense[take_train:]
            memmaps["valid"]["sparse"][valid_slice] = sparse[take_train:]
            memmaps["valid"]["labels"][valid_slice] = labels[take_train:]
            valid_positive += int(labels[take_train:].sum(dtype=np.int64))
            valid_offset += take_valid

    for split_memmaps in memmaps.values():
        for memmap_array in split_memmaps.values():
            memmap_array.flush()

    if train_offset != train_rows:
        raise ValueError(f"Expected {train_rows} train rows but wrote {train_offset}")
    valid_rows = int(memmaps["valid"]["labels"].shape[0])
    if valid_offset != valid_rows:
        raise ValueError(f"Expected {valid_rows} valid rows but wrote {valid_offset}")

    return {
        "train": {
            "num_rows": train_offset,
            "positive_count": train_positive,
            "negative_count": train_offset - train_positive,
            "positive_ratio": float(train_positive / max(train_offset, 1)),
        },
        "valid": {
            "num_rows": valid_offset,
            "positive_count": valid_positive,
            "negative_count": valid_offset - valid_positive,
            "positive_ratio": float(valid_positive / max(valid_offset, 1)),
        },
    }


def build_feature_schema(
    config: dict[str, Any],
    resolved_paths: dict[str, Path],
    sparse_cardinalities: dict[str, int],
    total_rows: int,
    split_stats: dict[str, Any],
    sparse_encoding: str,
) -> dict[str, Any]:
    memmap_paths = expected_memmap_paths(resolved_paths["feature_dir"] / MEMMAP_DIRNAME)
    return {
        "label_column": "label",
        "dense_features": DENSE_COLS,
        "sparse_features": SPARSE_COLS,
        "sparse_cardinalities": sparse_cardinalities,
        "num_rows": int(total_rows),
        "backend": "memmap",
        "sparse_encoding": sparse_encoding,
        "training": {
            "embed_dim": int(config["training"]["embed_dim"]),
            "hidden_units": list(config["training"]["hidden_units"]),
            "dropout": float(config["training"].get("dropout", 0.0)),
        },
        "paths": {
            "raw_data": str(resolved_paths["raw_data"]),
            "feature_dir": str(resolved_paths["feature_dir"]),
            "checkpoint_dir": str(resolved_paths["checkpoint_dir"]),
            "log_dir": str(resolved_paths["log_dir"]),
            "train_dense": str(memmap_paths["train"]["dense"]),
            "train_sparse": str(memmap_paths["train"]["sparse"]),
            "train_labels": str(memmap_paths["train"]["labels"]),
            "valid_dense": str(memmap_paths["valid"]["dense"]),
            "valid_sparse": str(memmap_paths["valid"]["sparse"]),
            "valid_labels": str(memmap_paths["valid"]["labels"]),
        },
        "stats": split_stats,
    }


def resolve_criteo_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Path]:
    paths = config["paths"]
    return {
        "raw_data": resolve_workspace_path(base_dir, paths["raw_data"]),
        "feature_dir": resolve_workspace_path(base_dir, paths["feature_dir"]),
        "log_dir": resolve_workspace_path(base_dir, paths["log_dir"]),
        "checkpoint_dir": resolve_workspace_path(base_dir, paths["checkpoint_dir"]),
    }


@dataclass
class CriteoMemmapSplit:
    name: str
    dense_path: Path
    sparse_path: Path
    label_path: Path
    num_rows: int

    def __post_init__(self) -> None:
        self._dense: np.memmap | None = None
        self._sparse: np.memmap | None = None
        self._labels: np.memmap | None = None

    @property
    def dense(self) -> np.memmap:
        if self._dense is None:
            self._dense = np.load(self.dense_path, mmap_mode="r")
        return self._dense

    @property
    def sparse(self) -> np.memmap:
        if self._sparse is None:
            self._sparse = np.load(self.sparse_path, mmap_mode="r")
        return self._sparse

    @property
    def labels(self) -> np.memmap:
        if self._labels is None:
            self._labels = np.load(self.label_path, mmap_mode="r")
        return self._labels

    def get_batch(self, index: slice | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dense = np.array(self.dense[index], dtype=np.float32, copy=True)
        sparse = np.array(self.sparse[index], dtype=np.int64, copy=True)
        labels = np.array(self.labels[index], dtype=np.float32, copy=True)
        return dense, sparse, labels

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        shuffle_block_rows: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.num_rows == 0:
            return

        if not shuffle:
            for start in range(0, self.num_rows, batch_size):
                stop = min(start + batch_size, self.num_rows)
                yield self.get_batch(slice(start, stop))
            return

        block_rows = max(batch_size, shuffle_block_rows)
        block_starts = np.arange(0, self.num_rows, block_rows, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(block_starts)

        for block_start in block_starts:
            block_end = min(int(block_start + block_rows), self.num_rows)
            indices = np.arange(block_start, block_end, dtype=np.int64)
            rng.shuffle(indices)
            for offset in range(0, len(indices), batch_size):
                batch_indices = indices[offset: offset + batch_size]
                yield self.get_batch(batch_indices)


def load_prepared_dataset(feature_dir: Path) -> dict[str, Any]:
    schema = load_yaml(feature_dir / "feature_schema.yaml")
    stats = load_json(feature_dir / DATA_STATS_NAME)
    paths = schema["paths"]
    datasets = {
        "train": CriteoMemmapSplit(
            name="train",
            dense_path=Path(paths["train_dense"]),
            sparse_path=Path(paths["train_sparse"]),
            label_path=Path(paths["train_labels"]),
            num_rows=int(stats["train"]["num_rows"]),
        ),
        "valid": CriteoMemmapSplit(
            name="valid",
            dense_path=Path(paths["valid_dense"]),
            sparse_path=Path(paths["valid_sparse"]),
            label_path=Path(paths["valid_labels"]),
            num_rows=int(stats["valid"]["num_rows"]),
        ),
    }
    return {"schema": schema, "datasets": datasets, "stats": stats}


def prepare_criteo_dataset(
    config: dict[str, Any],
    base_dir: Path,
    *,
    force_prepare: bool = False,
) -> dict[str, Any]:
    data_cfg = config["data"]
    chunk_size = int(data_cfg.get("chunk_size", 250_000))
    sparse_encoding = str(data_cfg.get("sparse_encoding", "hash")).lower()
    if sparse_encoding not in {"hash", "vocab"}:
        raise ValueError(f"Unsupported sparse_encoding={sparse_encoding!r}; expected 'hash' or 'vocab'")

    train_ratio = float(data_cfg["train_ratio"])
    valid_ratio = float(data_cfg["valid_ratio"])
    validate_split_config(train_ratio, valid_ratio)

    resolved_paths = resolve_criteo_paths(config, base_dir)
    raw_path = resolved_paths["raw_data"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Criteo raw data not found: {raw_path}\n"
            "Use --raw-data to point train_criteo.py at the full train.txt file."
        )

    feature_dir = resolved_paths["feature_dir"]
    feature_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths["log_dir"].mkdir(parents=True, exist_ok=True)
    resolved_paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    signature = build_prepare_signature(config, resolved_paths)
    if not force_prepare and bool(data_cfg.get("reuse_cache", True)) and cache_is_valid(feature_dir, signature):
        print(f"Reusing prepared memmap cache from {feature_dir}")
        return load_prepared_dataset(feature_dir)

    num_samples = data_cfg.get("num_samples")
    total_rows = count_criteo_rows(raw_path, num_samples)
    if total_rows < 2:
        raise ValueError(f"Criteo dataset must contain at least 2 rows, got {total_rows}")

    train_rows = max(1, int(total_rows * train_ratio))
    train_rows = min(train_rows, total_rows - 1)
    valid_rows = total_rows - train_rows

    print(f"Preparing Criteo dataset from {raw_path}")
    print(f"Rows: total={total_rows:,} train={train_rows:,} valid={valid_rows:,} chunk_size={chunk_size:,}")
    print("Pass 1/2: fitting dense feature statistics...")
    dense_stats = fit_dense_stats_streaming(raw_path, num_samples, chunk_size, train_rows)
    encoder_dir = feature_dir / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    dump_json(encoder_dir / "dense_stats.json", dense_stats)

    sparse_vocabs: dict[str, dict[str, int]] | None = None
    if sparse_encoding == "hash":
        sparse_cardinalities = build_hash_cardinalities(data_cfg)
        dump_json(encoder_dir / "sparse_buckets.json", sparse_cardinalities)
    else:
        threshold = int(data_cfg.get("sparse_threshold", 20))
        max_vocab_size = data_cfg.get("max_vocab_size")
        max_vocab_size = int(max_vocab_size) if max_vocab_size is not None else None
        print("Building sparse vocabularies. This mode is intended for sampled data, not full Criteo.")
        sparse_vocabs = build_vocabs_streaming(
            raw_path,
            num_samples,
            chunk_size,
            train_rows,
            threshold,
            max_vocab_size,
        )
        sparse_cardinalities = {col: len(vocab) for col, vocab in sparse_vocabs.items()}
        dump_json(encoder_dir / "sparse_vocabs.json", sparse_vocabs)

    print("Pass 2/2: encoding features and writing memmap splits...")
    memmaps = create_memmaps(feature_dir / MEMMAP_DIRNAME, train_rows, valid_rows)
    split_stats = populate_memmaps(
        raw_path,
        num_samples,
        chunk_size,
        train_rows,
        dense_stats,
        sparse_encoding,
        sparse_cardinalities,
        sparse_vocabs,
        memmaps,
    )

    stats = {
        "num_rows": int(total_rows),
        "backend": "memmap",
        "sparse_encoding": sparse_encoding,
        "num_dense_features": NUM_DENSE,
        "num_sparse_features": NUM_SPARSE,
        "train": split_stats["train"],
        "valid": split_stats["valid"],
    }
    schema = build_feature_schema(
        config=config,
        resolved_paths=resolved_paths,
        sparse_cardinalities=sparse_cardinalities,
        total_rows=total_rows,
        split_stats=split_stats,
        sparse_encoding=sparse_encoding,
    )

    dump_yaml(feature_dir / "feature_schema.yaml", schema)
    dump_json(feature_dir / DATA_STATS_NAME, stats)
    dump_json(feature_dir / PREPARE_CONFIG_NAME, signature)
    print(
        f"Prepared cache at {feature_dir} "
        f"(train_pos_ratio={split_stats['train']['positive_ratio']:.4f}, "
        f"valid_pos_ratio={split_stats['valid']['positive_ratio']:.4f})"
    )
    return load_prepared_dataset(feature_dir)


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return compute_classification_metrics(labels, probs)

# Project

## Goal

Maintain a small but complete DeepFM training framework that can prepare data, train, evaluate, and track results for both TAAC2026 and Criteo. The primary comparison metric is validation AUC.

## Baseline

Trusted Criteo baseline:

- Config: `configs/base/criteo.yaml`
- Checkpoint: `local_workspace/checkpoints/criteo/deepfm/best_model.pt`
- Metric: validation AUC `0.801568`, logloss `0.452217`, accuracy `0.7895`
- Date: April 10, 2026

TAAC2026 remains the secondary task and uses the same framework structure with its own config and feature pipeline.

## Data And Metrics

- Criteo raw data: `/media/shenyu/data/archive/dac/train.txt`
- TAAC2026 raw data: configured via `configs/base/taac2026.yaml`
- Criteo split: positional `90/10` train/valid
- TAAC2026 split: timestamp-based `70/30` train/valid
- Primary metric: validation AUC
- Supporting metrics: logloss and accuracy
- Default seed: `2026`

## Constraints

- Main compute target is a single CUDA GPU.
- TAAC2026 config can require a specific device keyword.
- Full Criteo runs must avoid loading the entire dataset into RAM; memmap caching is the trusted path.
- Generated outputs belong in `local_workspace/` and should stay out of git.

## Repo Structure

- `configs/`: tracked base, experiment, and debug configs
- `src/deepfm/data/`: dataset loading and preprocessing
- `src/deepfm/models/`: DeepFM layers and model definition
- `src/deepfm/train/`: task-specific training and evaluation pipelines
- `src/deepfm/eval/`: shared metrics
- `src/deepfm/common/`: shared config and runtime helpers
- `scripts/`: reproducible prepare, train, evaluate, and smoke-test entrypoints
- `docs/`: stable project context and short work log
- `local_workspace/`: ignored training artifacts and caches

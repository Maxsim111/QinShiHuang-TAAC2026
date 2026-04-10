# QinShiHuang-TAAC2026

Docs-first training repo for DeepFM experiments on TAAC2026 and Criteo.

Source of truth for project context: `docs/project.md`
Source of truth for work log: `docs/log.md`

## Layout

- `configs/`: tracked base, experiment, and debug configs
- `src/`: DeepFM data, model, training, and evaluation code
- `scripts/`: reproducible entrypoints for prepare, train, evaluate, and smoke tests
- `docs/`: stable project context and short work log
- `checks/`: lightweight guidance checks
- `DeepInterestNetwork/`: legacy reference code kept as archive

## Quick Start

```bash
conda env create -f environment.taac2026-torch.yml
conda activate taac2026-torch
python scripts/train.py --config configs/base/criteo.yaml
python scripts/evaluate.py --config configs/base/criteo.yaml
```

# Log

## Entries

- Date: 2026-04-10
- Task: Added epoch-level training artifact logging and moved the old TensorFlow reference repo under `legacy/`.
- Files: `src/deepfm/common/reporting.py`, `src/deepfm/train/`, `legacy/`, `README.md`, `docs/project.md`
- Verify: `python -m py_compile ...`, `python scripts/smoke_test.py`, `python scripts/evaluate.py --config configs/debug/criteo_smoke.yaml --checkpoint ...`
- Result: Training now writes structured history and summary artifacts, evaluation writes JSON reports, and archived code is explicitly separated from the active framework.
- Next: Add richer comparison/report tooling if experiment volume grows.

- Date: 2026-04-10
- Task: Integrated the harness scaffold into the repo and migrated DeepFM into a docs-first training framework.
- Files: `README.md`, `AGENTS.md`, `docs/`, `configs/`, `scripts/`, `src/deepfm/`, `.gitignore`
- Verify: `python -m py_compile ...`, harness top-level check, config-driven smoke training, config-driven evaluation
- Result: The project now uses tracked configs, `src/` package code, reproducible entrypoints, and a documented baseline.
- Next: Iterate on experiment configs and add task-specific reports if needed.

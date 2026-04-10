# src/

Recommended split:

- `deepfm/data/`: dataset loading and preprocessing
- `deepfm/models/`: model definitions
- `deepfm/train/`: task-specific training and evaluation orchestration
- `deepfm/eval/`: metrics and reporting helpers
- `deepfm/common/`: shared config and runtime utilities

Rules:

- Keep entrypoints in `scripts/`
- Keep configs out of code
- Promote helpers into `common/` only when they are reused

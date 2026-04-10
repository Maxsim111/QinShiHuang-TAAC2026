# scripts/

Keep only reproducible entrypoints here.

## Commands

- `train.py`: launches training from a tracked config
- `prepare_data.py`: prepares cached features without training
- `evaluate.py`: evaluates a checkpoint on the validation split
- `smoke_test.py`: cheap end-to-end check using a debug config

## Recorded Outputs

- `train.py` writes epoch history to `train_history.jsonl` and `train_history.csv`
- `train.py` writes run summaries to `train_summary.json`
- `evaluate.py` writes structured reports under `evaluations/`

## Rules

- One script should do one purpose well
- Accept config paths explicitly
- Print a rerun command near the start of execution

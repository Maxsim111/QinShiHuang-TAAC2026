# scripts/

Keep only reproducible entrypoints here.

## Commands

- `train.py`: launches training from a tracked config
- `prepare_data.py`: prepares cached features without training
- `evaluate.py`: evaluates a checkpoint on the validation split
- `smoke_test.py`: cheap end-to-end check using a debug config

## Rules

- One script should do one purpose well
- Accept config paths explicitly
- Print a rerun command near the start of execution

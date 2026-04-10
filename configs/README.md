# configs/

Use this directory for tracked configuration only.

## Split

- `base/`: stable configs for supported tasks
- `experiments/`: small overrides for one idea
- `debug/`: cheap smoke-test configs

## Rules

- Prefer layered configs over copied full files.
- Keep `task` explicit so entrypoints can dispatch cleanly.
- Record trusted config paths and outcomes in `docs/project.md` or `docs/log.md`.

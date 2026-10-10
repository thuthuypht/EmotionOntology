# Reproducibility notes

The camera-ready run used:

- Model: `vinai/phobert-base-v2`
- Max sequence length: 128
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 2e-5
- Weight decay: 0.01
- Warm-up ratio: 0.10
- Max epochs: 5
- Early-stopping patience: 2
- Seeds: 1, 2, 3, 4, 5
- Final evaluation file: `data/benchmark/synthetic_val.csv`

The exact run configuration is also preserved in:
`results/camera_ready/run_config.json`.

Primary summary tables:
- `results/camera_ready/all_seed_metrics.csv`
- `results/camera_ready/summary_mean_std.csv`
- `results/camera_ready/paper_ready_table.csv`

Per-seed predictions, training history, and confusion matrices are under:
`results/camera_ready/seeds/`.

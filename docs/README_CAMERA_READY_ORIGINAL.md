# CSoNet 2026 Camera-Ready Experiments

This package was prepared for the manuscript **Ontology-Driven Sentiment Analytics for Teaching Outcome Assessment from Vietnamese Open-Ended Student Feedback**.

## What the experiment does

The script uses the supplied `synthetic_train.csv` and `synthetic_val.csv` **without replacing them with a different benchmark split**. The final validation set is held fixed. For each seed `1, 2, 3, 4, 5`, `synthetic_train.csv` is split stratifiably into 80% model-training and 20% internal-development data.

It runs two experiment families:

1. **Lightweight replication (fast, CPU-compatible)**
   - `Lightweight-Text`
   - `Lightweight+Topic` — explicit text + topic-metadata ablation without ontology lexical cues
   - `Lightweight+Ontology` — current 33-D ontology lexical + topic metadata conservative integration

2. **Strong Vietnamese PLM comparison (GPU recommended)**
   - `PhoBERT-Text`
   - `PhoBERT+Topic` — same topic-metadata ablation
   - `PhoBERT+Ontology` — conservative second-stage integration using the 33-D ontology feature representation

Only **one PhoBERT encoder is fine-tuned per seed**. The Topic and Ontology variants use the trained PhoBERT decision scores plus lightweight conservative meta-classifiers, so the full five-seed experiment is much cheaper than fine-tuning three PhoBERT models per seed.

## Default PhoBERT protocol

- Model: `vinai/phobert-base-v2`
- Maximum sequence length: 128
- Head: dropout 0.30 + linear 768→3
- Optimizer: AdamW
- Learning rate: 2e-5
- Batch size: 16
- Maximum epochs: 5
- Weight decay: 0.01
- Linear warm-up ratio: 0.10
- Gradient clipping: 1.0
- Checkpoint selection: best internal-development Macro-F1
- Early stopping patience: 2
- Seeds: 1, 2, 3, 4, 5
- Final evaluation: fixed `synthetic_val.csv`

These defaults are explicit so they can be reported directly in the camera-ready paper. They are configurable from the command line.

## Recommended Kaggle workflow

1. Upload `EmotionOntology_CameraReady_Experiments.zip` as a private Kaggle Dataset.
2. Open `CameraReady_CSoNet_PhoBERT_5Seeds.ipynb` in a Kaggle GPU notebook (T4/P100 or better).
3. Enable Internet for the first Hugging Face download if `vinai/phobert-base-v2` is not already cached.
4. Run the preflight cell.
5. Run the optional one-seed smoke test.
6. Run the full five-seed cell.
7. Download `camera_ready_outputs.zip`.

The full run is **seed-resumable**: after each completed seed, `outputs/seeds/seed_<n>/completed_metrics.json` is written. Re-running the command skips completed seeds unless `--force` is passed.

## Command-line use

Preflight only:

```bash
python camera_ready_experiments.py \
  --input_dir input \
  --output_dir outputs \
  --preflight_only
```

One-seed smoke test:

```bash
python camera_ready_experiments.py \
  --input_dir input \
  --output_dir outputs_smoke \
  --run_phobert \
  --seeds 1 \
  --epochs 1 \
  --max_train_samples 900 \
  --max_dev_samples 300
```

Full camera-ready run:

```bash
python camera_ready_experiments.py \
  --input_dir input \
  --output_dir outputs \
  --run_phobert \
  --seeds 1 2 3 4 5 \
  --epochs 5 \
  --batch_size 16
```

If GPU memory is limited, reduce `--batch_size` to 8. Do **not** change the data partitions or seeds after some runs are completed unless all runs are repeated consistently.

## Main outputs

- `all_seed_metrics.csv` — one row per seed/model
- `summary_mean_std.csv` — numeric mean and sample standard deviation across seeds
- `paper_ready_table.csv` — formatted `mean ± std` table
- `dataset_statistics.csv` — exact split / class / topic counts
- `ontology_kg_statistics.csv` — statistics from the supplied V8 ontology **with a schema-alignment caution**
- `human_annotation_iaa.csv` — 200-sample two-annotator agreement summary
- `sparql_representative_results.csv` — de-identified representative outputs from the supplied SPARQL workbook
- `camera_ready_replacement_notes.md` — exact prose/numbers to use when replacing the red camera-ready notes
- `run_config.json` — full reproducibility configuration
- `seeds/seed_*/phobert_training_history.csv`
- `seeds/seed_*/phobert_predictions.csv`
- per-model confusion matrices

## Important scientific cautions

1. The supplied synthetic CSVs establish exact dataset sizes and class distributions, but **do not themselves document their exact derivation from the official UIT-VSFC release**. Keep the wording *UIT-VSFC-style/derived synthetic benchmark* unless provenance is independently documented.
2. `Ontology_NTU_FeedbackSystem_V8.owl` has a schema that differs from the ontology terminology currently used in the CSoNet manuscript. Its counts are exported, but should not be inserted into the paper until the schema is reconciled.
3. The 200-sample IAA workbook supports annotation reliability for Sentiment, Emotion, and Topic. It does not contain adjudicated gold labels for TeachingCriterion, perceived LearningOutcome, or ImprovementSuggestion; those reviewer requests remain pending.
4. The ontology lexical feature builder learns additional discriminative lexical cues **from the seed-specific training partition only**, never from `synthetic_val.csv`, to prevent test leakage.

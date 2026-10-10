# EmotionOntology-CSoNet2026

Reproducibility package for the AI-TECH Workshop at CSoNet 2026 paper:

**Ontology-Driven Sentiment Analytics for Teaching Outcome Assessment from Vietnamese Open-Ended Student Feedback**

Authors: Pham Thi Thu Thuy, Bui Chi Thanh, and Huynh Minh Duc  
Affiliation: Nha Trang University, Vietnam

Expected repository URL: https://github.com/thuthuypht/EmotionOntology-CSoNet2026

## What is included

- Final camera-ready experiment code for the five-seed benchmark.
- Kaggle-ready notebook used for the PhoBERT experiments.
- Exact synthetic benchmark splits used in the reported camera-ready experiments.
- Public ontology schema/lexicalization export derived from the final EONLP_V10 ontology.
- Five-seed predictions, confusion matrices, training histories, and summary tables.
- De-identified SPARQL examples and human-annotation agreement summaries.
- Legacy scripts/notebooks retained for provenance and reproducibility.

## Main camera-ready result

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Lightweight-Text | 0.8245 ± 0.0038 | 0.8236 ± 0.0036 |
| Lightweight+Topic | 0.8275 ± 0.0035 | 0.8263 ± 0.0035 |
| Lightweight+Ontology | 0.8297 ± 0.0031 | 0.8287 ± 0.0030 |
| PhoBERT-Text | 0.8781 ± 0.0032 | 0.8777 ± 0.0033 |
| PhoBERT+Topic | 0.8789 ± 0.0036 | 0.8784 ± 0.0037 |
| PhoBERT+Ontology | 0.8784 ± 0.0029 | 0.8779 ± 0.0030 |

## Quick reproduction

```bash
pip install -r requirements.txt
python code/camera_ready/camera_ready_experiments.py \
  --input_dir data/benchmark \
  --output_dir results/reproduction \
  --run_phobert \
  --seeds 1 2 3 4 5 \
  --epochs 5 \
  --batch_size 16 \
  --num_workers 0
```

For Kaggle, use `notebooks/CameraReady_CSoNet_PhoBERT_5Seeds_v3.ipynb`.

## Privacy and data availability

The repository intentionally **does not include identifiable NTU institutional survey files or the fully populated institutional KG** because the source files contain lecturer names/codes and free-text comments. The public bundle includes the benchmark data needed to reproduce the reported five-seed predictive experiments, a public ontology schema, de-identified SPARQL examples, and aggregate annotation-reliability results.

See `docs/DATA_AVAILABILITY.md` for details.

## Repository structure

- `code/` — experiment and institutional pipeline source code
- `notebooks/` — Kaggle camera-ready notebook
- `data/benchmark/` — exact benchmark splits used for the reported experiments
- `ontology/` — public schema/lexicalization resources
- `results/` — full five-seed camera-ready outputs and legacy result tables
- `figures/` — publication figure(s)
- `docs/` — reproducibility, privacy, and upload notes

## Citation

If you use this repository, please cite the CSoNet 2026 paper. A machine-readable citation template is provided in `CITATION.cff`.

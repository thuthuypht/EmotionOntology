# Camera-ready replacement notes

## 1. DATA CHECK — ready to replace

The synthetic benchmark contains 10,180 instances: 8,144 in synthetic_train.csv and 2,036 in synthetic_val.csv. The training split contains 2,711 Negative, 2,724 Neutral, and 2,709 Positive instances; the validation split contains 686 Negative, 670 Neutral, and 680 Positive instances. The final validation split is held fixed across runs, while each seed creates a stratified 80/20 model-training/internal-development split from synthetic_train.csv.

**Caution:** The supplied CSV files do not by themselves document the exact derivation procedure from the official UIT-VSFC release. Keep the wording 'UIT-VSFC-style/derived synthetic benchmark' unless that provenance is independently documented.

## 2. STRONG PHOBERT BASELINE — replace after the five-seed run completes

PhoBERT configuration: model=vinai/phobert-base-v2; tokenizer=AutoTokenizer for the same checkpoint; max sequence length=128; classification head=dropout(0.30)+linear(768→3); optimizer=AdamW; learning rate=2e-05; batch size=16; maximum epochs=5; weight decay=0.01; warm-up ratio=0.1; checkpoint selection=best internal-development Macro-F1 with early stopping patience=2; seeds=[1, 2, 3, 4, 5].

Five-seed summary:

- PhoBERT+Ontology: Accuracy 0.8784 ± 0.0029; Macro-Precision 0.8783 ± 0.0032; Macro-Recall 0.8778 ± 0.0029; Macro-F1 0.8779 ± 0.0030.

- PhoBERT+Topic: Accuracy 0.8789 ± 0.0036; Macro-Precision 0.8786 ± 0.0038; Macro-Recall 0.8783 ± 0.0036; Macro-F1 0.8784 ± 0.0037.

- PhoBERT-Text: Accuracy 0.8781 ± 0.0032; Macro-Precision 0.8781 ± 0.0035; Macro-Recall 0.8775 ± 0.0032; Macro-F1 0.8777 ± 0.0033.

## 3. RESULT TABLE — automatically generated

Use `paper_ready_table.csv` for the numerical table required by the reviewer. `all_seed_metrics.csv` contains every seed-level result and `summary_mean_std.csv` contains numeric means and standard deviations.

## 4. ONTOLOGY/KG STATISTICS — conditional

The supplied Ontology_NTU_FeedbackSystem_V8.owl contains 15 named classes, 15 object properties, 15 datatype properties, 2198 named individuals, 27711 RDF triples, and 0 skos:altLabel assertions.

**Do not insert these counts into the paper until the V8 ontology schema is reconciled with the manuscript's StudentFeedbackForm/FeedbackEntry/TeachingCriterion/LearningOutcome schema.**

## 5. HUMAN-ANNOTATION EVIDENCE — partially ready

- Sentiment: n=200, observed agreement=92.0%, Cohen's κ=0.8198.

- Emotion: n=200, observed agreement=85.0%, Cohen's κ=0.7752.

- Topic: n=200, observed agreement=90.0%, Cohen's κ=0.8262.

These data support annotation reliability for Sentiment/Emotion/Topic, but the supplied file does not contain adjudicated gold labels for TeachingCriterion, perceived LearningOutcome, or ImprovementSuggestion. Keep that part of the reviewer warning until those dimensions are annotated and evaluated.

## 6. SPARQL RESULTS — ready as practical evidence after schema consistency check

Representative de-identified rows are exported to `sparql_representative_results.csv`. They can support a compact paper table once the queried KG is confirmed to be the same final KG described in the manuscript.

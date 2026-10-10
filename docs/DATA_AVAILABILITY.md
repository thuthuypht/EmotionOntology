# Data availability and privacy

## Publicly included
The public repository contains the exact `synthetic_train.csv` and `synthetic_val.csv`
splits used in the camera-ready five-seed experiments, together with derived benchmark
variants, aggregate annotation-agreement results, experiment outputs, and a public
ontology schema/lexicalization export.

## Withheld from the public GitHub repository
The following institutional artifacts are deliberately excluded from this public bundle:

- `Khoa CNTT Gop y.xlsx` / `Khoa CNTT - Gop y.xlsx`
- `NTU_feedback_updated_v2.csv`
- processed institutional Excel workbooks containing original comments
- `Ontology_NTU_FeedbackSystem_V8.owl`
- the fully populated `EONLP_V10_skos_enriched_final.owl`
- raw SPARQL outputs containing original institutional comments

Reason: these files contain lecturer names/codes and/or free-text institutional feedback.
They should not be placed in a public repository without explicit institutional approval
and a documented de-identification/data-sharing process.

## What the public ontology file contains
`ontology/EONLP_V10_public_schema.owl` preserves the non-sensitive schema and lexical
resources required to understand the model/ontology integration, but removes populated
record/course/lecturer/question individuals and raw comment-bearing assertions.

## Reproducibility scope
All predictive performance reported in the camera-ready five-seed benchmark can be
reproduced from the public benchmark splits and code in this repository. Institutional
SPARQL examples are released only in de-identified form.

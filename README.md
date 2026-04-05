# Speech Emotion Recognition

Research-grade Speech Emotion Recognition (SER) pipeline with:
- Dataset preprocessing and label unification
- MFCC baseline and wav2vec2 representations
- In-domain and cross-domain evaluation
- Pooling study (`mean`, `max`, `mean_std`)
- Reproducible split export and automatic result logging

## Datasets

This repository excludes raw data. Place datasets locally in:
- `Radvess/`
- `Crema D/`

## Core Scripts

- `ravdess_preprocessing.py`
  - Audio preprocessing (mono, 16kHz, normalization)
  - RAVDESS + CREMA-D label/speaker parsing
- `wav2vec2_cross_dataset_eval.py`
  - wav2vec2 cached pipeline
  - MFCC baseline pipeline
  - In-domain and cross-domain experiments
  - Pooling study
  - Result export (`results/`)
- `baseline_ser_mfcc.py`, `cross_dataset_eval.py`
  - Legacy/standalone MFCC baselines

## Environment

```bash
python -m pip install -U numpy scipy scikit-learn pandas soundfile librosa torch transformers
```

## Run Experiments

Full run:
```bash
python wav2vec2_cross_dataset_eval.py --run_mode full
```

Optional quick check:
```bash
python wav2vec2_cross_dataset_eval.py --run_mode debug
```

## Output Artifacts

### Feature cache
- `features/ravdess/*.npy`
- `features/crema_d/*.npy`

### Experiment results
Stored under `results/`:
- `results.json` (append-style detailed records)
- `results.csv` (append-style tabular records)
- `split_info.json` (speaker split + seed)
- `cm_{feature}_{pooling}_{eval_type}.csv` (confusion matrices)
- `final_comparison.csv` (summary rows for report tables)

## Latest Full-Run Highlights

Speaker split (seed=42):
- Train speakers: 19
- Test speakers: 5
- Speaker overlap: 0

### wav2vec2

| Pooling | In-Domain Acc | In-Domain F1 | Cross-Domain Acc | Cross-Domain F1 |
|---|---:|---:|---:|---:|
| `mean` | 0.5538 | 0.5569 | 0.2776 | 0.2658 |
| `max` | 0.5885 | 0.5848 | 0.2368 | 0.2244 |
| `mean_std` | 0.5769 | 0.5755 | 0.2549 | 0.2530 |

### MFCC

| Pooling | In-Domain Acc | In-Domain F1 | Cross-Domain Acc | Cross-Domain F1 |
|---|---:|---:|---:|---:|
| `mean` | 0.3731 | 0.3470 | 0.2295 | 0.1316 |
| `max` | 0.3769 | 0.3587 | 0.2741 | 0.2065 |
| `mean_std` | 0.4500 | 0.4368 | 0.2106 | 0.1547 |

Best in-domain Macro-F1:
- wav2vec2 + `max`: **0.5848**

Best cross-domain Macro-F1:
- wav2vec2 + `mean`: **0.2658**

Summary:
- Better in-domain model: `wav2vec2`
- Better cross-domain model: `wav2vec2`
- Best pooling overall: `mean_std`

## Notes

- wav2vec2 caching makes reruns much faster by avoiding recomputation.
- Logistic Regression uses `class_weight="balanced"` for class imbalance handling.
- Common cross-dataset label set:
  - `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`

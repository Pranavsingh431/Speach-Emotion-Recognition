# Speech Emotion Recognition

Research-oriented Speech Emotion Recognition (SER) pipeline using:
- RAVDESS and CREMA-D datasets
- MFCC baseline
- wav2vec2 (`facebook/wav2vec2-base`) embeddings
- In-domain and cross-domain evaluation

## Project Structure

- `ravdess_preprocessing.py` - dataset loading, label mapping, preprocessing
- `baseline_ser_mfcc.py` - MFCC baseline experiments
- `cross_dataset_eval.py` - MFCC cross-dataset evaluation
- `wav2vec2_cross_dataset_eval.py` - wav2vec2 pipeline with caching, pooling study, in-domain + cross-domain evaluation

## Datasets

This repository does **not** include raw datasets.

Expected local folders:
- `Radvess/`
- `Crema D/`

## Setup

```bash
python -m pip install -U numpy scipy scikit-learn pandas soundfile librosa torch transformers
```

## Run

```bash
python wav2vec2_cross_dataset_eval.py
```

## Notes

- wav2vec2 embeddings are cached in `features/` to support resumable runs.
- Pooling strategies supported: `mean`, `max`, `mean_std`.

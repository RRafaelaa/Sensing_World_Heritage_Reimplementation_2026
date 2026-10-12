# BERTopic random-seed stability data

This directory contains the outputs of the random-seed stability experiment for the selected BERTopic configuration. The experiment covers 464 documents. Seed 42 is the reference run; seeds 0–9 are ten additional runs. The document embeddings and all model settings were held fixed across runs, with only the UMAP `random_state` changed.

## Files

| File | Contents |
| --- | --- |
| `seed_runs.csv` | Metrics for the reference run and each additional seed. |
| `doc_assignments.csv` | Raw and reference-matched topic assignments for each document and seed. The `Key` column can be joined to `../meta_data.csv`. |
| `topic_matching.csv` | Matches between each run's topics and the reference topics, including document-set overlap. |
| `topic_stability_summary.csv` | Stability measures summarized for each of the nine reference topics. |

## Fixed configuration

| Component | Setting |
| --- | --- |
| BERTopic topic count | `auto` |
| UMAP neighbors | 10 |
| UMAP components | 8 |
| UMAP minimum distance | 0.1 |
| HDBSCAN minimum cluster size | 10 |
| HDBSCAN minimum samples | 5 |
| HDBSCAN cluster selection | `eom` |
| Outlier reduction | Enabled; `distributions` strategy, threshold 0.1 |
| Embeddings | Generated fresh for the experiment and held fixed across seed runs |

The topic-keyword metrics in `seed_runs.csv` use 1–3-word n-grams, KeyBERTInspired followed by MMR, 20 terms per topic, and stem-based deduplication.

## Recorded software environment

| Package | Version |
| --- | --- |
| Python | 3.10.13 |
| BERTopic | 0.17.4 |
| umap-learn | 0.5.9.post2 |
| hdbscan | 0.8.41 |
| scikit-learn | 1.7.2 |
| NumPy | 1.26.4 |
| sentence-transformers | 5.2.0 |
| PyTorch | 2.9.1 |
| gensim | 4.4.0 |
| numba | 0.63.1 |

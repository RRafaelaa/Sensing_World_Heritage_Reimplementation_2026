# Sensing World Heritage: reimplementation materials

This release candidate reorganizes the reusable materials behind **“Sensing World Heritage with Social Media and Artificial Intelligence: A Systematic Literature Review.”** It was assembled from the manuscript folder and the two working folders `Extraction_Project` and `bertopic`.

The package is intentionally smaller than the working directories. It keeps code, prompts, structured study-level data, derived inputs, and a small set of reference outputs needed to reimplement the computational workflow. It excludes third-party PDFs and extracted full text, model weights, caches, duplicate archives, exploratory backups, failed API responses, IDE files, and author photographs.

## What can be reimplemented

1. PDF-to-text conversion with PDFPlumber, PyMuPDF, and OCR fallback.
2. Schema-constrained Gemini extraction, including a baseline pass and a targeted higher-capability pass.
3. JSON validation, merge, and conversion to flat study-level tables.
4. Place extraction, Nominatim geocoding, UNESCO/Tentative List name matching, and paper-level aggregation.
5. BERTopic modelling using `all-MiniLM-L6-v2`, UMAP, HDBSCAN, c-TF-IDF, KeyBERTInspired, and MMR.
6. The 192-configuration search reported in the manuscript and its weighted comparison.
7. Term normalization/binarization and method-theme network analysis through the retained notebooks.

The PRISMA search, database export, ASReview screening decisions, manual World Heritage corrections, ArcGIS map styling, Gephi layout decisions, and expert topic naming are not fully automated by the available files. They remain documented human or external-tool steps.

## Repository layout

```text
config/extraction/        schemas and extraction instructions
src/extraction/           PDF conversion and LLM extraction pipeline
src/geography/            place/geocoding/UNESCO matching pipeline
src/topic_modeling/       BERTopic run, grid search, ranking, visualization
src/utilities/            release-hygiene utilities
analysis/notebooks/       stripped notebooks for term and network analysis
data/                     selected study-level and derived research data
reference_outputs/        compact expected outputs for cross-checking
```

See `MANIFEST.md` for provenance and `OPEN_SOURCE_REVIEW.md` before publishing.

## Environment

Python 3.10 or 3.11 is recommended. Create separate environments if dependency resolution becomes difficult.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-extraction.txt
pip install -r requirements-analysis.txt
```

OCR additionally requires system installations of Tesseract and Poppler. The SentenceTransformer model downloads from Hugging Face by default; pass `--model_path /path/to/all-MiniLM-L6-v2` for offline use.

## 1. PDF to structured extraction

The papers themselves are not redistributed. Place lawfully obtained PDFs in `data/pdfs/`; this path is ignored by Git.

```bash
python src/extraction/extract_pdf_text.py
```

This writes plain text to `data/texts/`. Run a small API test first:

```bash
export GOOGLE_API_KEY='your-key'
python src/extraction/heritage_extractor_gemini.py \
  --input data/texts \
  --output data/outputs/gemini_flash \
  --schema config/extraction/schema_full.json \
  --instructions config/extraction/instructions_full.txt \
  --model gemini-2.0-flash-exp \
  --max-papers 5
```

For the targeted second pass, use `schema_targeted.json`, `instructions_targeted.txt`, and the historically reported `gemini-2.5-pro` model. Model IDs and the legacy `google-generativeai` SDK may no longer be available; substituting a current model is a reimplementation, not an exact replication.

Merge passes and flatten results:

```bash
python src/extraction/merge_extractions.py \
  --flash data/outputs/gemini_flash \
  --gemini data/outputs/gemini_targeted \
  --output data/outputs/merged

python src/extraction/validate_extractions.py --output data/outputs/merged
python src/extraction/json_to_csv.py \
  --input data/outputs/merged \
  --output outputs/extracted_tables
```

The two supplied reviewer workbooks are not redistributed because they contain
local attachment paths. Their 32 cell-level differences are preserved in
`data/extractions/manual_corrections.csv`. To regenerate that change set from
authorized copies:

```bash
python src/extraction/export_manual_corrections.py \
  before_review.xlsx after_review.xlsx \
  data/extractions/manual_corrections.csv
```

## 2. Place and UNESCO mapping

The scripts preserve the original sequence and write several files to the current directory. Running them from a dedicated output directory keeps generated files contained.

```bash
mkdir -p outputs/geography
cd outputs/geography

python ../../src/geography/place_extractor.py ../../data/outputs/merged
python ../../src/geography/robust_geo_mapper.py all_places.txt places_complete.csv
python ../../src/geography/match_unesco.py \
  places_complete.csv \
  ../../data/geography/unesco_name_dictionary.json \
  places_complete_with_unesco.csv
python ../../src/geography/match_unesco.py \
  places_complete.csv \
  ../../data/geography/tentative_name_dictionary.json \
  places_complete_with_unesco_tentative.csv
python ../../src/geography/paper_unesco_aggregator.py \
  place_summary.csv \
  places_complete_with_unesco.csv \
  places_complete_with_unesco_tentative.csv \
  papers_unesco_status.csv
```

Nominatim usage must comply with its current usage policy. The dictionaries in `data/geography/` are snapshots; record the official UNESCO source date when refreshing them.

## 3. BERTopic final configuration

The paper’s final configuration is:

- `all-MiniLM-L6-v2` embeddings;
- UMAP `n_neighbors=10`, `n_components=8`, `min_dist=0.1`, cosine metric;
- HDBSCAN `min_cluster_size=10`, `min_samples=5`, `cluster_selection_method=eom`;
- c-TF-IDF over 1–3 grams;
- KeyBERTInspired followed by MMR (`diversity=0.3`);
- `nr_topics=auto`;
- outlier reduction with the distributions strategy and threshold `0.1`.

```bash
python src/topic_modeling/bertopic_cluster.py \
  --data_file topic_modeling/meta_data.csv \
  --n_topics auto \
  --umap_n_neighbors 10 \
  --umap_n_components 8 \
  --umap_min_dist 0.1 \
  --hdbscan_min_cluster_size 10 \
  --hdbscan_min_samples 5 \
  --hdbscan_cluster_selection eom \
  --reduce_outliers \
  --outlier_strategy distributions \
  --outlier_threshold 0.1
```

The historical reference reports 464 documents, nine topics plus an outlier set, coherence C_V `0.6736`, outlier ratio `0.1509`, and silhouette `0.1061`. Stochastic libraries, package versions, model revisions, and hardware can change exact results.

Use the printed experiment ID to generate the visualizations:

```bash
python src/topic_modeling/visualize_experiment.py --experiment_id EXPERIMENT_ID
```

## 4. Full 192-configuration search

```bash
python src/topic_modeling/run_grid_search.py
```

The grid varies `n_neighbors` (10/15/20/25), `n_components` (5/8/10), `min_dist` (0.0/0.1), `min_samples` (3/5), HDBSCAN selection (`eom`/`leaf`), and outlier reduction (off/on): `4 × 3 × 2 × 2 × 2 × 2 = 192` runs. Fixed settings are `min_cluster_size=10` and `nr_topics=auto`.

The ranking script min-max normalizes available metrics and uses the manuscript weights: coherence C_V 0.25, silhouette 0.20, topic diversity 0.15, outlier ratio 0.15, Davies–Bouldin 0.10, low-confidence ratio 0.10, and topic-size imbalance 0.05. The top five are a shortlist; the final selection still requires expert inspection.

## Verification boundary

This package has been statically checked for syntax, relative paths, data counts, and alignment of the retained reference metrics. The full LLM extraction, 192 BERTopic runs, online geocoding, ArcGIS map, and Gephi layout were not re-executed during packaging.

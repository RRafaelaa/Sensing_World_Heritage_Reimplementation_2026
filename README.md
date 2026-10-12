# Sensing Cultural Significance: reimplementation materials

This repository shares code, configuration, inputs, and selected results for three methods used in *Sensing Cultural Significance with Social Media and Artificial Intelligence: A Systematic Literature Review*: structured extraction from papers, geographic and UNESCO name matching, and BERTopic modelling. It also shares a cross-method binary table that combines extracted terms with topic assignments. See [data/README.md](data/README.md) for the shared data and [term_mapping/README.md](term_mapping/README.md) for the analysis notebooks.

| Method | Code and configuration | Shared data |
| --- | --- | --- |
| PDF/TXT to structured extraction | `src/extraction/`, `config/extraction/` | `data/extractions/complete_dataset.xlsx`, `data/extractions/Robustness_alignment_Hassan.xlsx` |
| Place and UNESCO matching | `src/geography/` | `data/geography/UNESCO_to_Map.xlsx` and the two UNESCO name dictionaries |
| BERTopic | `src/topic_modeling/` | `data/topic_modeling/meta_data.csv`, `data/topic_modeling/final_probability_distribution.xlsx`, `data/topic_modeling/comparison_report_sorted.csv`, `data/topic_modeling/seed_stability/` |

The cross-method result is `data/term_mapping/Binary_encodings.csv`. Its supporting notebooks are in `term_mapping/`, with their mapping rules in `src/mapping_dictionaries.py`.

Article PDFs, extracted full text, API responses, model caches, and generated intermediate files are not distributed. Put papers you may use in `data/pdfs/`. Generated files can be placed under `data/outputs/` or `outputs/`; those directories are ignored by Git.

## Install dependencies

Create a Python environment and install the packages for the part you want to run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-extraction.txt
```

`requirements-extraction.txt` covers PDF extraction, Gemini, table conversion, and geocoding. For BERTopic and the analysis notebooks, use a separate environment if needed and install `requirements-topic-modeling.txt`. OCR also needs system installations of Tesseract and Poppler. The requirements specify minimum versions, not a locked copy of the historical environment.

## 1. PDF/TXT to structured extraction

Run the commands below from the repository root. `extract_pdf_text.py` reads `data/pdfs/` and writes `data/texts/`; `pdf_to_text.py` supplies its OCR helper.

```bash
python src/extraction/extract_pdf_text.py
```

Set `GOOGLE_API_KEY` in your environment. The full and targeted instruction/schema files are predefined extraction rules; they are inputs to the Gemini requests, not outputs generated from each paper.

For each paper, `src/extraction/heritage_extractor_gemini.py` assembles the prompt sent to Gemini by combining the extracted paper text, its filename, the selected schema and instructions, and a fixed request template in the script.

Run the baseline pass and then the targeted pass:

```bash
python src/extraction/heritage_extractor_gemini.py \
  --input data/texts \
  --output data/outputs/gemini_flash \
  --schema config/extraction/schema_full.json \
  --instructions config/extraction/instructions_full.txt \
  --model gemini-2.0-flash-exp

python src/extraction/heritage_extractor_gemini.py \
  --input data/texts \
  --output data/outputs/gemini_targeted \
  --schema config/extraction/schema_targeted.json \
  --instructions config/extraction/instructions_targeted.txt \
  --model gemini-2.5-pro

python src/extraction/merge_extractions.py \
  --flash data/outputs/gemini_flash \
  --gemini data/outputs/gemini_targeted \
  --output data/outputs/merged

python src/extraction/validate_extractions.py --output data/outputs/merged
python src/extraction/json_to_csv.py \
  --input data/outputs/merged \
  --output outputs/extracted_tables
```

The merger writes one nested JSON `.txt` file per paper. The converter writes flat CSVs and `master_dataset.xlsx`. The shared `complete_dataset.xlsx` is the published study-level extraction dataset; `Robustness_alignment_Hassan.xlsx` documents the Appendix C consistency check between two records of the same study. These published workbooks are not direct, unmodified outputs of the commands above. Historical model IDs or API behaviour may require adaptation for a fresh run.

## 2. Place and UNESCO matching

The place extractor reads the **merged per-paper JSON files** from step 1. It does not read `complete_dataset.xlsx` directly. To start from the published Excel workbook alone, its `merged` sheet would first need conversion to the nested location fields expected by `place_extractor.py`.

The following commands keep generated geography files together:

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

`reverse_geocoder.py` can identify a place from coordinates already present in an extraction. `offline_consolidator.py` can consolidate geocoded places by paper; neither is needed to run the name-matching commands above. Geocoding uses Nominatim, so responses can change over time. UNESCO and Tentative List matching uses the name-dictionary snapshots in `data/geography/` and produces candidate matches.

The automatic paper-level output is `papers_unesco_status.csv`. The shared `UNESCO_to_Map.xlsx` adds selections and `City_Country_Pairs` from author-affiliation locations; the current scripts do not reconstruct that final workbook. It contains UNESCO site **names**, not a table of official UNESCO site coordinates. Coordinates in `places_complete.csv` refer to geocoded place names extracted from papers.

## 3. BERTopic

`data/topic_modeling/meta_data.csv` is the supplied bibliographic input. The model joins title, abstract, and keywords for each record. In the commands below, `--data_file topic_modeling/meta_data.csv` is resolved relative to `data/` by the script.

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

The model uses `all-MiniLM-L6-v2` embeddings by default. To explore the reported 192 parameter combinations, run `python src/topic_modeling/run_grid_search.py`; it calls `compare_experiments.py` to rank the runs. To produce detailed visualizations and a probability workbook for one run, use its printed experiment ID:

```bash
python src/topic_modeling/visualize_experiment.py --experiment_id EXPERIMENT_ID
```

That visualization script writes `probability_distribution.xlsx` under `outputs/figures_improved_v3/EXPERIMENT_ID/`. The shared `final_probability_distribution.xlsx` is the selected historical result. The shared `comparison_report_sorted.csv` ranks the 192 parameter configurations from the historical comparison run.

The dedicated random-seed stability script belongs at `src/topic_modeling/seed_stability.py`. Its results are in `data/topic_modeling/seed_stability/`; see the [data guide](data/topic_modeling/seed_stability/README.md) for the experimental settings and recorded environment. They compare the selected configuration at reference UMAP seed 42 with ten additional seeds (0–9). The files provide per-seed metrics, document topic assignments, topic matches against the reference, and a summary by reference topic. Exact values may differ in a new run because model versions, embeddings, and stochastic steps can differ.

## Cross-method binary encodings

`data/term_mapping/Binary_encodings.csv` is the shared paper-level table of extracted-term indicators and binary topic assignments. The original notebooks in `term_mapping/` show how terms were mapped and combined with BERTopic assignments. They expect intermediate CSV files that are not distributed; [term_mapping/README.md](term_mapping/README.md) identifies the corresponding published workbook sheets and explains the original notebooks' run-time settings. Running the saved notebook cells without those preparations does not directly reconstruct the shared binary table.

# Material manifest and provenance

## Included

| New location | Source | Purpose |
| --- | --- | --- |
| `config/extraction/` | `Extraction_Project/config/` | Full and targeted schemas/instructions. The archived instruction version is retained because the exact historical prompt-to-run association is not recorded. |
| `src/extraction/` | `Extraction_Project/src/` plus packaging utility | PDF conversion, Gemini extraction, validation, merge, JSON-to-table conversion, and reproducible export of reviewer corrections. |
| `src/geography/` | `Extraction_Project/scripts/` | Place extraction, Nominatim geocoding, UNESCO matching, consolidation, and aggregation. |
| `src/topic_modeling/` | latest reusable scripts under `bertopic/bertopic/bertopic/` and `bertopic0112-for_reference/` | BERTopic modelling and visualization. Personal paths were removed; the paper’s outlier-reduction option and 192-run grid were restored. |
| `analysis/notebooks/` | three root notebooks in `Extraction_Project/` | Metadata preparation, terminology processing, and method-theme network analysis. Outputs and execution counters were stripped. |
| `data/extractions/manual_corrections.csv` | Difference between `complete_dataset林一婧校对.xlsx` and `complete_dataset林一婧校对 1.xlsx` | Path-free, cell-level record of the 32 targeted-model corrections, keyed by source filename and field. |
| `data/topic_modeling/meta_data.csv` | `Extraction_Project/data/csv/meta_data.csv` | 464-record BERTopic input with stable keys, title, abstract, keywords, year, and bibliographic metadata. |
| `data/term_mapping/` | selected files from `Extraction_Project/data/csv/` | Baseline and targeted extraction tables plus binary encodings used by the term/network notebooks. |
| `data/geography/` | selected final geography files and dictionaries | Inputs and checkpoints for reproducing the geography/UNESCO pipeline. |
| `reference_outputs/` | selected final BERTopic outputs, MDS matrices, and `JSC.gephi` | Compact cross-check targets, not proof of a fresh rerun. |

## Deliberately excluded

- `Extraction_Project/data/pdfs/`, `pdfs.zip`, and `data/texts/`: third-party full text and derivative text are not required in a public code repository and may have redistribution restrictions.
- `Extraction_Project/data/outputs/`: thousands of raw/repeated model responses, error files, and intermediate copies.
- the two manual-review workbooks: they contain reviewer-local attachment paths. Their 32-cell difference is exported as a path-free CSV instead.
- `bertopic/models/` and all model caches: downloadable dependencies and generated artifacts.
- all large ZIP files and duplicated nested project copies under `bertopic/`.
- old cluster workbooks, repeated HTML visualizations, backup figures, and notebook checkpoints.
- `.git`, `.idea`, `__pycache__`, `.DS_Store`, logs, and failed-response debugging files.
- the manuscript PDF, LaTeX classes, author photographs, and journal production files: useful for publication production, not for method reimplementation.

## Known provenance gaps

1. The manuscript mentions a standalone extraction prompt of roughly 12k characters, but `Extraction_Project/config/prompt.txt` is empty. The retained Gemini script contains the executable prompt wrapper; the long standalone prompt is not recoverable from the supplied folders.
2. The available targeted schema/instruction filenames mention Gemini Pro 3, whereas the manuscript reports Gemini 2.5 Pro. Their content is retained, but an exact checksum tying those files to the December 2025 API run is unavailable.
3. The manual-review workbook contains 465 rows, while the final topic-model input contains 464 keyed records. The retained folders do not encode the exact one-record exclusion decision.
4. Manual checks of World Heritage matches, expert topic labels, ArcGIS styling, and Gephi layout are outcomes of human decisions and are not fully captured as code.

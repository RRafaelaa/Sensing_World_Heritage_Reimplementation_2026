# Term and topic analysis notebooks

The two notebooks in this directory document how extracted terms and BERTopic assignments were combined. Run them from the repository root so their relative `data/` and `src/` paths resolve. The notebooks are preserved as originally written; the publicly shared `data/term_mapping/Binary_encodings.csv` is the historical result, not a file to overwrite while exploring the notebooks.

| Notebook | Purpose | Inputs expected by the saved cells |
| --- | --- | --- |
| `term_mapping.ipynb` | Normalize extracted terms with `src/term_mapping/mapping_dictionaries.py`, count them, and create binary indicators for frequent terms. | Two `complete_dataset.csv` files listed below, plus `data/topic_modeling/meta_data.csv`. |
| `method_theme_network.ipynb` | Combine those indicators with topic assignments and explore their relationships. | The two `dataset_with_binary_columns.csv` files produced by the first notebook, `data/topic_modeling/probability_distribution.csv`, and `data/topic_modeling/meta_data.csv`. |

## Prepare local inputs

The intermediate CSV files read by the notebooks are not included in the repository. The corresponding published workbook sheets are:

| Published source | CSV path expected by the notebook |
| --- | --- |
| `data/extractions/complete_dataset.xlsx`, sheet `Gemini_2_Flash` | `data/term_mapping/Gemini_2_flash_all/complete_dataset.csv` |
| `data/extractions/complete_dataset.xlsx`, sheet `Gemini_2_5_pro` | `data/term_mapping/Gemini_2_5_pro/complete_dataset.csv` |
| `data/topic_modeling/final_probability_distribution.xlsx`, sheet `Full_Probabilities` | `data/topic_modeling/probability_distribution.csv` |

Export the two extraction sheets as UTF-8 CSVs before running `term_mapping.ipynb`. Its saved cells write `dataset_with_binary_columns.csv` alongside each input CSV. The topic CSV is read by `method_theme_network.ipynb` with a semicolon delimiter and a comma decimal mark (`sep=';'`, `decimal=','`). Its source sheet includes `Key`, `Assigned_Topic`, `Topic_Label`, and the `Prob_Topic_*` columns required by that notebook. The local intermediate paths above are ignored by Git.

## Notes for running the original notebooks

- `term_mapping.ipynb` uses the standard Python type names `Dict`, `List`, `Set`, and `Tuple` in function definitions but omits their import. Before running those definitions, execute `from typing import Dict, List, Set, Tuple` in the notebook session. This only supplies names for type annotations; it does not change the mapping rules.
- The saved call in `method_theme_network.ipynb` passes `binary_topics=False`, producing continuous `Prob_Topic_*` columns. To create `Binary_Topic_*` indicators like those in the shared binary table, use `binary_topics=True` for that call.
- A saved cell writes to `data/term_mapping/Binary_encodings.csv`. Redirect that output to a new path when experimenting so the shared result is preserved.

These preparation steps and settings explain the relationship between the published inputs, the original notebooks, and the shared result. They do not guarantee a byte-for-byte reconstruction of the historical CSV.

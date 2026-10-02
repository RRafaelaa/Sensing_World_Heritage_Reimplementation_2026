# Data inventory

## `extractions/`

`manual_corrections.csv` contains the 32 cell-level differences between the two
manual-review workbooks. Each row records the sheet, workbook row, source
filename, field, old value, and corrected value. The workbooks themselves are
not included because they contain reviewer-local attachment paths.

## `topic_modeling/`

- `meta_data.csv`: 464 keyed records used as the retained BERTopic corpus input. The model concatenates Title, Abstract Note, and All Keywords. Local attachment paths have been reduced to filenames.
- `probability_distribution.csv`: topic probabilities used by the method-theme synthesis notebook.

## `term_mapping/`

- baseline (`Gemini_2_flash_all`) and targeted (`Gemini_2_5_pro`) complete flat tables;
- matching binary-column tables used for term occurrence analysis;
- `Binary_encodings.csv`, the combined method/topic matrix checkpoint.

These are processed research data, not raw observations. Empty cells mean the source paper/model extraction did not supply a value; they must not be interpreted as zero or absence.

## `geography/`

The folder contains place-level and paper-level checkpoints, coordinates, UNESCO/Tentative matches, and name dictionaries. `all_places.xlsx` includes the intermediate tables required to audit the mapping chain.

Automated name overlap is only a candidate match. The paper reports manual correction of World Heritage properties; those expert decisions should be treated as part of the protocol.

## Redistribution warning

The data include bibliographic metadata, abstracts/keywords, and model-extracted text derived from published papers. Review source-database terms and quotation/copyright risk before assigning a public data license.

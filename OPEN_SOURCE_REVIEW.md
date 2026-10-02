# Open-source release review

This directory is a release candidate, not a legally cleared publication.

## Completed during packaging

- Removed raw article PDFs, extracted article text, model weights, caches, large archives, and repeated outputs.
- Removed personal absolute paths from the retained BERTopic scripts.
- Reduced local attachment paths in retained delimited data to filenames and replaced the path-bearing reviewer workbook with a 32-row correction table.
- Confirmed that no API key matching common Google/OpenAI key patterns appears in the retained text/code/config files.
- Kept credentials out of commands and added `.env`/key patterns to `.gitignore`.
- Limited reference outputs to compact verification targets.

## Required before a public release

1. Choose and add a code license. No license has been selected automatically because that is an author/rights-holder decision.
2. Decide a separate data license. Confirm that bibliographic metadata, abstracts, keywords, and extracted quotations may be redistributed under that license and under the original database/export terms.
3. Review the study-level CSV tables for long verbatim excerpts from papers. Facts and structured labels may be releasable while substantial quotations may require removal or source-specific permission.
4. Document the exact Web of Science/Scopus export permissions and the UNESCO list snapshot/source date.
5. Confirm author approval for the manually corrected data and topic labels.
6. Add the final paper citation/DOI when available.
7. Run a fresh clean-environment smoke test, then archive exact package versions and random seeds.

Do not describe the historical reference outputs as newly reproduced results. They are retained outputs from the working folders.

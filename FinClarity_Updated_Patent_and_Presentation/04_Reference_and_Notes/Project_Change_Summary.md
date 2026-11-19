# FinClarity update summary

## Materials reviewed

- Current FinClarity workspace source and configuration.
- `Project_Report_compressed.pdf` and `Project_Presentation.pptx`.
- `FINCLARITY PATIENT_compressed (1).pdf`, a previously completed FinClarity invention disclosure whose section order matches the supplied filing format.
- `patent llm with US patent.docx`, a groundwater invention disclosure sample. It was used only to inspect the style and form sections; no groundwater content, claims, patent IDs, or wording was reused.
- `FINCLARITYSUBBB.zip`, which contains the earlier report and slide materials, not the current application source.
- The supplied signature image was reviewed but is not inserted into the new filing draft.

## Verified project facts used

- The active saved artifact is a Random Forest pipeline for historical accepted-file versus rejected-file source membership.
- The legacy workflow compares Logistic Regression and Random Forest using validation average precision. Saved validation AP: LR 0.9363, RF 0.9521.
- Active features: loan amount, DTI, employment length, and purpose. The 80,000 row training sample is balanced between the historical files.
- Test metrics from the saved report: precision 0.9140, recall 0.8945, F1 0.9041, ROC-AUC 0.9570, average precision 0.9509, and Brier score 0.0730.
- Local SHAP is attached to an individual assessment; global permutation importance measures held-out average-precision change.
- Ollama serves the configured pretrained `llama3.1:8b` model locally. `llama3.2:3b` is also installed. Live general chat was tested; no LLM fine-tuning is claimed.
- The code has a controlled tool registry and Python-side intent routing. A live general question completed in about 31 seconds on the test laptop. The live model-performance path returned a missing-artifact notice in the direct-call test context; it must not be described as a tested live metrics answer with the application artifact loaded.
- Multi-dataset management/training exists in the current source, including explicit labels, column mapping, compatibility filtering and versioning. Automated tests use small fixtures; external source data was not downloaded in the verification run.

## Deliberate statements and omissions

- No novelty, patentability, ownership, freedom-to-operate, or prior-art conclusion is asserted.
- No patent claims are included because the supplied disclosure format does not require them and no professional claim review has occurred.
- Inventor table carries Raghav Kumar from the current request and identifies Tejesh Balaga from prior FinClarity materials as needing confirmation. Contact fields are to be confirmed; signature blocks are left blank.
- Disclosure history, public links/dates, MOU status, companies to approach, and filing route remain marked for confirmation because the supplied material did not verify current status.
- No prior UI screenshots are presented as current screenshots. The technical diagrams in the presentation are editable schematics.

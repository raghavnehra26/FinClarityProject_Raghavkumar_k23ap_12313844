# FinClarity implementation audit

Audit date: 10 November 2026. This audit reflects the files and artifacts available in the supplied workspace and the attached archive.

## Existing implementation reviewed

The workspace contains `app.py`, `ai_agent.py`, `app_logic.py`, `config.py`, `download_dataset.py`, `explainability.py`, `inspect_dataset.py`, both training modules, requirements, README, environment template, ignore rules, and seven pytest modules. `train_model.py` is present, resolving the missing-module issue described in the original brief. The UI uses Streamlit's sidebar navigation and implements Overview, Dataset inspection, Applicant assessment, Explainability, Synthetic demonstrations, Model evaluation, Ask FinClarity, and Responsible AI pages.

## Data and model evidence

The attached ZIP includes both accepted and rejected LendingClub CSVs (compressed and extracted entries). Existing reports describe 2,260,701 accepted rows and 27,648,741 rejected rows. The checked-in workspace has a 28,861,629-byte joblib model artifact and JSON reports. The evaluation report identifies its target as accepted-file versus rejected-file membership, uses four shared features, selects a random forest by validation average precision, and evaluates on a balanced held-out sample of 12,000 records. Reported metrics are precision 0.9140, recall 0.8945, F1 0.9041, ROC-AUC 0.9570, average precision 0.9509, and Brier score 0.0730. These are not loan approval or eligibility metrics.

## Material limitations and verification

- The archive contains raw financial records and is deliberately excluded from the deliverable ZIP. Users should extract it locally and point the training script to the CSVs.
- Existing JSON inspection metadata included machine-specific temporary file paths. Those paths are removed from the final package's copies of reports.
- Loading the saved artifact and generating a sample prediction plus SHAP values was verified; SHAP contributions summed to the model score within floating-point precision.
- The current broad test run could not complete reliably in the sandbox because pytest's default temporary directory was inaccessible; it showed several setup errors and was interrupted while stalled. Rerun with `--basetemp .pytest-tmp` from the project folder. No pass is claimed for the full suite in this audit.
- Microsoft Foundry and Content Safety are optional; no credentials or resources were configured, so their live integration was not tested.
- The report and presentation summarize saved evaluation artifacts and document the limits above. No institutional certificate or unverified individual team contributions are asserted.

## Recommendations

Use a supported Python 3.11 or 3.12 environment if dependency wheels for the installed Python version are unavailable. Keep raw data and locally trained artifacts out of public repositories. Before any external demonstration, explain source-file membership and the unmatched populations, and do not call the score an approval probability.

A focused module run passed 13 tests. The full suite with a writable basetemp advanced through the non-UI tests but stalled in Streamlit AppTest; a separate focused AppTest run also did not complete within the available execution window. Streamlit launched as a process, but local HTTP health probing was denied by the sandbox socket policy. Therefore browser-level navigation is not claimed.

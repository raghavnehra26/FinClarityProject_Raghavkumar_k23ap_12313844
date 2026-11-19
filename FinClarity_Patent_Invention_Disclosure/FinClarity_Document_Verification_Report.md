# FinClarity disclosure verification report

## Materials inspected

- Reference: `C:\Users\ragha\Downloads\patent llm with US patent.docx` (Annexure 3B sample; 2 Word sections, Letter portrait, 1-inch margins; single internal-inventor table layout, prior-art table, disclosure table, and blank NOC appendix).
- Project source: `app.py`, `ai_agent.py`, `agent_tools.py`, `train_application_model.py`, `train_model.py`, `multi_dataset_training.py`, `dataset_manager.py`, `dataset_discovery.py`, `explainability.py`, `config.py`, `README.md`, `.env.example`, and requirements.
- Saved artifacts: `reports/model_evaluation.json`, `reports/dataset_inspection.json`, and `models/finclarity_model.joblib`.

## Verified technical description

- Active paired-file classifier: historical accepted-file versus rejected-file source membership; not loan eligibility, actual approval, creditworthiness, or default prediction.
- Algorithms compared: Logistic Regression and Random Forest; selection by validation average precision. Saved artifact selects Random Forest.
- Features: `loan_amnt`, `dti`, `emp_length`, and `purpose`.
- Saved sample: 80,000 total training-workflow records (40,000/source); held-out test 12,000 balanced rows. Validation AP: Logistic Regression 0.9363; Random Forest 0.9521. Test precision 0.9140, recall 0.8945, F1 0.9041, ROC-AUC 0.9570, AP 0.9509, Brier 0.0730. These measure source membership only.
- Explainability: local SHAP and global held-out permutation importance. Neither establishes causation or fairness.
- Assistant: code supports Ollama and configured default `llama3.1:8b`, approved Python tools, bounded orchestration, and local deterministic fallback. The local API endpoint was not listening during this preparation; installed model and live inference/tool calling are unverified.
- Dataset counts from saved inspection: accepted 2,260,701 x 151 columns; rejected 27,648,741 x 9 columns. The files are unmatched populations.

## Prior-art references

Three patent publication records were reviewed for bibliographic/abstract-level technical comparison; see `FinClarity_Patent_Research_References.md`. No novelty, patentability, claim-scope, or freedom-to-operate conclusion is made. Company links are official homepages checked through their official sites.

## Confirmation and rendering limits

- Inventor phone, personal email, signature, disclosure history, commercialization status, regulatory approvals, and any collaboration/funding terms remain for confirmation.
- External organization NOC remains a blank appendix. No second inventor is listed.
- Dataset and model license/third-party rights require review.
- The reference and final DOCX could not be rendered through the packaged renderer because LibreOffice (`soffice.exe`) is not installed. Microsoft Word automation was attempted but Windows denied it with `E_ACCESSDENIED`. The final PDF is generated from the same disclosure content using ReportLab, not exported from Word, so visual pagination is not guaranteed to match.
- The final PDF was rendered to eight PNG pages with Poppler and visually inspected as a contact sheet and at page scale. It is Letter size, opens successfully, and contains no replacement glyphs in extracted text. The prior-art table continues cleanly across pages; the architecture diagram and appendix are present.
- Structural content checks found one inventor name occurrence, no Tejesh Balaga, no unrelated sample-invention text, four Word tables (inventor, prior art, results, and disclosure questions), and one Word section. Native Word visual QA remains outstanding because automation access was denied.
- Ollama live checks were attempted at `http://localhost:11434`; no listener was present.

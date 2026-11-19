# Presentation verification report

## Output
Folder: `C:\Users\ragha\Documents\FinClarity\FinClarity_Updated_Presentation`

- `FinClarity_Updated_Project_Presentation.pptx` (20 slides; includes presenter notes)
- `FinClarity_Presentation_PDF.pdf` (20 pages, rendered from the same slide previews)
- `Presentation_Content_and_Speaker_Notes.docx`
- `Presentation_Assets/` (four project architecture/workflow diagrams)

## Files inspected
- Current application and agent code: `app.py`, `ai_agent.py`, `agent_tools.py`, `app_logic.py`, `config.py`.
- Training and explainability code: `train_application_model.py`, `train_model.py`, `multi_dataset_training.py`, `explainability.py`.
- Dataset code and docs: `dataset_manager.py`, `dataset_discovery.py`, `README.md`, `.env.example`, `requirements.txt`.
- Tests under `tests/` and saved artifacts `reports/model_evaluation.json` and `reports/dataset_inspection.json`.
- Previous project report `FinClarity_Final/submission/Project_Report.pdf`.
- Previous 17-slide deck in `FinClarity_Updated_Patent_and_Presentation/02_Professional_Presentation/`. It contained an invention-disclosure slide and asserted the model tags were verified. Those claims/content were excluded or corrected.
- Updated academic report package `FinClarity_Updated_Project_Report/`.

## Implemented project facts presented
- Active classifier: Random Forest selected over Logistic Regression for historical accepted-file vs rejected-file source membership.
- Inputs: `loan_amnt`, `dti`, `emp_length`, and `purpose`, using a fitted preprocessing pipeline.
- Local explanation: SHAP; global explanation: held-out permutation importance.
- Dataset registry, explicit labels, schema mapping/provenance, curated discovery and model versioning are described as code capabilities, not as evidence of a fresh remote download or multi-source retraining.
- Ask FinClarity code is configured for a pretrained Llama model through local Ollama and approved Python tools. It does not replace the classifier.

## Numerical results and source
`reports/model_evaluation.json` contains the figures used: training sample 80,000 rows (40,000 per class); test sample 12,000 rows (6,000 per class); validation average precision 0.9363 for Logistic Regression and 0.9521 for Random Forest; test precision 0.9140, recall 0.8945, F1 0.9041, ROC-AUC 0.9570, average precision 0.9509, Brier score 0.0730. Test values describe source membership on the balanced historical sample only. Global permutation importance is shown from the same saved artifact.

## Verification status
- Automated run: `.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_streamlit_app.py` completed with 52 passed in 16.19 seconds.
- The Streamlit AppTest was excluded because a previous run stalled in Windows Python 3.13 asyncio Proactor setup. It is not claimed as passed.
- Configured model tag: `llama3.1:8b` (`.env.example` and `ai_agent.py`). Installed model status is unknown because Ollama was not responding at `http://localhost:11434` during this verification. No current live Llama response or tool call was verified. Automated agent tests are not live service tests.
- No browser inspection of the current running app was completed. No real current-version screenshots were available. Slide 18 clearly labels the screenshot placeholder.
- The new PPTX was rendered to 20 slide images and visually reviewed as a deck and on individual slides. Structural package and geometry validators passed with 0 findings; the quantitative chart validator passed for native editable charts on slides 12 and 15. The native PowerPoint application was not opened in this environment.
- The PDF contains 20 pages, one for each slide render, and was visually reviewed. The artifact-tool PDF export path was unavailable because its bundled PDF runtime could not resolve `#standard-fonts/Helvetica`; the delivered PDF uses the exact PNG slide renders.
- The finalizer’s publish step could not create a Windows hard link in this managed workspace. The validated candidate was copied to the requested output path after package, layout, and chart checks passed.

## Missing screenshots and unverified claims
- Capture genuine screenshots from the current running application before a live UI demonstration.
- Current Ollama installation, model availability, live inference, and live tool-calling remain unverified until the local service is started and checked.
- No Microsoft Foundry or Azure AI Content Safety service is claimed as operational.
- The model is not a real loan eligibility, approval, creditworthiness, or default predictor. SHAP and permutation importance do not prove causality or fairness.
- The speaker-notes DOCX was parsed successfully with `python-docx` (166 paragraphs). Visual DOCX rendering was attempted but could not run because LibreOffice (`soffice.exe`) is not installed; native Word inspection was unavailable.

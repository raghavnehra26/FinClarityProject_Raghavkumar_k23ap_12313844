# FinClarity

**Explainable Loan Eligibility Agent** — a local educational prototype. The supplied accepted/rejected CSV files do **not** contain matched decisions for the same applicants. The model trained from them predicts membership in either historical source file; it does not predict actual loan approval, eligibility, or default. FinClarity must never be used to make real lending decisions.

## What exists today

- Streamlit pages for an executive dashboard, dataset management/discovery/exploration, model training, applicant assessment, explainability, an AI assistant, model performance, responsible AI, and settings.
- A streaming trainer for the supplied large accepted/rejected CSVs. It compares Logistic Regression and Random Forest, selects by validation average precision, evaluates on a held-out test set, and saves the real pipeline.
- Shared input-time fields only: requested amount, debt-to-income ratio, employment length, and normalized loan purpose. Post-outcome fields and `loan_status` are not model inputs.
- Per-applicant SHAP explanations rank the inputs that moved the model score most and show the direction and size of each shift relative to a reference profile. Held-out permutation importance shows which features mattered across the evaluation sample.
- Locally hosted Ollama Llama agent with general-purpose conversation, FinClarity evidence tools, bounded tool execution, session context, and an honest offline fallback.
- Managed multi-CSV registry with explicit accepted/rejected labels, provenance, checksum duplicate detection, schema mapping, chunked inspection, and raw-file preservation.
- Multi-source training for Logistic Regression or Random Forest, compatible shared feature selection, stratified train/validation/test splits, saved version history, and rollback.
- Curated dataset discovery for user-triggered source options. Authentication, license terms, and label semantics must be checked before files enter training.
- Unit tests use small temporary fixtures to test program behavior. They are not a training dataset and do not produce project performance claims.

**Dataset verified and model trained:** the available accepted file has 2,260,701 rows and 151 columns; the rejected file has 27,648,741 rows and 9 columns. The accepted file includes issued-loan outcomes (`loan_status`: 1,076,751 Fully Paid, 878,317 Current, 268,559 Charged Off, plus other statuses and 33 missing). The rejected file is a different set of applications, not matched counterparts.

The trained random forest's held-out results are precision **0.914**, recall **0.895**, F1 **0.904**, ROC-AUC **0.957**, average precision **0.951**, and Brier score **0.073** on a balanced test sample of 6,000 records from each source file. These measure only source-file classification on this historical sample. They are **not** evidence of approval accuracy, default risk, or performance on real applications. Inspect the evaluation report and limitations before demonstrating.

## Project layout

```text
FinClarity/
|-- app.py                 Streamlit UI
|-- app_logic.py           Applicant input validation
|-- ai_agent.py            Ollama client, tool orchestration, and local fallback
|-- agent_tools.py         Approved read-only model, dataset, project, and concept tools
|-- dataset_manager.py     CSV inspection, explicit labels, mapping, provenance, and registry
|-- dataset_discovery.py   Curated source catalog and authenticated download manager
|-- multi_dataset_training.py  Multi-source training, evaluation, and model versions
|-- config.py              Local paths and environment configuration
|-- explainability.py      SHAP, permutation importance, synthetic examples
|-- inspect_dataset.py    CSV schema and quality report
|-- train_model.py         Reproducible model training/evaluation
|-- train_application_model.py  Streaming source-membership model for the supplied pair
|-- requirements.txt
|-- .env.example           Ollama local endpoint and installed model tag
|-- data/README.md         Dataset acquisition and target limitations
|-- tests/                 Automated tests with temporary fixtures
|-- models/                Generated model artifact (ignored by Git)
|-- reports/               Generated JSON reports (ignored by Git)
```

## Windows setup and run

The project already has a Python 3.13 virtual environment at `.venv`. The global `python` and `py` commands were not found in the inspected terminal, so start with the environment's Python directly:

```powershell
Set-Location C:\Users\ragha\Documents\FinClarity
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In the VS Code integrated terminal, activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Once activated, use the standard commands below:

```powershell
python train_application_model.py
python -m pytest -q
python -m streamlit run app.py
```

For a direct launch without activating the virtual environment (including from
PowerShell opened in `C:\Windows\System32`), use:

```powershell
Set-Location C:\Users\ragha\Documents\FinClarity
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "llama3.1:8b"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The last command opens the app in your browser. The workspace's VS Code settings already point to `.venv\Scripts\python.exe`.

### Supplied dataset files

The files are available as:

```powershell
C:\Users\ragha\AppData\Local\Temp\accepted_2007_to_2018Q4.csv
C:\Users\ragha\AppData\Local\Temp\rejected_2007_to_2018Q4.csv
```

Train with the project environment's Python from the project folder:

```powershell
.\.venv\Scripts\python.exe train_application_model.py
```

The script defaults to those two paths, scans them in chunks (without copying either 1.7 GB file), keeps at most 40,000 examples per source, then writes the model and JSON reports. If you move either file, provide explicit paths:

```powershell
python train_application_model.py --accepted 'C:\path\accepted.csv' --rejected 'C:\path\rejected.csv'
```

If these temporary files are removed, extract `accepted_2007_to_2018Q4.csv` and `rejected_2007_to_2018Q4.csv` from the ZIP into a permanent local folder and pass those extracted paths. A CSV path cannot be a path inside an unextracted ZIP archive.

The separate `inspect_dataset.py` and `train_model.py` scripts are for a single compatible LendingClub historical-outcome CSV; they are not the training route for this accepted/rejected pair.

### Optional alternative: KaggleHub competition CSV

This downloads a different single-file competition dataset; it is **not** needed
for the supplied accepted/rejected source files above. Use it only if you
intentionally switch to that historical default-outcome workflow.

The project includes `download_dataset.py`, which wraps the KaggleHub snippet and lists the downloaded CSV files before copying anything. First open the [Lending Club competition page](https://www.kaggle.com/c/lending-club/data), sign in, and accept its terms. Then create a Kaggle API token at [Kaggle account settings](https://www.kaggle.com/settings/api).

```powershell
python download_dataset.py --login --file loanStats_train.csv
```

Enter a newly generated token only into the hidden local prompt; do not send or commit it. KaggleHub keeps `login()` credentials in memory for that Python process, so the helper authenticates and downloads in the same run. The Kaggle competition currently lists `loanStats_train.csv` as the training set; the helper verifies that exact filename against the downloaded files and copies it to `data\loanStats_train.csv`. To review files interactively instead, use `python download_dataset.py --login --select`.

The helper refuses test/sample/submission filenames and never overwrites an existing dataset. KaggleHub caches its download outside the project. Verify the selected file's target definition and terms before training. Each `--login` run prompts for the token; the token is not saved by this project.

The inspector prints column names, data types, missing counts, duplicate count, dataset size, target distribution, and possible sensitive/identifying column names. To save the schema-only report:

```powershell
python inspect_dataset.py --output reports\dataset_inspection.json
```

The trainer also saves `models\finclarity_model.joblib`, `reports\dataset_inspection.json`, and `reports\model_evaluation.json`. Training is never performed on app startup. Artifacts are generated locally and ignored by Git.

## Modeling and data limitations

For the supplied pair:

- `source_outcome = 1`: sampled from the accepted file (issued-loan records).
- `source_outcome = 0`: sampled from the rejected file (rejected application records).

This is **source-file membership**, not a bank approval prediction. Rows are unmatched, the source files may have different coverage and collection processes, and historical accepted amounts may not exactly represent the amount requested. The rejected dataset has extreme numeric outliers; invalid DTI values are treated as missing and imputed within the training pipeline. The sampled classes are balanced for training, so the resulting score is not a real-world probability or prevalence estimate.

Only shared fields are used: requested/loan amount, reported DTI, employment length, and purpose/title normalized to common categories. `loan_status` and all post-outcome information are excluded from model features. Employment and financial fields can still act as sensitive proxies; removing named protected attributes does not establish fairness.

Streaming reservoir sampling scans each file and retains up to 40,000 rows per source. A stratified random split of that balanced sample is used; because date is not a shared input, temporal generalization is not evaluated. Model selection uses validation average precision; test metrics describe only this sample and split. Scores are uncalibrated classifier outputs, not approval probabilities. Full-file duplicate counts are not assessed. Column names are recorded, data types are inferred from the first 1,000 rows, and selected-field missing/invalid counts are scanned over the full files. Potentially sensitive/identifying column names are flagged, but this name-based check is not a privacy or fairness audit. The Kaggle page labels the dataset CC0: Public Domain; check both the Kaggle dataset information and applicable source terms before redistribution. See the saved dataset report for the source and caveat.

Synthetic examples are generated independently from training-set feature summaries, then ranked by actual saved-model scores. Displayed labels, scores, and SHAP values are calculated at runtime; they are not hardcoded. Profiles may not represent plausible correlated real-world applicants.

## Multiple datasets and the training center

Open **Dataset Management** to upload multiple accepted and rejected CSV files,
preview each one, explicitly choose its historical source label, and map available
columns to canonical feature names. Imported raw CSVs are retained under
`data\managed`; a registry records their checksums, mappings, import time, and
provenance. Identical file content is detected by checksum. The app does not infer
an accepted/rejected label from a filename. Missing values and duplicate rows are
reported, and inspection uses chunks for larger CSVs.

Open **Model Training Center** to select multiple registered sources. Training
uses only compatible fields available across all selected files, preserves their
explicit source labels, and splits the combined sample into stratified training,
validation, and test sets. Numeric imputation/scaling and categorical imputation/
encoding are fit within the training pipeline and reused at inference. Each
successful run stores a versioned pipeline and metadata under `models\versions`,
records evaluation history, and can be reactivated from the version list. The
existing fixed-pair streaming trainer remains a separate workflow;
`train_model.py` remains the distinct historical repayment/default workflow.

Compatibility is conservative: a field must map in every selected source.
Features known to be post-decision (including interest rate and grade) are
excluded from source-membership training. The app reports schema coverage and
source distribution differences, but cannot establish that unmatched accepted
and rejected records are comparable or represent a real lending population.

## Dataset discovery and downloads

**Dataset Discovery** shows a small curated catalog and allows explicit download
attempts. The Lending Club accepted/rejected source requires user-configured
Kaggle authentication; the app does not request or store a Kaggle token. The
download is staged locally, checksum-identified, integrity-checked, and shown in
the registry workflow for explicit label assignment and review before training.
Review the current Kaggle license before use or redistribution; the app does not
assert license approval. The separate Lending Club competition CSV is a
repayment-outcome workflow, not an accepted/rejected source-membership dataset.
UCI Credit Approval is listed as reference information because its anonymized
binary target does not establish accepted/rejected Lending Club source labels.
If network access or Kaggle authentication is unavailable, manual uploads and
local data remain supported. Dataset downloads were not performed as part of
this implementation check.

Importing any downloaded file requires users to inspect provenance, schema, and
the original meaning of its labels. A label must never be inferred where the
source does not document it. More rows do not automatically mean a better model.

## Microsoft AI services (separate from the local agent)

The current Imagine Cup FAQ states that a submission must include at least two Microsoft AI services, and asks for a pitch deck, recorded pitch, and demo video. The FAQ also contains 2026-specific eligibility dates. Confirm the rules and deadlines for the cycle you actually enter:

- [Imagine Cup FAQ](https://imaginecup.microsoft.com/en-us/support/faq)
- [Official rules linked by the FAQ](https://aka.ms/icrulesregs)
- [Microsoft Foundry overview](https://learn.microsoft.com/en-us/azure/foundry/what-is-foundry)
- [Azure AI Content Safety overview](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)

Ask FinClarity uses the Llama model served by Ollama on this laptop. It does not
call Microsoft Foundry, Azure OpenAI, or another hosted LLM. Microsoft services in
the competition references are separate options, not part of this local agent.

## Architecture

```text
Local LendingClub CSV
        |
        +--> inspect_dataset.py --> schema/quality JSON
        |
        +--> train_application_model.py --> chunked source scans + balanced sample
                                           preprocessing + selected classifier
                                   |              |
                                   |              +--> held-out evaluation
                                   +--> joblib model artifact
                                                |
Streamlit UI <----------------------------------+
     |                                          |
     +--> actual prediction --> SHAP local + permutation global explanations
     |
     +--> Ask FinClarity --> Ollama on localhost --> approved Python evidence tools
              |                         |
              |                         +--> current assessment / SHAP
              |                         +--> global importance / metrics
              |                         +--> dataset / responsible AI / glossary
              +--> deterministic local answer
              +--> Llama writes a response from retrieved evidence locally
```

## Ask FinClarity AI agent

Ask FinClarity is an explanation layer over the existing saved classifier and a
local general-purpose educational/technical assistant. It does not train,
replace, or override that model. Ordinary greetings and general questions are
sent to Llama without invoking project evidence tools. Questions about actual
project state use approved Python tools. The agent can retrieve the current
assessment, its actual SHAP values, held-out permutation importance, saved
validation and test metrics, dataset inspection summaries, responsible-AI
limitations, and curated general financial definitions through explicitly
allowlisted, read-only Python tools: current assessment, current SHAP explanation,
global permutation importance, model performance, dataset summary, responsible AI,
project information, and one selected financial concept. It calls Ollama's local
`/api/chat` endpoint with tools, executes at most three tool rounds (four calls
per round), and returns tool results to Llama before final generation. If a model
does not return a tool call, controlled Python-side intent routing retrieves
evidence. No arbitrary code is executed. Invalid tool requests and API failures
produce distinct sanitized notices. Numerical claims are checked against retrieved
evidence, and unsafe lending-decision wording is rejected. Chat history is held in
Streamlit session state, can be cleared, and is reset when the current assessment
changes. Dark and light theme tokens, suggested project and general questions,
clear-chat controls, and session conversation history are part of the app UI.
Applicant context is refreshed from current application state rather than treated
as permanent personal memory.

No extra Python dependency is required. Ollama is a separate local application.
FinClarity defaults to `llama3.1:8b`; `llama3.2:3b` is also available locally
as an optional faster model from the Ask FinClarity model selector. Check the
installed model tags and set the exact tag in the environment if you want to
override the default:

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
ollama list
$env:OLLAMA_MODEL = "llama3.1:8b"
python -m streamlit run app.py
```

FinClarity's primary model is `llama3.1:8b` (approximately 4.9 GB); the optional
`llama3.2:3b` is also installed (approximately 2 GB). If the
Ollama desktop app is not running, start it from the Start menu. For a
CLI-only setup, run `ollama serve` in a separate terminal. `ollama list` reports the actual installed tags;
the Ask FinClarity page also queries `/api/tags`, shows installed models, and lets
you choose one. Set `OLLAMA_MODEL` to retain a default selection. The UI reports
Connected only after Ollama responds and the selected tag is present. If Ollama
or the selected model is unavailable, the existing deterministic local response
is used only for supported local greetings, glossary concepts, or available
FinClarity evidence; other general questions report that Ollama is unavailable.
The page explains why. If no Llama tag is installed, install the
model you intend to use with `ollama pull <model-tag>` and verify it with
`ollama list`. All LLM requests go to the configured local
Ollama endpoint; there is no external LLM API or credential.

The agent treats user messages and retrieved data as untrusted and exposes no
Python execution. Financial-concept selection is allowlisted; model-performance
filters accept only known metric names. It uses a low temperature and rejects
unsupported lending claims and unsupported numeric claims. No fairness test is
claimed unless a saved, explicit fairness result exists. Run `ollama list` to
identify your model and verify that it supports tool calls; otherwise the
controlled Python router provides evidence to the model for its response.

## Responsible-use safeguards

- No application approval or rejection prediction is represented as a bank decision.
- No score or explanation is shown before a real model artifact exists.
- Inputs are validated and not persisted by the app; LLM calls stay on the configured local Ollama endpoint.
- No API key is required for the local Ollama connection.
- SHAP describes model behavior, not causation or fairness.
- The historical source may contain selection bias, economic-period effects, label noise, and features that proxy for protected attributes.
- This prototype is not legal, regulatory, lending, or financial advice.

## Verification performed for this workspace update

- `python -m compileall -q ...` completed successfully for the application, agent,
  dataset, training, and explainability modules.
- `python -m pytest -q --ignore=tests/test_streamlit_app.py`: **52 passed**.
- The Streamlit server started and returned HTTP 200 on localhost. This verifies
  startup and page delivery, not visual quality or every interactive page.
- Ollama responded on `http://localhost:11434`; `llama3.1:8b` and `llama3.2:3b`
  were listed, and the configured model resolved to `llama3.1:8b`. A live general
  question completed. A live model-performance question correctly reported that
  no trained model artifact was loaded in that process.
- Streamlit AppTest did not complete in this execution environment. It stalled
  while Python 3.13's Windows Proactor event loop created its socketpair in
  Streamlit's local script runner (`socket.accept`); this is an environment/UI
  test-runner limitation, not a passing UI test. No browser-based visual review
  was completed.
- Dataset download behavior was covered with mocked tests; no external dataset
  was downloaded during these checks.

## Two-day build and submission plan

1. **Day 1 — data and working model:** supply Kaggle data; inspect and verify its documentation/schema/terms; run tests; train; examine held-out results; fix errors or stop if the target/features are unsuitable.
2. **Day 2 — explanations and demo:** run the Streamlit app; verify source-membership prediction and SHAP correspondence; generate model-backed synthetic score examples; optionally configure and test both Microsoft AI services without unreviewed costs; prepare the pitch, demo recording, and rules checklist.

### Short pitch outline

1. Problem: people often receive little explanation about financial-model outputs.
2. Prototype: FinClarity turns a historical source-membership model output into a readable, feature-grounded explanation.
3. Demonstration: show why these unmatched source files cannot establish actual approval decisions, then show actual test metrics, a profile score, and SHAP factors.
4. Responsible AI: disclose class balancing, unmatched records, uncalibrated scores, bias, privacy, and historical-data limits.
5. Next step: validate with people and domain experts, audit group performance, and verify local Ollama model behavior.

### Demonstration script

1. Start on Overview and say explicitly this is not a loan approval system.
2. Open Dataset inspection and show the actual target and schema report.
3. Open Model evaluation; explain precision/recall, the confusion matrix, and the uncalibrated score caveat.
4. Enter a demonstration profile in Applicant assessment; show its real prediction and per-applicant SHAP factors.
5. Open Explainability for held-out global permutation importance.
6. Generate Synthetic demonstrations and explain that the model generated the scores and explanations.
7. Ask a question and show the evidence label and local Llama response (or explain the deterministic fallback if Ollama is unavailable).

## Competition checklist at this workspace inspection

- [x] Local Streamlit app, inspection, training, evaluation, input checks, SHAP, synthetic profile generation, and tests implemented.
- [x] Dataset is not fabricated; training and outputs remain unavailable without it.
- [x] FAQ reviewed; two Microsoft AI services, pitch deck, recorded pitch, and demo video are called out.
- [x] Accepted/rejected CSV schemas and full row counts scanned; accepted historical labels verified. Kaggle page license noted; full-file duplicate count not assessed.
- [x] Actual model trained and held-out evaluation saved; distinct synthetic model scores and matching SHAP outputs verified.
- [ ] Real model trained; actual evaluation and two distinct synthetic examples generated.
- [x] Ollama model `llama3.1:8b` installed; a real tool-grounded local Llama response verified.
- [ ] Current cycle rules, eligibility, deadlines, and required submission assets confirmed.
- [ ] Fairness/group evaluation and accessibility reviewed with suitable data and expertise.

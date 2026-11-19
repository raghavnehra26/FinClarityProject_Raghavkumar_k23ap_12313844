# Verification and limitations

## Source and artifact inspection
The report was checked against the current `app.py`, `ai_agent.py`, `agent_tools.py`, training, dataset, and explainability modules; `README.md`; `requirements.txt`; and saved evaluation and dataset-inspection JSON files. The active classifier task is `source_outcome` membership in accepted versus rejected historical source files. `train_model.py` also supports a distinct historical repayment workflow and was not conflated with the active model.

## Model evidence
- Training workflow sample: 80,000 rows, reservoir sampled at up to 40,000 per source.
- Held-out test: 12,000 rows, balanced 6,000 per source.
- Selected model: Random Forest; selected by validation average precision.
- Actual metrics and global permutation importance are reproduced from `reports/model_evaluation.json`.
- Dataset profile: 2,260,701 accepted-file rows (151 columns); 27,648,741 rejected-file rows (9 columns).

## Verification status
- Previously recorded test run: 52 tests passed when excluding `tests/test_streamlit_app.py`. The Streamlit AppTest stalled during Windows Python 3.13 asyncio Proactor socket-pair setup. The suite was not rerun during report generation.
- A prior live Ollama check and general response were observed; the configured default is `llama3.1:8b` at `http://localhost:11434`. Ollama was not rechecked during report generation. A direct project-evidence prompt previously returned an unavailable-artifact response when the artifact was not loaded, so complete live multi-tool verification is not claimed.
- Streamlit HTTP 200 was observed previously. No complete browser visual inspection across pages/themes/viewports was performed for this report.
- Dataset manager/training tests use synthetic temporary fixtures. No newly discovered public dataset download or production-scale multi-source retraining is claimed.
- No current-application screenshots were available and none were fabricated or reused from an earlier UI version. The included images are labeled technical diagrams.

## Limitations
1. The active model learns source-file membership between unmatched issued-loan records and rejected applications. It does not infer an individual lender decision, genuine eligibility, creditworthiness, or default probability.
2. The balanced sample changes class prevalence, and random splitting does not establish external or temporal validity.
3. Feature distributions and measurement differ across the sources; the strong employment-length importance may capture source artifacts.
4. Scores are uncalibrated for any real-world application population. SHAP and permutation importance are not causal or fairness evidence.
5. LLM output can be wrong; bounded evidence tools reduce unsupported application-specific answers but do not eliminate generation errors. Local model latency and memory needs depend on hardware.
6. Current verification does not constitute full browser/UI, live multi-tool, external-data licensing, fairness, or production-readiness validation.

## Missing manual information
No signatures, submission date, institutional certificate text, or other declarations were supplied. Complete any university-required front matter through the official academic process.

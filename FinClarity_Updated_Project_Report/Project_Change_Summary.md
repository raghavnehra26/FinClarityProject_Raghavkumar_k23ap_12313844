# FinClarity project change summary

This report was updated against the current project tree and saved evaluation artifacts, with a direct comparison to the prior three-page project report. The separate patent disclosure is not used as a report template or technical authority.

## Differences from the previous report
- The earlier report documented the four-feature accepted-file versus rejected-file experiment, a persisted pipeline, SHAP/global importance, synthetic examples, and a deterministic local explanation agent.
- It mentioned optional Microsoft Foundry and Azure Content Safety calls. Those are not present in the current dependency set/current agent implementation and are not described as current functionality.
- The current code integrates a local pretrained Llama model through Ollama, with general/project-aware routing, allow-listed Python evidence tools, bounded calls, session context, and connection/error handling. Current Ollama service verification failed during report preparation; the configured model is `llama3.1:8b`.
- Newer application code adds multi-CSV registration, explicit accepted/rejected assignment, schema mapping, provenance/checksum tracking, curated dataset discovery, registry-based multi-source training, model version history, activation, and rollback.
- The active classifier continues to use the same four common inputs and source-membership target. The distinct historical repayment workflow is kept separate.

## Updated factual evaluation
The saved artifact reports 80,000 balanced training-workflow samples and a 12,000-row held-out balanced test set. Random Forest validation average precision is 0.9521 versus 0.9363 for Logistic Regression. Test average precision is 0.9509, ROC-AUC 0.9570, precision 0.9140, recall 0.8945, F1 0.9041, and Brier score 0.0730. These measure source-file membership on the historical sample only.

## Important qualification
The classifier is not a real loan eligibility, creditworthiness, approval, rejection, or default model. No values or claims from the separate patent package were treated as evidence of implementation.

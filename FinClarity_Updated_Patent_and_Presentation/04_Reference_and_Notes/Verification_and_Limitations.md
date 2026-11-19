# Verification and limitations

## Code and runtime verification snapshot

- Python module compile check passed for the application, agent, dataset, model, and explainability modules.
- 52 automated tests passed with `tests/test_streamlit_app.py` excluded.
- Streamlit AppTest stalled during Python 3.13 Windows Proactor event-loop socketpair initialization. It did not pass.
- Streamlit server returned HTTP 200 on localhost. This verifies server startup/page delivery, not every interactive page or a visual audit.
- Ollama endpoint responded at `http://localhost:11434`; tags `llama3.1:8b` and `llama3.2:3b` were listed, with Llama 3.1 8B resolving as the configured default.
- A live general Llama 3.1 8B question returned in about 31 seconds on the test laptop.
- A live evidence-routed performance question produced an unavailable-artifact response because the direct request context had no loaded artifact. No claim is made that this specific request answered with live evaluation metrics.
- Browser-based UI inspection was not completed.
- Dataset discovery/download tests are mocked; no remote dataset was downloaded. Multi-dataset training tests use small synthetic fixtures.

## Core limitation

The active classifier predicts historical accepted-source vs rejected-source membership from unmatched datasets. It is not a real loan approval, eligibility, creditworthiness, or default model. The metrics are for a balanced held-out sample from those historical sources and cannot support lending decisions.

## Manual completion before institutional filing

1. Confirm contribution-based inventor list. Prior FinClarity form and deck name both Raghav Kumar (UID 12313844) and Tejesh Balaga (UID 12323707); the current request explicitly details Raghav.
2. Verify current contact details and addresses, then collect original signatures from all inventors.
3. Confirm whether class presentations, the supplied prior report/deck, project repository, or any demos were publicly available; record dates and URLs.
4. Confirm collaboration/MOU, software and data licenses, any third-party rights, and current commercialization contacts.
5. Ask the LPU IPR Cell to complete prior-art review, ownership/inventorship assessment, and filing-route advice.

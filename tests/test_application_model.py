import numpy as np
import pandas as pd
import pytest

from explainability import explain_prediction
import train_application_model as training


def test_reservoir_sampling_replaces_earlier_rows_across_chunks(tmp_path, monkeypatch):
    source = tmp_path / "accepted.csv"
    pd.DataFrame(
        {
            "loan_amnt": range(1000, 1030),
            "dti": [10] * 30,
            "emp_length": ["5 years"] * 30,
            "purpose": ["car"] * 30,
            "loan_status": ["Fully Paid"] * 30,
        }
    ).to_csv(source, index=False)

    class ControlledRandom:
        def __init__(self):
            self.chunk = 0

        def random(self, size):
            self.chunk += 1
            priorities = (0.1, 0.9, 0.0)[self.chunk - 1]
            return np.full(size, priorities)

    monkeypatch.setattr(
        training.np.random, "default_rng", lambda _seed: ControlledRandom()
    )

    sample, report = training._reservoir_sample(
        source, "accepted", maximum=10, chunksize=10
    )

    assert len(sample) == 10
    assert sample["loan_amnt"].min() >= 1020
    assert report["total_rows"] == 30
    assert report["sampled_rows"] == 10
    assert report["column_sample_rows"] == 30
    assert report["column_sample_dtypes"]["loan_amnt"] == "int64"


def test_rejected_fields_normalize_to_shared_features():
    chunk = pd.DataFrame(
        {
            "Amount Requested": [12_000, 20_000],
            "Debt-To-Income Ratio": ["18.5%", "3626400"],
            "Employment Length": ["2 years", "< 1 year"],
            "Loan Title": ["Credit Card Consolidation", "car loan"],
        }
    )

    normalized = training._normalize_chunk(chunk, "rejected")

    assert normalized["loan_amnt"].tolist() == [12_000, 20_000]
    assert normalized["dti"].iloc[0] == 18.5
    assert pd.isna(normalized["dti"].iloc[1])
    assert normalized["emp_length"].tolist() == ["2 years", "< 1 year"]
    assert normalized["purpose"].tolist() == ["Debt consolidation", "Vehicle"]


def test_training_saves_honest_target_metrics_and_shap(tmp_path):
    accepted_path = tmp_path / "accepted.csv"
    rejected_path = tmp_path / "rejected.csv"
    amounts = np.arange(1_000, 1_160)
    dti = np.linspace(2, 45, len(amounts))
    dti[-1] = 3626400
    shared = {
        "dti": dti,
        "emp_length": ["5 years", "10+ years"] * 80,
        "purpose": ["debt_consolidation", "credit_card"] * 80,
    }
    pd.DataFrame(
        {
            "loan_amnt": amounts,
            **shared,
            "loan_status": ["Fully Paid"] * len(amounts),
        }
    ).to_csv(accepted_path, index=False)
    pd.DataFrame(
        {
            "Amount Requested": amounts + 500,
            "Debt-To-Income Ratio": shared["dti"],
            "Employment Length": shared["emp_length"],
            "Loan Title": shared["purpose"],
        }
    ).to_csv(rejected_path, index=False)

    model_path = tmp_path / "model.joblib"
    evaluation = training.train_application_model(
        accepted_path,
        rejected_path,
        model_path,
        tmp_path / "evaluation.json",
        tmp_path / "dataset.json",
        maximum_per_class=160,
        chunksize=40,
    )
    import joblib

    artifact = joblib.load(model_path)
    applicant = {
        "loan_amnt": 8_000,
        "dti": 20.0,
        "emp_length": "5 years",
        "purpose": "Debt consolidation",
    }
    explanation = explain_prediction(artifact, applicant)

    assert artifact["metadata"]["not_approval_prediction"] is True
    assert artifact["metadata"]["positive_score_name"] == (
        "accepted-file membership score"
    )
    assert evaluation["test"]["confusion_matrix"]["labels"] == [
        "Rejected-file application record",
        "Accepted-file loan record",
    ]
    assert explanation["prediction_label"] in {
        "Accepted-file loan record",
        "Rejected-file application record",
    }
    assert explanation["model_score"] == pytest.approx(
        artifact["pipeline"].predict_proba(pd.DataFrame([applicant]))[0, 1]
    )
    assert explanation["base_value"] + explanation["contribution_sum"] == pytest.approx(
        explanation["model_score"], abs=0.02
    )

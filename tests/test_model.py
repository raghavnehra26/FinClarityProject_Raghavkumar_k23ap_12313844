import numpy as np
import pandas as pd
import pytest

from explainability import explain_prediction, make_synthetic_examples
from train_model import prepare_training_data, train_model


def _small_fixture() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    rows = 160
    loan_amount = generator.integers(1_000, 30_000, size=rows)
    annual_income = generator.integers(20_000, 150_000, size=rows)
    purpose = generator.choice(["debt_consolidation", "home_improvement"], rows)
    risk = (loan_amount > 15_000).astype(int)
    return pd.DataFrame(
        {
            "bad_loan": np.where(risk, 1, 2),
            "loan_amnt": loan_amount,
            "annual_inc": annual_income,
            "purpose": purpose,
            "total_pymnt": generator.random(rows) * 10_000,
            "member_id": np.arange(rows),
        }
    )


def test_training_uses_only_allowlisted_features_and_saves_real_outputs(tmp_path):
    source = tmp_path / "source.csv"
    model_path = tmp_path / "model.joblib"
    evaluation_path = tmp_path / "evaluation.json"
    inspection_path = tmp_path / "inspection.json"
    frame = _small_fixture()
    frame.to_csv(source, index=False)

    evaluation = train_model(
        source,
        model_path,
        evaluation_path,
        inspection_path,
    )

    assert model_path.is_file()
    assert evaluation_path.is_file()
    assert inspection_path.is_file()
    assert "total_pymnt" not in evaluation["feature_columns"]
    assert "member_id" not in evaluation["feature_columns"]
    assert evaluation["not_approval_prediction"] is True
    assert evaluation["rows_used"] == len(frame)
    assert evaluation["test"]["test_rows"] > 0
    assert 0 <= evaluation["test"]["metrics"]["precision"] <= 1
    assert 0 <= evaluation["test"]["metrics"]["brier_score"] <= 1


def test_shap_explanation_matches_the_saved_pipeline_score(tmp_path):
    source = tmp_path / "source.csv"
    model_path = tmp_path / "model.joblib"
    frame = _small_fixture()
    frame.to_csv(source, index=False)
    train_model(
        source,
        model_path,
        tmp_path / "evaluation.json",
        tmp_path / "inspection.json",
    )
    import joblib

    artifact = joblib.load(model_path)
    applicant = {
        "loan_amnt": 12_000,
        "annual_inc": 60_000,
        "purpose": "debt_consolidation",
    }

    explanation = explain_prediction(artifact, applicant)
    assert explanation["model_score"] == pytest.approx(
        artifact["pipeline"].predict_proba(pd.DataFrame([applicant]))[0, 1]
    )
    assert explanation["base_value"] + explanation["contribution_sum"] == pytest.approx(
        explanation["model_score"], abs=0.02
    )
    assert all(
        np.isfinite(item["contribution"])
        for item in explanation["explanations"]
    )


def test_supported_targets_do_not_infer_approval_from_issued_loan_rows():
    frame = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current", "Late"],
            "loan_amnt": [1_000, 2_000, 3_000, 4_000],
            "purpose": ["car", "car", "home", "home"],
        }
    )

    features, labels, details = prepare_training_data(frame)

    assert len(features) == 2
    assert labels.tolist() == [0, 1]
    assert details["target_column"] == "loan_status"
    assert "not predict loan approval" in details["target_definition"]


def test_bad_loan_zero_one_encoding_maps_one_to_default():
    frame = pd.DataFrame(
        {
            "bad_loan": [0, 1, 0, 1],
            "loan_amnt": [1_000, 2_000, 3_000, 4_000],
            "purpose": ["car", "car", "home", "home"],
        }
    )

    _, labels, details = prepare_training_data(frame)

    assert labels.tolist() == [0, 1, 0, 1]
    assert "0=did not default, 1=default" in details["target_definition"]


def test_training_drops_features_that_are_completely_missing():
    frame = pd.DataFrame(
        {
            "bad_loan": [0, 1, 0, 1],
            "loan_amnt": [1_000, 2_000, 3_000, 4_000],
            "annual_inc": [None, None, None, None],
            "purpose": ["car", "car", "home", "home"],
        }
    )

    features, _, details = prepare_training_data(frame)

    assert "annual_inc" not in features
    assert details["excluded_empty_features"] == ["annual_inc"]


def test_unknown_target_is_not_guessed():
    frame = pd.DataFrame(
        {
            "application_status": ["approved", "rejected"],
            "loan_amnt": [1_000, 2_000],
            "purpose": ["car", "home"],
        }
    )
    with pytest.raises(ValueError, match="No supported target"):
        prepare_training_data(frame)


def test_synthetic_examples_are_generated_from_a_trained_model(tmp_path):
    source = tmp_path / "source.csv"
    model_path = tmp_path / "model.joblib"
    _small_fixture().to_csv(source, index=False)
    train_model(
        source,
        model_path,
        tmp_path / "evaluation.json",
        tmp_path / "inspection.json",
    )
    import joblib

    result = make_synthetic_examples(joblib.load(model_path), candidate_count=12)
    assert result["examples"]
    assert len(result["examples"]) == 2
    scores = [item["result"]["model_score"] for item in result["examples"]]
    assert scores[0] <= scores[1]

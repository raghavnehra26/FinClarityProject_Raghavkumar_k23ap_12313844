import joblib
import numpy as np
import pandas as pd

from dataset_manager import compatible_features
from multi_dataset_training import activate_model_version, train_from_registry


def _entry(tmp_path, name, label, frame, mapping):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return {
        "dataset_id": ("a" if label == "accepted" else "b") * 16,
        "name": name,
        "label": label,
        "stored_file": str(path),
        "mapping": mapping,
        "sha256": label * 64,
        "source_url": "",
        "license": "fixture",
    }


def test_multiple_schemas_train_versioned_reusable_pipeline_and_rollback(tmp_path):
    n = 140
    accepted = pd.DataFrame({
        "amount_requested": np.linspace(4000, 18000, n),
        "debt_to_income_ratio": np.linspace(5, 30, n),
        "years_employed": ["5 years", "10+ years"] * (n // 2),
        "purpose": ["car", "home improvement"] * (n // 2),
        "annual_income": np.linspace(30000, 100000, n),
    })
    rejected = pd.DataFrame({
        "loan_amnt": np.linspace(1000, 25000, n),
        "dti": np.linspace(10, 50, n),
        "emp_length": ["< 1 year", "2 years"] * (n // 2),
        "loan_title": ["medical", "credit card"] * (n // 2),
        "annual_inc": np.linspace(20000, 70000, n),
    })
    accepted_map = {"loan_amnt": "amount_requested", "dti": "debt_to_income_ratio", "emp_length": "years_employed", "purpose": "purpose", "annual_inc": "annual_income"}
    rejected_map = {"loan_amnt": "loan_amnt", "dti": "dti", "emp_length": "emp_length", "purpose": "loan_title", "annual_inc": "annual_inc"}
    entries = [
        _entry(tmp_path, "accepted.csv", "accepted", accepted, accepted_map),
        _entry(tmp_path, "rejected.csv", "rejected", rejected, rejected_map),
    ]
    features = compatible_features(entries)
    assert features == ["loan_amnt", "annual_inc", "dti", "emp_length", "purpose"]
    report = train_from_registry(
        entries, features, maximum_per_class=120, chunksize=40,
        model_path=tmp_path / "active.joblib",
        versions_dir=tmp_path / "versions", report_dir=tmp_path / "reports",
        model_kind="logistic_regression",
    )
    artifact = joblib.load(tmp_path / "active.joblib")
    assert report["not_approval_prediction"]
    assert len(report["datasets"]) == 2
    assert artifact["metadata"]["feature_columns"] == features
    test_row = pd.DataFrame([{name: artifact["metadata"]["input_defaults"][name] for name in features}])
    assert artifact["pipeline"].predict(test_row).shape == (1,)
    activate_model_version(report["version"], versions_dir=tmp_path / "versions", model_path=tmp_path / "rolled_back.joblib")
    assert joblib.load(tmp_path / "rolled_back.joblib")["metadata"]["version"] == report["version"]

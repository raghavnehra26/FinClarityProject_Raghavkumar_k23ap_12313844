import pandas as pd
import pytest

from inspect_dataset import inspect_dataset


def test_inspection_reports_schema_quality_and_target(tmp_path):
    fixture_path = tmp_path / "schema_fixture.csv"
    pd.DataFrame(
        {
            "bad_loan": [1, 2, 1],
            "loan_amnt": [1000, None, 3000],
        }
    ).to_csv(fixture_path, index=False)

    report = inspect_dataset(fixture_path)

    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["duplicate_rows"] == 0
    assert report["target_column"] == "bad_loan"
    assert report["target_distribution"] == [
        {"value": "1", "count": 2},
        {"value": "2", "count": 1},
    ]
    assert report["column_report"][1]["missing_count"] == 1


def test_inspection_requires_a_real_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="loanStats_train.csv"):
        inspect_dataset(tmp_path / "missing.csv")

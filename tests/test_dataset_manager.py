import json

import pandas as pd
import pytest

from dataset_manager import inspect_csv_bytes, register_uploaded_csv, remove_dataset


def _csv_bytes():
    frame = pd.DataFrame({
        "loan_amount": [1000, 2000, 2000],
        "debt_to_income": [10.0, None, None],
        "years_employed": ["3 years", "5 years", "5 years"],
        "loan_purpose": ["car", "medical", "medical"],
        "extra_field": ["a", "b", "b"],
    })
    return frame.to_csv(index=False).encode()


def test_csv_inspection_detects_aliases_missing_values_and_duplicate_rows():
    report = inspect_csv_bytes(_csv_bytes())
    assert report["row_count"] == 3
    assert report["detected_mapping"]["loan_amnt"] == "loan_amount"
    assert report["detected_mapping"]["dti"] == "debt_to_income"
    assert report["missing_counts"]["debt_to_income"] == 2
    assert report["duplicate_rows"] == 1
    assert "extra_field" in report["unmapped_columns"]


def test_register_requires_an_explicit_source_label_and_deduplicates_checksum(tmp_path):
    data = _csv_bytes()
    with pytest.raises(ValueError, match="explicitly"):
        register_uploaded_csv("rows.csv", data, "unknown", registry_path=tmp_path / "registry.json", storage_root=tmp_path / "managed")
    first = register_uploaded_csv("rows.csv", data, "accepted", registry_path=tmp_path / "registry.json", storage_root=tmp_path / "managed")
    again = register_uploaded_csv("renamed.csv", data, "rejected", registry_path=tmp_path / "registry.json", storage_root=tmp_path / "managed")
    assert first["duplicate_file"] is False
    assert again["duplicate_file"] is True
    assert first["label"] == "accepted"
    assert json.loads((tmp_path / "registry.json").read_text())[0]["sha256"] == first["sha256"]
    assert remove_dataset(first["dataset_id"], registry_path=tmp_path / "registry.json", storage_root=tmp_path / "managed")


@pytest.mark.parametrize("contents", [b"", b"a,b\n", b"not,a,csv\n\"broken"])
def test_empty_header_only_and_malformed_csv_are_rejected(contents):
    with pytest.raises(ValueError):
        inspect_csv_bytes(contents)

import json

import pytest

from agent_tools import build_tools, execute_tool


def test_agent_tools_return_only_existing_model_evidence(tmp_path, monkeypatch):
    report = {
        "total_rows": 99,
        "column_count": 4,
        "columns": ["loan_amnt", "loan_status"],
        "file": "C:/private/path.csv",
        "accepted_loan_status_distribution": {"Fully Paid": 90},
    }
    (tmp_path / "dataset.json").write_text(json.dumps(report), encoding="utf-8")
    import agent_tools

    result = {
        "prediction_label": "Accepted-file record",
        "target": "source membership",
        "model_score": 0.8,
        "score_warning": "uncalibrated",
        "applicant_inputs": {"loan_amnt": 1000},
        "base_value": 0.5,
        "contribution_sum": 0.3,
        "explanations": [{"feature": "loan_amnt", "contribution": 0.3}],
    }
    artifact = {
        "metadata": {"target_definition": "accepted/rejected source membership"},
        "evaluation": {
            "global_feature_importance": [{"label": "Amount", "importance_mean": 0.1}],
            "validation_average_precision": {"Random Forest": 0.92},
            "test": {"metrics": {"f1": 0.9}},
        },
    }
    tools = build_tools(result, artifact)
    monkeypatch.setattr(agent_tools, "DATASET_REPORT_PATH", tmp_path / "dataset.json")
    assert tools["get_applicant_assessment"]()["model_score"] == 0.8
    assert tools["get_shap_explanation"]()["features"][0]["contribution"] == 0.3
    assert tools["get_global_feature_importance"]()["features"][0]["label"] == "Amount"
    assert tools["get_model_performance"]()["test_metrics"] == {"f1": 0.9}
    assert tools["get_model_performance"]()["validation_average_precision"] == {"Random Forest": 0.92}
    dataset = tools["get_dataset_summary"]()
    assert dataset["available"]
    assert "C:/private/path.csv" not in str(dataset)
    assert tools["get_responsible_ai_guidance"]()["tested_fairness"] is False
    assert "debt_to_income" in tools["get_financial_knowledge"]()["concepts"]


def test_tools_reject_unapproved_names_and_arguments():
    tools = build_tools(None, None)
    with pytest.raises(ValueError, match="Unapproved"):
        execute_tool("run_python", {}, tools)
    with pytest.raises(ValueError, match="does not accept arguments"):
        execute_tool("get_applicant_assessment", {"code": "print(1)"}, tools)
    with pytest.raises(ValueError, match="does not accept arguments"):
        execute_tool("get_applicant_assessment", [], tools)


def test_performance_tool_accepts_only_allowlisted_metric_filters():
    artifact = {"evaluation": {"test": {"metrics": {"f1": 0.8, "recall": 0.7}}}}
    tools = build_tools(None, artifact)
    data = execute_tool("get_model_performance", {"include_metrics": ["f1"]}, tools)
    assert data["test_metrics"] == {"f1": 0.8}
    with pytest.raises(ValueError, match="Invalid metric list"):
        execute_tool("get_model_performance", {"include_metrics": ["approval_probability"]}, tools)


def test_missing_current_artifacts_are_reported_without_fabricating():
    tools = build_tools(None, None)
    assert tools["get_applicant_assessment"]()["available"] is False
    assert tools["get_shap_explanation"]()["available"] is False
    assert tools["get_model_performance"]()["available"] is False


def test_registry_tools_return_actual_dataset_counts_schema_provenance_and_search(tmp_path, monkeypatch):
    import agent_tools

    registry = [{
        "dataset_id": "abc123", "name": "accepted.csv", "label": "accepted",
        "label_definition": "user-confirmed source label", "row_count": 12,
        "column_count": 3, "columns": ["loan_amnt", "dti", "loan_status"],
        "mapping": {"loan_amnt": "loan_amnt", "dti": "dti"},
        "unmapped_columns": ["loan_status"], "missing_counts": {"dti": 2},
        "duplicate_rows": 1, "duplicate_check_complete": True,
        "source_url": "https://example.invalid/dataset", "license": "CC BY",
        "imported_at": "2026-01-01T00:00:00+00:00", "sha256": "a" * 64,
        "file_size_bytes": 1024,
    }, {"dataset_id": "def456", "name": "rejected.csv", "label": "rejected", "row_count": 20, "columns": ["loan_amount"], "mapping": {"loan_amnt": "loan_amount"}}]
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(agent_tools, "DATASET_REGISTRY_PATH", registry_path)
    tools = build_tools(None, None)
    stats = execute_tool("get_dataset_statistics", {}, tools)
    assert stats["accepted_count"] == 1
    assert stats["rejected_count"] == 1
    schema = execute_tool("get_dataset_schema", {}, tools)
    assert schema["datasets"][0]["canonical_mapping"]["dti"] == "dti"
    search = execute_tool("search_dataset", {"query": "loan_status"}, tools)
    assert search["match_count"] == 1
    assert "applicant-level rows" in search["scope"]
    provenance = execute_tool("get_dataset_provenance", {}, tools)
    assert provenance["datasets"][0]["license"] == "CC BY"

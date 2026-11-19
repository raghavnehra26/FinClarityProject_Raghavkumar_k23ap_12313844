"""Explicit, read-only FinClarity evidence tools exposed to the chat agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from config import DATASET_REGISTRY_PATH, DATASET_REPORT_PATH, MODEL_VERSIONS_DIR


def _assessment(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {"available": False, "message": "No current assessment has been run."}
    return {
        "available": True,
        "target": result.get("target"),
        "predicted_historical_class": result.get("prediction_label"),
        "prediction_label": result.get("prediction_label"),
        "prediction": result.get("prediction"),
        "positive_class_label": result.get("positive_class_label"),
        "negative_class_label": result.get("negative_class_label"),
        "positive_score_name": result.get("positive_score_name"),
        "model_score": result.get("model_score"),
        "score_warning": result.get("score_warning"),
        "input_features": result.get("applicant_inputs", {}),
        "applicant_inputs": result.get("applicant_inputs", {}),
        "explanations": result.get("explanations", []),
        "interpretation": (
            "This is only the classifier's historical dataset membership/outcome "
            "label. It is not actual loan eligibility, creditworthiness, default "
            "probability, or a lending decision."
        ),
    }


def _shap(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {"available": False, "message": "No current SHAP explanation is available."}
    return {
        "available": True,
        "target": result.get("target"),
        "score_name": result.get("positive_score_name"),
        "base_value": result.get("base_value"),
        "contribution_sum": result.get("contribution_sum"),
        "features": result.get("explanations", []),
        "meaning": "SHAP values describe model behavior relative to its reference profile; they do not establish causation or fairness.",
    }


def _importance(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not artifact:
        return {"available": False, "message": "No trained model artifact is loaded."}
    rows = artifact.get("evaluation", {}).get("global_feature_importance")
    if rows is None:
        return {"available": False, "message": "Permutation importance is absent from the model artifact."}
    return {
        "available": True,
        "target": artifact.get("metadata", {}).get("target_definition"),
        "method": "Held-out permutation importance; score change is average precision.",
        "features": sorted(rows, key=lambda row: row.get("importance_mean", 0), reverse=True),
        "caveat": "Importance is not causation and is not a fairness assessment.",
    }


def _metrics(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not artifact:
        return {"available": False, "message": "No trained model artifact is loaded."}
    evaluation = artifact.get("evaluation", {})
    test = evaluation.get("test", {})
    return {
        "available": bool(test.get("metrics")),
        "target": evaluation.get("target_definition"),
        "selected_model": evaluation.get("selected_model"),
        "split_method": evaluation.get("split_method"),
        "rows_used": evaluation.get("rows_used"),
        "validation_average_precision": evaluation.get("validation_average_precision"),
        "test_metrics": test.get("metrics"),
        "limitations": evaluation.get("limitations", []),
        "warning": test.get("probability_warning"),
    }


def _dataset(artifact: dict[str, Any] | None = None, path: Path = DATASET_REPORT_PATH) -> dict[str, Any]:
    registry_summary = _registered_dataset_summary()
    if not Path(path).is_file():
        return {"available": registry_summary["available"], "message": "No legacy dataset report is available; returning the local registry only.", "registered_datasets": registry_summary}
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "message": "The saved dataset report could not be read."}
    # Remove filesystem paths and raw applicant-level data from the model context.
    if "accepted" in report and "rejected" in report:
        sources = {}
        for name in ("accepted", "rejected"):
            source = report[name]
            sources[name] = {
                key: source.get(key) for key in (
                    "total_rows", "column_count", "rows_with_valid_loan_amount",
                    "columns", "column_sample_dtypes", "missing_counts_selected_columns",
                    "invalid_numeric_counts", "accepted_loan_status_distribution",
                    "potentially_sensitive_or_identifying_columns",
                ) if key in source
            }
        metadata = (artifact or {}).get("metadata", {})
        legacy = {
            "available": True,
            "target_definition": report.get("target_definition"),
            "sources": sources,
            "model_feature_summaries": metadata.get("demo_distributions", {}),
            "feature_labels": {
                name: spec.get("label")
                for name, spec in metadata.get("feature_schema", {}).items()
            },
            "feature_summary_note": "Numeric ranges are training-sample 10th–90th percentiles; category weights are training-sample counts, not current applicant population estimates.",
        }
        legacy["registered_datasets"] = registry_summary
        return legacy
    legacy = {
        "available": True,
        "rows": report.get("rows"),
        "columns": report.get("columns"),
        "target_definition": report.get("target_definition"),
        "target_distribution": report.get("target_distribution"),
        "column_report": report.get("column_report"),
        "sensitive_or_identifying_columns": report.get("sensitive_or_identifying_columns"),
        "warning": report.get("approval_target_warning"),
        "model_feature_summaries": (artifact or {}).get("metadata", {}).get("demo_distributions", {}),
    }
    legacy["registered_datasets"] = registry_summary
    return legacy


def _registered_dataset_summary() -> dict[str, Any]:
    try:
        from dataset_manager import load_registry
        entries = load_registry(DATASET_REGISTRY_PATH)
    except (OSError, ValueError):
        return {"available": False, "message": "The local dataset registry could not be read."}
    if entries:
        return {
        "available": bool(entries), "count": len(entries),
        "accepted_count": sum(row.get("label") == "accepted" for row in entries),
        "rejected_count": sum(row.get("label") == "rejected" for row in entries),
        "accepted_record_count": sum(int(row.get("row_count", 0) or 0) for row in entries if row.get("label") == "accepted"),
        "rejected_record_count": sum(int(row.get("row_count", 0) or 0) for row in entries if row.get("label") == "rejected"),
        "datasets": [{key: row.get(key) for key in ("dataset_id", "name", "label", "row_count", "column_count", "columns", "missing_counts", "duplicate_rows", "source_url", "license", "imported_at", "mapping") if key in row} for row in entries],
        }
    try:
        legacy = json.loads(Path(DATASET_REPORT_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        legacy = {}
    if isinstance(legacy, dict) and "accepted" in legacy and "rejected" in legacy:
        return {"available": True, "count": 0, "accepted_count": 0, "rejected_count": 0,
                "accepted_record_count": legacy["accepted"].get("total_rows"),
                "rejected_record_count": legacy["rejected"].get("total_rows"), "registered": False,
                "message": "Counts are from the saved legacy dataset inspection report; files are not in the user-managed registry.",
                "datasets": [{"name": legacy[name].get("file", name), "label": name, "row_count": legacy[name].get("total_rows"), "column_count": legacy[name].get("column_count"), "columns": legacy[name].get("columns", [])} for name in ("accepted", "rejected")]}
    return {"available": False, "count": 0, "accepted_count": 0, "rejected_count": 0,
            "accepted_record_count": None, "rejected_record_count": None, "datasets": [],
            "message": "No saved or registered dataset summaries are available."}


def _dataset_schema(_: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = _registered_dataset_summary()
    return {"available": summary["available"], "datasets": [{"dataset_id": row.get("dataset_id"), "name": row.get("name"), "columns": row.get("columns", []), "canonical_mapping": row.get("mapping", {}), "unmapped_columns": row.get("unmapped_columns", [])} for row in summary.get("datasets", [])], "message": summary.get("message")}


def _training_history(_: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(MODEL_VERSIONS_DIR) / "history.json"
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"available": False, "message": "No versioned multi-dataset training runs are recorded yet.", "versions": []}
    except (OSError, json.JSONDecodeError):
        return {"available": False, "message": "Training history could not be read.", "versions": []}
    return {"available": bool(history), "versions": history}


def _search_dataset(arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 100:
        raise ValueError("Provide a non-empty dataset metadata search term (up to 100 characters).")
    needle = query.casefold().strip()
    summary = _registered_dataset_summary()
    matches = []
    for dataset in summary.get("datasets", []):
        searchable = " ".join([str(dataset.get("name", "")), str(dataset.get("label", "")), " ".join(dataset.get("columns", [])), str(dataset.get("source_url", "")), json.dumps(dataset.get("mapping", {}))]).casefold()
        if needle in searchable:
            matches.append({key: dataset.get(key) for key in ("dataset_id", "name", "label", "row_count", "columns", "source_url", "license")})
    return {"available": summary["available"], "query": query, "match_count": len(matches), "matches": matches, "scope": "Search is limited to registry metadata and column names; it does not scan applicant-level rows."}


def _compare_models(_: dict[str, Any] | None = None) -> dict[str, Any]:
    history = _training_history()
    versions = history.get("versions", [])
    return {"available": bool(versions), "versions": [{"version": item.get("version"), "classifier": item.get("selected_model"), "validation_average_precision": item.get("validation_average_precision"), "test_metrics": item.get("test_metrics"), "features": item.get("features"), "dataset_ids": item.get("dataset_ids")} for item in versions], "comparison_note": "Compare only runs with compatible targets and data sources; a metric increase does not establish genuine approval performance."}


def _responsible(_: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "tested_fairness": False,
        "known_limitations": [
            "This classifier reflects historical dataset membership/outcomes, not a real lending decision.",
            "SHAP and permutation importance describe associations/model behavior, not causal effects.",
            "Historical data can contain selection bias, label noise, temporal effects, and proxies for protected traits.",
            "No group fairness evaluation is available unless explicitly present in the saved evaluation report.",
        ],
        "practice": "Do not use this educational prototype for lending, eligibility, creditworthiness, or individual decisions.",
    }


FINANCIAL_CONCEPTS = {
            "loan": "A loan is money borrowed under an agreement to repay the principal, usually with interest and possibly fees, according to set terms.",
            "debt_to_income": "Monthly debt obligations divided by gross monthly income, commonly expressed as a percentage; definitions vary by lender.",
            "loan_amount": "The principal requested or borrowed; fees and interest affect total repayment.",
            "interest_rate": "The cost of borrowing, expressed as a rate applied to the outstanding principal; the exact cost also depends on fees and repayment timing.",
            "monthly_payment": "A scheduled amount paid periodically toward a debt. It may include principal, interest, and fees, depending on the contract.",
            "principal": "The original amount borrowed, before interest and fees.",
            "employment_length": "A description of time in employment; its meaning and treatment vary by lender and product.",
            "loan_purpose": "The borrower's stated intended use; category names do not prove actual use.",
            "credit_risk": "A lender's estimate of potential repayment risk using its own data, policy, and legally applicable rules.",
            "loan_terms": "The contract may specify principal, interest rate, term, fees, payment schedule, and other obligations.",
}


def _financial(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    concept = arguments.get("concept") or arguments.get("knowledge_type")
    if concept:
        return {"available": concept in FINANCIAL_CONCEPTS,
                "scope": "General financial education only; not a statement about this model or an individual's eligibility.",
                "concept": concept,
                "definition": FINANCIAL_CONCEPTS.get(concept)}
    return {"available": True, "scope": "General financial education only; not a statement about this model or an individual's eligibility.", "concepts": FINANCIAL_CONCEPTS}


def _project(_: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"available": True, "name": "FinClarity", "purpose": "Educational explainable ML demonstration for historical financial dataset records.",
            "architecture": ["Streamlit interface", "scikit-learn classifier", "SHAP local explanations", "held-out permutation importance", "local Ollama chat with approved Python evidence tools"],
            "limitation": "The classifier distinguishes historical dataset source/outcome labels. It is not a loan eligibility, creditworthiness, approval, rejection, or default prediction system."}


def build_tools(result: dict[str, Any] | None, artifact: dict[str, Any] | None) -> dict[str, Callable[..., dict[str, Any]]]:
    """Return the only callable functions available to the language model."""
    tools = {
        "get_current_assessment": lambda **_: _assessment(result),
        "get_current_shap_explanation": lambda **_: _shap(result),
        "get_global_feature_importance": lambda: _importance(artifact),
        "get_model_performance": lambda: _metrics(artifact),
        "get_dataset_summary": lambda: _dataset(artifact),
        "get_dataset_schema": lambda **_: _dataset_schema(),
        "get_dataset_statistics": lambda **_: _registered_dataset_summary(),
        "get_training_history": lambda **_: _training_history(),
        "get_active_model_info": lambda **_: {"available": bool(artifact), "version": (artifact or {}).get("metadata", {}).get("version", "legacy"), "selected_model": (artifact or {}).get("metadata", {}).get("selected_model"), "feature_columns": (artifact or {}).get("metadata", {}).get("feature_columns", []), "prediction_task": (artifact or {}).get("metadata", {}).get("prediction_task"), "target_definition": (artifact or {}).get("metadata", {}).get("target_definition")},
        "get_data_quality_report": lambda **_: {"available": bool(_registered_dataset_summary().get("datasets")), "datasets": [{"name": row.get("name"), "row_count": row.get("row_count"), "missing_counts": row.get("missing_counts"), "duplicate_rows": row.get("duplicate_rows"), "duplicate_check_complete": row.get("duplicate_check_complete"), "unmapped_columns": row.get("unmapped_columns", [])} for row in _registered_dataset_summary().get("datasets", [])]},
        "get_dataset_provenance": lambda **_: {"available": bool(_registered_dataset_summary().get("datasets")), "datasets": [{key: row.get(key) for key in ("dataset_id", "name", "label", "label_definition", "source_url", "license", "imported_at", "sha256", "file_size_bytes") if key in row} for row in _registered_dataset_summary().get("datasets", [])]},
        "search_dataset": lambda **args: _search_dataset(args),
        "compare_models": lambda **_: _compare_models(),
        "get_responsible_ai_information": lambda **_: _responsible(result),
        "get_project_information": lambda **_: _project(),
        "explain_financial_concept": lambda **args: _financial(args),
    }
    tools.update({"get_applicant_assessment": tools["get_current_assessment"], "get_shap_explanation": tools["get_current_shap_explanation"], "get_responsible_ai_guidance": tools["get_responsible_ai_information"], "get_financial_knowledge": tools["explain_financial_concept"]})
    tools.update({"get_local_shap_explanation": tools["get_current_shap_explanation"], "get_model_metrics": tools["get_model_performance"], "run_applicant_assessment": tools["get_current_assessment"]})
    return tools


def execute_tool(name: str, arguments: dict[str, Any], tools: dict[str, Callable[[], dict[str, Any]]]) -> dict[str, Any]:
    """Call a registered read-only tool after validating its small argument schema."""
    def outcome(data: dict[str, Any]) -> dict[str, Any]:
        data.setdefault("status", "success" if data.get("available", True) else "empty")
        return data
    if name not in tools:
        raise ValueError("Unapproved agent tool.")
    if not isinstance(arguments, dict):
        raise ValueError("This tool does not accept arguments.")
    if name in ("explain_financial_concept", "get_financial_knowledge"):
        if not isinstance(arguments, dict) or set(arguments) - {"concept", "knowledge_type"}:
            raise ValueError("Unsupported financial-concept argument.")
        concept = arguments.get("concept", arguments.get("knowledge_type"))
        if concept is None and name == "get_financial_knowledge" and not arguments:
            return outcome(_financial())
        if not isinstance(concept, str) or concept not in FINANCIAL_CONCEPTS:
            raise ValueError("Choose a supported financial concept.")
        return outcome(_financial({"concept": concept}))
    if name == "search_dataset":
        if set(arguments) != {"query"}:
            raise ValueError("Dataset search requires exactly one query argument.")
        return outcome(tools[name](**arguments))
    if name == "get_model_performance":
        if set(arguments) - {"include_metrics"}:
            raise ValueError("Unsupported model-performance argument.")
        requested = arguments.get("include_metrics")
        allowed = {"precision", "recall", "f1", "roc_auc", "average_precision", "brier_score"}
        if requested is not None and (
            not isinstance(requested, list)
            or any(not isinstance(item, str) or item not in allowed for item in requested)
            or len(set(requested)) != len(requested)
        ):
            raise ValueError("Invalid metric list.")
        data = tools[name]()
        if requested is not None:
            data["test_metrics"] = {
                key: value for key, value in (data.get("test_metrics") or {}).items()
                if key in requested
            }
        return outcome(data)
    if arguments:
        raise ValueError("This tool does not accept arguments.")
    return outcome(tools[name]())


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": ({
                "include_metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["precision", "recall", "f1", "roc_auc", "average_precision", "brier_score"]},
                    "description": "Optional exact metric names to include from the saved test report.",
                }
            } if name == "get_model_performance" else ({
                "concept": {"type": "string", "enum": list(FINANCIAL_CONCEPTS), "description": "One supported concept to explain."}
            } if name == "explain_financial_concept" else {})),
            "required": ["concept"] if name == "explain_financial_concept" else [],
            "additionalProperties": False,
        },
    }}
    for name, description in [
        ("get_current_assessment", "Retrieve the current saved-model assessment and input features."),
        ("get_current_shap_explanation", "Retrieve actual SHAP values for the current assessment."),
        ("get_global_feature_importance", "Retrieve actual held-out permutation importance."),
        ("get_model_performance", "Retrieve saved validation/test metrics and limitations."),
        ("get_dataset_summary", "Retrieve the saved dataset schema, counts, and missingness summary."),
        ("get_dataset_schema", "Retrieve column lists and confirmed feature mappings for imported datasets."),
        ("get_dataset_statistics", "Retrieve locally registered dataset counts, labels, and quality statistics."),
        ("get_training_history", "Retrieve metadata for completed versioned model training runs."),
        ("get_active_model_info", "Retrieve the active model version, target, and feature list."),
        ("get_data_quality_report", "Retrieve missingness, duplicate-check, and unmapped-column reports."),
        ("get_dataset_provenance", "Retrieve locally recorded source, license, checksum, and import date."),
        ("search_dataset", "Search dataset registry metadata and column names only; never scan applicant rows."),
        ("compare_models", "Compare versioned runs with their recorded features, datasets, and held-out metrics."),
        ("get_local_shap_explanation", "Retrieve actual SHAP values for the current applicant assessment."),
        ("get_model_metrics", "Retrieve saved evaluation metrics for the active model."),
        ("run_applicant_assessment", "Retrieve the assessment already run in the current session; does not create a new applicant profile."),
        ("get_responsible_ai_information", "Retrieve known responsible AI limitations and fairness status."),
        ("get_project_information", "Retrieve verified project purpose, architecture, and limitations."),
        ("explain_financial_concept", "Explain one supported general financial concept; not applicant-specific advice."),
    ]
]

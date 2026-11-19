"""Train a versioned source-membership model from explicitly labeled CSVs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from config import MODEL_PATH, MODEL_VERSIONS_DIR, REPORT_DIR
from dataset_manager import FEATURE_LABELS, NUMERIC_FEATURES
from train_model import RANDOM_STATE, _evaluate, _make_pipeline, _positive_scores


def _normalize_values(series: pd.Series, feature: str) -> pd.Series:
    values = series.astype("string").str.strip()
    if feature in NUMERIC_FEATURES:
        cleaned = values.str.replace(r"[$,%]", "", regex=True).str.replace(",", "", regex=False)
        if feature == "term":
            cleaned = cleaned.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        numeric = pd.to_numeric(cleaned, errors="coerce").replace([np.inf, -np.inf], np.nan)
        limits = {"loan_amnt": (0, 1_000_000), "annual_inc": (0, 1_000_000_000), "dti": (0, 100), "int_rate": (0, 100), "term": (0, 600)}
        low, high = limits[feature]
        return numeric.where((numeric > low) & (numeric <= high))
    return values.mask(values.isna() | values.str.casefold().isin({"", "nan", "null", "none"}), "Unknown").str.strip()


def _sample_one(entry: dict[str, Any], features: list[str], maximum: int, chunksize: int, progress: Callable[[int], None] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(entry["stored_file"])
    mapping = entry["mapping"]
    selected = {feature: mapping[feature] for feature in features}
    if len(set(selected.values())) != len(selected):
        raise ValueError(f"Dataset {entry['name']} maps multiple features to the same column.")
    rows = 0
    invalid: dict[str, int] = {feature: 0 for feature in features}
    reservoir = pd.DataFrame(columns=features)
    reservoir_priorities = np.array([], dtype=float)
    rng = np.random.default_rng(RANDOM_STATE + int(entry["dataset_id"][:6], 16) % 1_000_000)
    for chunk in pd.read_csv(path, usecols=list(selected.values()), chunksize=chunksize, low_memory=False):
        rows += len(chunk)
        if progress:
            progress(rows)
        normalized = pd.DataFrame({feature: _normalize_values(chunk[column], feature) for feature, column in selected.items()})
        for feature in features:
            invalid[feature] += int(normalized[feature].isna().sum())
        incoming = normalized.assign(_priority=rng.random(len(normalized)))
        candidates = pd.concat([reservoir.assign(_priority=reservoir_priorities), incoming], ignore_index=True)
        if len(candidates) > maximum:
            candidates = candidates.nsmallest(maximum, columns="_priority")
        reservoir_priorities = candidates.pop("_priority").to_numpy()
        reservoir = candidates.reset_index(drop=True)
    feature_summaries: dict[str, Any] = {}
    for feature in features:
        values = reservoir[feature]
        if feature in NUMERIC_FEATURES:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            feature_summaries[feature] = {"non_missing_sample_rows": int(len(numeric)), "quantiles": {"p10": float(numeric.quantile(.1)), "median": float(numeric.median()), "p90": float(numeric.quantile(.9))} if not numeric.empty else None}
        else:
            counts = values.fillna("Unknown").astype(str).value_counts().head(10)
            feature_summaries[feature] = {"non_missing_sample_rows": int(values.notna().sum()), "top_categories": {str(key): int(value) for key, value in counts.items()}}
    return reservoir, {
        "dataset_id": entry["dataset_id"], "name": entry["name"], "label": entry["label"],
        "label_definition": entry.get("label_definition"), "source_rows": rows,
        "sampled_rows": len(reservoir), "invalid_or_missing_by_feature": invalid,
        "feature_summaries": feature_summaries, "sha256": entry["sha256"],
        "source_url": entry.get("source_url", ""), "license": entry.get("license", ""),
        "mapping": selected, "unmapped_columns": entry.get("unmapped_columns", []),
        "date_columns_detected": entry.get("date_columns_detected", []),
        "date_sample_report": entry.get("date_sample_report", {}),
        "duplicate_rows": entry.get("duplicate_rows"),
        "duplicate_check_complete": entry.get("duplicate_check_complete", False),
    }


def _metadata(features: pd.DataFrame, train_idx: np.ndarray, feature_names: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training = features.iloc[train_idx]
    schema: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    distributions: dict[str, Any] = {}
    for feature in feature_names:
        label = FEATURE_LABELS[feature]
        if feature in NUMERIC_FEATURES:
            values = pd.to_numeric(training[feature], errors="coerce").dropna()
            median = float(values.median()) if not values.empty else 0.0
            schema[feature] = {"type": "number", "label": label, "integer": feature == "term", "non_negative": True}
            defaults[feature] = median
            distributions[feature] = {"quantiles_10_90": [float(values.quantile(.1)), float(values.quantile(.9))] if not values.empty else [0.0, 0.0]}
        else:
            values = training[feature].fillna("Unknown").astype(str)
            counts = values.value_counts()
            options = [str(item) for item in counts.index[:50]] or ["Unknown"]
            schema[feature] = {"type": "category", "label": label, "options": options}
            defaults[feature] = options[0]
            distributions[feature] = {"values": options, "weights": [int(counts.get(item, 0)) for item in options]}
    return schema, defaults, distributions


def train_from_registry(
    entries: list[dict[str, Any]],
    features: list[str],
    *,
    maximum_per_class: int = 40_000,
    chunksize: int = 100_000,
    model_path: Path = MODEL_PATH,
    versions_dir: Path = MODEL_VERSIONS_DIR,
    report_dir: Path = REPORT_DIR,
    model_kind: str = "random_forest",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if model_kind not in {"random_forest", "logistic_regression"}:
        raise ValueError("Choose a supported model: random_forest or logistic_regression.")
    if len(features) < 2 or len(set(features)) != len(features):
        raise ValueError("Select at least two distinct compatible features.")
    if not (0.50 <= train_fraction <= 0.85 and 0.05 <= validation_fraction <= 0.30 and train_fraction + validation_fraction < 0.95):
        raise ValueError("Training and validation split fractions are outside the supported range.")
    if not entries or {item.get("label") for item in entries} != {"accepted", "rejected"}:
        raise ValueError("Select at least one explicitly labeled accepted and one rejected dataset.")
    for entry in entries:
        if any(feature not in entry.get("mapping", {}) for feature in features):
            raise ValueError(f"{entry.get('name')} is missing a selected feature mapping.")

    files_per_label = {label: max(1, sum(item.get("label") == label for item in entries)) for label in ("accepted", "rejected")}
    frames: list[pd.DataFrame] = []
    source_reports: list[dict[str, Any]] = []
    total_source_rows = sum(int(item.get("row_count", 0) or 0) for item in entries)
    rows_before_file = 0
    for entry in entries:
        per_file_cap = max(100, maximum_per_class // files_per_label[entry["label"]])
        def update_file_progress(rows_read: int, *, entry_name: str = entry["name"], base: int = rows_before_file) -> None:
            if progress_callback and total_source_rows:
                fraction = min(.55, .05 + .50 * (base + rows_read) / total_source_rows)
                progress_callback(fraction, f"Scanning {entry_name}: {rows_read:,} rows")
        sample, report = _sample_one(entry, features, per_file_cap, chunksize, progress=update_file_progress)
        rows_before_file += int(report["source_rows"])
        sample["source_outcome"] = int(entry["label"] == "accepted")
        frames.append(sample)
        source_reports.append(report)
    combined = pd.concat(frames, ignore_index=True)
    class_frames = []
    for label in (0, 1):
        subset = combined.loc[combined["source_outcome"] == label]
        if len(subset) < 100:
            raise ValueError(f"Each source class needs at least 100 usable rows; found {len(subset)} for label {label}.")
        if len(subset) > maximum_per_class:
            subset = subset.sample(n=maximum_per_class, random_state=RANDOM_STATE)
        class_frames.append(subset)
    sample = pd.concat(class_frames, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    if progress_callback:
        progress_callback(.60, f"Prepared {len(sample):,} normalized rows; building train/validation/test splits")
    labels = sample.pop("source_outcome").astype("int8")
    numeric = [name for name in features if name in NUMERIC_FEATURES]
    categorical = [name for name in features if name not in NUMERIC_FEATURES]
    indices = np.arange(len(sample))
    train_idx, remainder = train_test_split(indices, test_size=1-train_fraction, random_state=RANDOM_STATE, stratify=labels)
    validation_share_of_remainder = validation_fraction / (1-train_fraction)
    validation_idx, test_idx = train_test_split(remainder, test_size=1-validation_share_of_remainder, random_state=RANDOM_STATE, stratify=labels.iloc[remainder])
    classifier = (RandomForestClassifier(n_estimators=120, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
                  if model_kind == "random_forest" else LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE))
    pipeline = _make_pipeline(numeric, categorical, classifier)
    pipeline.fit(sample.iloc[train_idx], labels.iloc[train_idx])
    if progress_callback:
        progress_callback(.78, "Fitted the selected classifier and preprocessing pipeline on training rows")
    val_ap = float(average_precision_score(labels.iloc[validation_idx], _positive_scores(pipeline, sample.iloc[validation_idx])))
    test_report = _evaluate(pipeline, sample.iloc[test_idx], labels.iloc[test_idx])
    test_report["confusion_matrix"]["labels"] = ["Rejected-file application record", "Accepted-file loan record"]
    test_report["class_counts"] = {"rejected_file_application_record": int((labels.iloc[test_idx] == 0).sum()), "accepted_file_loan_record": int((labels.iloc[test_idx] == 1).sum())}
    test_report["probability_warning"] = "Scores describe source membership in the selected historical files, not real-world approval probabilities."
    importance_frame = sample.iloc[test_idx]
    if len(importance_frame) > 500:
        importance_frame = importance_frame.sample(n=500, random_state=RANDOM_STATE)
    perm = permutation_importance(pipeline, importance_frame, labels.loc[importance_frame.index], scoring="average_precision", n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    global_importance = [{"feature": name, "label": FEATURE_LABELS[name], "importance_mean": float(mean), "importance_std": float(std)} for name, mean, std in zip(features, perm.importances_mean, perm.importances_std)]
    global_importance.sort(key=lambda row: abs(row["importance_mean"]), reverse=True)
    if progress_callback:
        progress_callback(.92, "Evaluated held-out rows and computed permutation importance")
    schema, defaults, distributions = _metadata(sample, train_idx, features)
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = hashlib.sha256("".join(sorted(entry["sha256"] for entry in entries)).encode()).hexdigest()[:8]
    version = f"{created}_{fingerprint}"
    target_text = "User-assigned accepted-file membership (1) versus rejected-file membership (0). These source labels are unmatched historical groups, not a lender decision."
    label_counts = {"accepted": int((labels == 1).sum()), "rejected": int((labels == 0).sum())}
    report = {
        "version": version, "created_at": datetime.now(timezone.utc).isoformat(), "selected_model": model_kind,
        "prediction_task": "Historical accepted/rejected source membership", "target_definition": target_text,
        "not_approval_prediction": True, "model_selection_metric": "held-out average precision",
        "validation_average_precision": val_ap, "split_method": f"stratified random train/validation/test split ({train_fraction:.0%}/{validation_fraction:.0%}/{1-train_fraction-validation_fraction:.0%})",
        "training_configuration": {"train_fraction": train_fraction, "validation_fraction": validation_fraction, "test_fraction": 1-train_fraction-validation_fraction, "random_state": RANDOM_STATE, "maximum_per_class": maximum_per_class, "classifier": model_kind},
        "rows_used": len(sample), "rows_eligible": len(sample), "feature_columns": features,
        "class_counts_before_split": label_counts,
        "class_imbalance_ratio": round(max(label_counts.values()) / min(label_counts.values()), 3),
        "test": test_report, "global_feature_importance": global_importance,
        "datasets": source_reports,
        "limitations": ["Accepted and rejected source rows are unmatched populations; their labels do not represent counterfactual lender decisions.", "The sources can differ in period, population, coverage, missingness, and feature measurement.", "Only features mapped and selected across every chosen dataset are included; other columns are retained in the originals and listed in each dataset record.", "Class balance, historical selection effects, and source proxies limit generalization; scores are uncalibrated.", "This classifier is not a genuine eligibility, creditworthiness, approval, or default model."],
    }
    metadata = {
        "version": version, "target_column": "source_outcome", "target_definition": target_text,
        "prediction_task": "Historical accepted/rejected source membership", "not_approval_prediction": True,
        "positive_class_label": "Accepted-file loan record", "negative_class_label": "Rejected-file application record",
        "positive_score_name": "accepted-file membership score", "score_warning": "Uncalibrated score for membership in the selected historical files; not a lender decision or approval probability.",
        "feature_columns": features, "feature_schema": schema, "input_defaults": defaults,
        "demo_distributions": distributions, "selected_model": model_kind,
        "source_file_names": [entry["name"] for entry in entries], "dataset_ids": [entry["dataset_id"] for entry in entries],
    }
    artifact = {"pipeline": pipeline, "metadata": metadata, "evaluation": report}
    versions_dir = Path(versions_dir)
    versions_dir.mkdir(parents=True, exist_ok=True)
    version_path = versions_dir / f"{version}.joblib"
    model_path = Path(model_path)
    temporary_version = version_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary_version, compress=3)
    temporary_version.replace(version_path)
    report_path = Path(report_dir) / f"training_{version}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    history_path = versions_dir / "history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append({"version": version, "path": str(version_path.resolve()), "created_at": report["created_at"], "selected_model": model_kind, "features": features, "dataset_ids": metadata["dataset_ids"], "rows_used": len(sample), "test_metrics": test_report["metrics"], "validation_average_precision": val_ap, "target_definition": target_text})
    temporary_history = history_path.with_suffix(".tmp")
    temporary_history.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    temporary_history.replace(history_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_active = model_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary_active, compress=3)
    temporary_active.replace(model_path)
    if progress_callback:
        progress_callback(1.0, "Saved and activated the new versioned model")
    return report


def model_history(versions_dir: Path = MODEL_VERSIONS_DIR) -> list[dict[str, Any]]:
    path = Path(versions_dir) / "history.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def activate_model_version(version: str, *, versions_dir: Path = MODEL_VERSIONS_DIR, model_path: Path = MODEL_PATH) -> None:
    record = next((item for item in model_history(versions_dir) if item.get("version") == version), None)
    if not record:
        raise ValueError("That model version is not in training history.")
    source = Path(record["path"]).resolve()
    allowed = Path(versions_dir).resolve()
    if not source.is_relative_to(allowed) or not source.is_file():
        raise ValueError("Model version file is missing or outside the versions directory.")
    artifact = joblib.load(source)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    target = Path(model_path)
    temporary = target.with_suffix(".tmp")
    joblib.dump(artifact, temporary, compress=3)
    temporary.replace(target)

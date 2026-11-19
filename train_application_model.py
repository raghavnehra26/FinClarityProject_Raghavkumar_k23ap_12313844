"""Fit a source-outcome model from LendingClub accepted/rejected CSV files.

The target is membership in the supplied Accepted/Rejected files, not a
reconstruction of an individual lender's approval decision.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    DATASET_REPORT_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_PATH,
)
from train_model import (
    _evaluate,
    _make_pipeline,
    _positive_scores,
)


ACCEPTED_DEFAULT = Path(
    r"C:\Users\ragha\AppData\Local\Temp\accepted_2007_to_2018Q4.csv"
)
REJECTED_DEFAULT = Path(
    r"C:\Users\ragha\AppData\Local\Temp\rejected_2007_to_2018Q4.csv"
)
MAX_ROWS_PER_CLASS = 40_000
CHUNK_SIZE = 100_000
RANDOM_STATE = 42

FEATURE_COLUMNS = (
    "loan_amnt",
    "dti",
    "emp_length",
    "purpose",
)
NUMERIC_FEATURES = ("loan_amnt", "dti")
CATEGORICAL_FEATURES = ("emp_length", "purpose")
FEATURE_LABELS = {
    "loan_amnt": "Requested loan amount",
    "dti": "Reported debt-to-income ratio",
    "emp_length": "Reported employment length",
    "purpose": "Reported loan purpose",
}
SELECTED_COLUMNS = {
    "accepted": (
        "loan_amnt",
        "dti",
        "emp_length",
        "purpose",
        "loan_status",
    ),
    "rejected": (
        "Amount Requested",
        "Debt-To-Income Ratio",
        "Employment Length",
        "Loan Title",
    ),
}


def _purpose_category(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    if not text or text in {"nan", "none", "null"}:
        return "Other or unspecified"
    rules = (
        (
            (
                "debt consolidation",
                "debtconsolidation",
                "credit card consolidation",
            ),
            "Debt consolidation",
        ),
        (("credit card", "creditcard"), "Credit card"),
        (("home improvement", "homeimprovement"), "Home improvement"),
        (("major purchase", "majorpurchase"), "Major purchase"),
        (("small business", "smallbusiness", "business"), "Small business"),
        (("car", "auto", "automobile", "motorcycle"), "Vehicle"),
        (("medical", "health", "dental"), "Medical"),
        (("moving", "relocation"), "Moving"),
        (("vacation", "travel"), "Vacation or travel"),
        (("renewable", "green energy"), "Renewable energy"),
        (("wedding",), "Wedding"),
        (("house", "real estate"), "House"),
        (("education", "student"), "Education"),
    )
    padded = f" {text} "
    for words, label in rules:
        if any(f" {word} " in padded for word in words):
            return label
    return "Other or unspecified"


def _employment_category(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value).casefold().strip())
    if not text or text in {"nan", "none", "null"}:
        return "Unknown"
    if "10+" in text or "10 years" in text or "10 yrs" in text:
        return "10+ years"
    if "<" in text or "less than" in text or "under 1" in text:
        return "< 1 year"
    match = re.search(r"\b([1-9])\b", text)
    if match:
        return f"{match.group(1)} year" if match.group(1) == "1" else f"{match.group(1)} years"
    return "Unknown"


def _numeric_series(values: pd.Series, *, dti: bool = False) -> pd.Series:
    cleaned = (
        values.astype("string")
        .str.replace(r"[$,%]", "", regex=True)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if dti:
        numeric = numeric.where((numeric >= 0) & (numeric <= 100))
    else:
        numeric = numeric.where((numeric > 0) & (numeric <= 1_000_000))
    return pd.Series(
        numeric.to_numpy(dtype=np.float64, na_value=np.nan),
        index=values.index,
        name=values.name,
    )


def _normalize_chunk(chunk: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "accepted":
        normalized = pd.DataFrame(
            {
                "loan_amnt": _numeric_series(chunk["loan_amnt"]),
                "dti": _numeric_series(chunk["dti"], dti=True),
                "emp_length": chunk["emp_length"].map(_employment_category),
                "purpose": chunk["purpose"].map(_purpose_category),
            },
            index=chunk.index,
        )
    else:
        normalized = pd.DataFrame(
            {
                "loan_amnt": _numeric_series(chunk["Amount Requested"]),
                "dti": _numeric_series(chunk["Debt-To-Income Ratio"], dti=True),
                "emp_length": chunk["Employment Length"].map(_employment_category),
                "purpose": chunk["Loan Title"].map(_purpose_category),
            },
            index=chunk.index,
        )
    return normalized.loc[normalized["loan_amnt"].notna()].reset_index(drop=True)


def _schema_summary(path: Path) -> dict[str, Any]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    sample = pd.read_csv(path, nrows=1_000, low_memory=False)
    column_sample_dtypes = {
        name: str(dtype) for name, dtype in sample.dtypes.items()
    }
    potentially_sensitive_fragments = (
        "address",
        "email",
        "phone",
        "birth",
        "race",
        "gender",
        "sex",
        "marital",
        "ssn",
        "social",
        "zip",
    )
    potentially_identifying_columns = [
        name
        for name in columns
        if name.casefold() in {"id", "url"}
        or name.casefold().endswith("_id")
        or any(
            fragment in name.casefold()
            for fragment in potentially_sensitive_fragments
        )
    ]
    return {
        "column_count": len(columns),
        "columns": columns,
        "column_sample_rows": len(sample),
        "column_sample_dtypes": column_sample_dtypes,
        "potentially_sensitive_or_identifying_columns": (
            potentially_identifying_columns
        ),
    }


def _reservoir_sample(
    path: Path,
    source: str,
    maximum: int,
    *,
    chunksize: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source.title()} CSV not found: {path}")
    columns = list(SELECTED_COLUMNS[source])
    total_rows = 0
    valid_rows = 0
    missing_values: Counter[str] = Counter()
    invalid_numeric_values: Counter[str] = Counter()
    loan_status_counts: Counter[str] = Counter()
    reservoir = pd.DataFrame(
        {
            **{name: pd.Series(dtype="object") for name in FEATURE_COLUMNS},
            "_priority": pd.Series(dtype="float64"),
        }
    )
    rng = np.random.default_rng(RANDOM_STATE + (0 if source == "accepted" else 1))
    started = time.monotonic()
    schema_report = _schema_summary(path)

    for chunk in pd.read_csv(
        path,
        usecols=columns,
        chunksize=chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        for column in columns:
            missing_values[column] += int(chunk[column].isna().sum())
        amount_column = (
            "loan_amnt" if source == "accepted" else "Amount Requested"
        )
        dti_column = (
            "dti" if source == "accepted" else "Debt-To-Income Ratio"
        )
        invalid_numeric_values["loan_amount"] += int(
            _numeric_series(chunk[amount_column]).isna().sum()
        )
        invalid_numeric_values["debt_to_income_ratio"] += int(
            _numeric_series(chunk[dti_column], dti=True).isna().sum()
        )
        if source == "accepted":
            loan_status_counts.update(
                chunk["loan_status"].fillna("<MISSING>").astype(str).value_counts().to_dict()
            )
        normalized = _normalize_chunk(chunk, source)
        valid_rows += len(normalized)
        if normalized.empty:
            continue
        priorities = rng.random(len(normalized))
        incoming = normalized.assign(_priority=priorities)
        candidates = pd.concat([reservoir, incoming], ignore_index=True)
        reservoir = (
            candidates.nsmallest(maximum, columns="_priority")
            .reset_index(drop=True)
        )
        if total_rows % 1_000_000 < len(chunk):
            print(
                f"{source.title()} scan: {total_rows:,} rows, "
                f"{len(reservoir):,} sampled",
                flush=True,
            )

    sampled = reservoir.drop(columns="_priority").reset_index(drop=True)
    return sampled, {
        "file": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        **schema_report,
        "selected_columns": columns,
        "selected_column_sampled_dtypes": {
            name: schema_report["column_sample_dtypes"][name]
            for name in columns
        },
        "total_rows": total_rows,
        "rows_with_valid_loan_amount": valid_rows,
        "sampled_rows": len(sampled),
        "missing_counts_selected_columns": dict(missing_values),
        "invalid_numeric_counts": dict(invalid_numeric_values),
        "accepted_loan_status_distribution": dict(loan_status_counts),
        "sampling_method": "Deterministic seeded reservoir sample after CSV chunk scan",
        "duplicate_rows": "Not assessed across the full file",
        "scan_seconds": round(time.monotonic() - started, 2),
    }


def build_training_sample(
    accepted_path: Path = ACCEPTED_DEFAULT,
    rejected_path: Path = REJECTED_DEFAULT,
    *,
    maximum_per_class: int = MAX_ROWS_PER_CLASS,
    chunksize: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if maximum_per_class < 100:
        raise ValueError("At least 100 rows per source class are required.")
    accepted, accepted_report = _reservoir_sample(
        Path(accepted_path), "accepted", maximum_per_class, chunksize=chunksize
    )
    rejected, rejected_report = _reservoir_sample(
        Path(rejected_path), "rejected", maximum_per_class, chunksize=chunksize
    )
    accepted["source_outcome"] = 1
    rejected["source_outcome"] = 0
    sample = pd.concat([accepted, rejected], ignore_index=True)
    if sample["source_outcome"].nunique() != 2:
        raise ValueError("Both source files need valid loan amount records.")
    report = {
        "accepted": accepted_report,
        "rejected": rejected_report,
        "sample_rows": len(sample),
        "sample_class_counts": {
            "accepted_file_membership": int((sample["source_outcome"] == 1).sum()),
            "rejected_file_membership": int((sample["source_outcome"] == 0).sum()),
        },
        "target_definition": (
            "1 = row sampled from accepted_2007_to_2018Q4.csv (issued-loan "
            "records); 0 = row sampled from rejected_2007_to_2018Q4.csv "
            "(rejected application records). This is source-file membership "
            "in historical, selected datasets, not an individual lender's "
            "approval or eligibility decision."
        ),
    }
    return sample, report


def _feature_metadata(
    sample: pd.DataFrame, train_indices: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = sample.iloc[train_indices]
    defaults: dict[str, Any] = {}
    schema: dict[str, Any] = {}
    distributions: dict[str, Any] = {}
    for name in FEATURE_COLUMNS:
        if name in NUMERIC_FEATURES:
            values = pd.to_numeric(training[name], errors="coerce").dropna()
            if values.empty:
                raise ValueError(f"No usable training values for {name}.")
            defaults[name] = float(values.median())
            schema[name] = {
                "type": "number",
                "label": FEATURE_LABELS[name],
                "integer": False,
                "non_negative": True,
            }
            distributions[name] = {
                "quantiles_10_90": [
                    float(values.quantile(0.10)),
                    float(values.quantile(0.90)),
                ]
            }
        else:
            counts = training[name].fillna("Unknown").astype(str).value_counts()
            options = [str(value) for value in counts.index]
            defaults[name] = options[0]
            schema[name] = {
                "type": "category",
                "label": FEATURE_LABELS[name],
                "options": options,
            }
            distributions[name] = {
                "values": options,
                "weights": [int(count) for count in counts],
            }
    return defaults, {"schema": schema, "distributions": distributions}


def train_application_model(
    accepted_path: Path = ACCEPTED_DEFAULT,
    rejected_path: Path = REJECTED_DEFAULT,
    model_path: Path = MODEL_PATH,
    evaluation_path: Path = EVALUATION_REPORT_PATH,
    dataset_report_path: Path = DATASET_REPORT_PATH,
    *,
    maximum_per_class: int = MAX_ROWS_PER_CLASS,
    chunksize: int = CHUNK_SIZE,
) -> dict[str, Any]:
    sample, dataset_report = build_training_sample(
        accepted_path,
        rejected_path,
        maximum_per_class=maximum_per_class,
        chunksize=chunksize,
    )
    features = sample.loc[:, FEATURE_COLUMNS].copy()
    labels = sample["source_outcome"].astype("int8")
    indices = np.arange(len(sample))
    train_idx, remainder = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    validation_idx, test_idx = train_test_split(
        remainder,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=labels.iloc[remainder],
    )

    candidates: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    fitted: dict[str, Pipeline] = {}
    validation_scores: dict[str, float] = {}
    for model_name, classifier in candidates.items():
        pipeline = _make_pipeline(
            list(NUMERIC_FEATURES),
            list(CATEGORICAL_FEATURES),
            classifier,
        )
        pipeline.fit(features.iloc[train_idx], labels.iloc[train_idx])
        fitted[model_name] = pipeline
        validation_scores[model_name] = float(
            average_precision_score(
                labels.iloc[validation_idx],
                _positive_scores(pipeline, features.iloc[validation_idx]),
            )
        )
    selected_name = max(validation_scores, key=validation_scores.get)
    pipeline = fitted[selected_name]
    test_report = _evaluate(pipeline, features.iloc[test_idx], labels.iloc[test_idx])

    importance_sample = features.iloc[test_idx]
    if len(importance_sample) > 500:
        importance_sample = importance_sample.sample(
            n=500, random_state=RANDOM_STATE
        )
    importance_labels = labels.loc[importance_sample.index]
    permutation = permutation_importance(
        pipeline,
        importance_sample,
        importance_labels,
        scoring="average_precision",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    global_importance = [
        {
            "feature": name,
            "label": FEATURE_LABELS[name],
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for name, mean, std in zip(
            FEATURE_COLUMNS,
            permutation.importances_mean,
            permutation.importances_std,
        )
    ]
    global_importance.sort(
        key=lambda item: abs(item["importance_mean"]), reverse=True
    )
    evaluation = {
        "target_column": "source_outcome",
        "target_definition": dataset_report["target_definition"],
        "prediction_task": (
            "Historical accepted-file versus rejected-file membership"
        ),
        "not_approval_prediction": True,
        "selected_model": selected_name,
        "model_selection_metric": "validation average precision",
        "validation_average_precision": validation_scores,
        "split_method": "stratified random split on the balanced reservoir sample",
        "rows_eligible": len(sample),
        "rows_used": len(sample),
        "sampled_for_runtime": True,
        "feature_columns": list(FEATURE_COLUMNS),
        "test": test_report,
        "global_feature_importance": global_importance,
        "limitations": [
            "Accepted-file records are issued loans; rejected-file records are rejected applications. The two sources do not contain matched applications.",
            "The classifier learns source membership in these selected files, not an individual lender's approval or eligibility decision.",
            "Feature completeness, measurement, and time coverage differ between accepted and rejected source files.",
            "Class balancing and reservoir sampling make the sample class proportions different from real application prevalence.",
            "Reported model scores are uncalibrated and must not be interpreted as approval probabilities.",
            "Historical data may encode selection bias, economic-period changes, and proxies for protected characteristics.",
            "SHAP describes model behavior and does not establish fairness, legality, or causation.",
            "This educational prototype is not for real lending decisions.",
        ],
    }
    test_report["class_counts"] = {
        "rejected_file_application_record": int(
            (labels.iloc[test_idx] == 0).sum()
        ),
        "accepted_file_loan_record": int((labels.iloc[test_idx] == 1).sum()),
    }
    test_report["confusion_matrix"]["labels"] = [
        "Rejected-file application record",
        "Accepted-file loan record",
    ]
    test_report["probability_warning"] = (
        "Scores are uncalibrated classifier outputs for membership in the "
        "balanced sample of these historical source files. The calibration "
        "curve and Brier score describe this held-out sample only; scores are "
        "not real-world approval probabilities."
    )
    evaluation["test"] = test_report
    dataset_report["license"] = {
        "kaggle_dataset_page": "CC0: Public Domain (as displayed on the Kaggle dataset page)",
        "caution": (
            "The files are derived from LendingClub records. Confirm applicable "
            "source terms and attribution before redistribution or deployment; "
            "this note is not legal advice."
        ),
        "source": "https://www.kaggle.com/wordsforthewise/lending-club",
    }
    defaults, schema_data = _feature_metadata(sample, train_idx)
    metadata = {
        "target_column": "source_outcome",
        "target_definition": dataset_report["target_definition"],
        "prediction_task": "Historical accepted/rejected source membership",
        "not_approval_prediction": True,
        "positive_class_label": "Accepted-file loan record",
        "negative_class_label": "Rejected-file application record",
        "positive_score_name": "accepted-file membership score",
        "score_warning": (
            "Uncalibrated model score for membership in these historical source "
            "files. It is not approval eligibility or an approval probability."
        ),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_schema": schema_data["schema"],
        "input_defaults": defaults,
        "demo_distributions": schema_data["distributions"],
        "selected_model": selected_name,
        "source_file_names": [
            Path(accepted_path).name,
            Path(rejected_path).name,
        ],
    }
    artifact = {"pipeline": pipeline, "metadata": metadata, "evaluation": evaluation}
    for path in (model_path, evaluation_path, dataset_report_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    Path(evaluation_path).write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    Path(dataset_report_path).write_text(
        json.dumps(dataset_report, indent=2) + "\n", encoding="utf-8"
    )
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an explicitly labeled accepted-source vs rejected-source "
            "model from the large LendingClub CSV files."
        )
    )
    parser.add_argument("--accepted", type=Path, default=ACCEPTED_DEFAULT)
    parser.add_argument("--rejected", type=Path, default=REJECTED_DEFAULT)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=EVALUATION_REPORT_PATH)
    parser.add_argument("--max-per-class", type=int, default=MAX_ROWS_PER_CLASS)
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    try:
        report = train_application_model(
            args.accepted,
            args.rejected,
            args.model_output,
            args.report_output,
            DATASET_REPORT_PATH,
            maximum_per_class=args.max_per_class,
            chunksize=args.chunksize,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

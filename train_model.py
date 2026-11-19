"""Train and evaluate a historical LendingClub default-risk model.

The supported labels describe outcomes for issued loans, not loan approvals.
Only explicitly allowlisted, application-time features are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from config import (
    DATA_PATH,
    DATASET_REPORT_PATH,
    EVALUATION_REPORT_PATH,
    MODEL_PATH,
)
from inspect_dataset import inspect_dataset


NUMERIC_FEATURES = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "acc_now_delinq",
)
CATEGORICAL_FEATURES = (
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
)
FEATURE_LABELS = {
    "loan_amnt": "Requested loan amount",
    "annual_inc": "Reported annual income",
    "dti": "Debt-to-income ratio reported in the source",
    "delinq_2yrs": "Delinquencies reported in the prior two years",
    "inq_last_6mths": "Credit inquiries in the prior six months",
    "open_acc": "Open credit accounts",
    "pub_rec": "Public-record indicators",
    "revol_bal": "Revolving credit balance",
    "total_acc": "Total credit accounts",
    "acc_now_delinq": "Currently delinquent accounts reported",
    "term": "Requested loan term",
    "emp_length": "Reported employment length",
    "home_ownership": "Reported home-ownership category",
    "verification_status": "Income-verification category",
    "purpose": "Reported loan purpose",
    "application_type": "Application type",
}
NON_NEGATIVE_FEATURES = frozenset(
    {
        "loan_amnt",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "total_acc",
        "acc_now_delinq",
    }
)
MAX_ROWS = 200_000
RANDOM_STATE = 42


def _target_values(frame: pd.DataFrame) -> tuple[pd.Series, str, str]:
    if "bad_loan" in frame.columns:
        raw = pd.to_numeric(frame["bad_loan"], errors="coerce")
        observed = set(raw.dropna().unique())
        if not observed:
            raise ValueError(
                "`bad_loan` contains no numeric labels. Verify the file's data "
                "dictionary and target encoding."
            )
        if observed.issubset({0, 1}):
            labels = raw.map({0: 0, 1: 1})
            mapping = "source labels 0=did not default, 1=default"
        elif observed.issubset({1, 2}):
            labels = raw.map({1: 1, 2: 0})
            mapping = "source labels 1=default, 2=did not default"
        else:
            raise ValueError(
                "`bad_loan` must use either a 0/1 or 1/2 binary encoding. "
                "Verify your data dictionary; no other mapping is inferred."
            )
        return (
            labels,
            "bad_loan",
            "Historical default among previously issued loans "
            f"({mapping}; model label 1=default). Verify the file's data "
            "dictionary before interpretation.",
        )

    if "loan_status" in frame.columns:
        statuses = frame["loan_status"].astype("string").str.strip().str.casefold()
        labels = pd.Series(np.nan, index=frame.index, dtype="float64")
        labels.loc[statuses.str.contains("charged off", na=False)] = 1
        labels.loc[statuses.str.fullmatch("fully paid", na=False)] = 0
        if labels.notna().sum() == 0:
            raise ValueError(
                "`loan_status` exists, but no exact Fully Paid or Charged Off "
                "outcomes were found. Inspect the status values before training."
            )
        return (
            labels,
            "loan_status",
            "Historical Charged Off versus Fully Paid outcomes for issued loans; "
            "other statuses are excluded. This does not predict loan approval.",
        )

    raise ValueError(
        "No supported target was found. Expected `bad_loan` from the classic "
        "LendingClub competition file or `loan_status` with Fully Paid/Charged "
        "Off outcomes. Approval/rejection will not be inferred."
    )


def prepare_training_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return only supported prediction-time features and verified outcome labels."""
    labels, target_column, target_definition = _target_values(frame)
    available_features = [
        name for name in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES) if name in frame
    ]
    if not available_features:
        raise ValueError(
            "No allowlisted application-time features were found. "
            "Inspect the schema; this version will not guess feature meanings."
        )

    eligible = labels.notna()
    features = frame.loc[eligible, available_features].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    labels = labels.loc[eligible].astype("int8")

    numeric_features = [name for name in NUMERIC_FEATURES if name in features]
    categorical_features = [name for name in CATEGORICAL_FEATURES if name in features]
    for name in numeric_features:
        features[name] = pd.to_numeric(features[name], errors="coerce")
    unusable_features = [
        name for name in features.columns if not features[name].notna().any()
    ]
    features = features.drop(columns=unusable_features)
    numeric_features = [name for name in numeric_features if name in features]
    categorical_features = [
        name for name in categorical_features if name in features
    ]
    if len(features.columns) < 2:
        raise ValueError(
            "Fewer than two non-empty allowlisted features remain after schema "
            "inspection; no model is trained."
        )

    details = {
        "target_column": target_column,
        "target_definition": target_definition,
        "feature_columns": list(features.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_non_target_columns": [
            name
            for name in frame.columns
            if name not in {*features.columns, target_column}
        ],
        "excluded_empty_features": unusable_features,
        "eligible_rows": len(features),
        "excluded_target_rows": int((~eligible).sum()),
    }
    return features, labels, details


def _split_indices(
    features: pd.DataFrame, labels: pd.Series, source: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if labels.value_counts().min() < 10:
        raise ValueError(
            "At least 10 examples of each outcome are required for a useful "
            "train/validation/test split."
        )

    if "issue_d" in source:
        dates = pd.to_datetime(source.loc[features.index, "issue_d"], format="mixed", errors="coerce")
        if dates.notna().mean() >= 0.95 and dates.nunique() >= 10:
            order = np.argsort(dates.to_numpy(), kind="stable")
            first = int(len(order) * 0.70)
            second = int(len(order) * 0.85)
            train_idx, validation_idx, test_idx = (
                order[:first],
                order[first:second],
                order[second:],
            )
            if all(labels.iloc[idx].nunique() == 2 for idx in (train_idx, validation_idx, test_idx)):
                return train_idx, validation_idx, test_idx, "chronological issue_d split"

    indices = np.arange(len(features))
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
    return train_idx, validation_idx, test_idx, "stratified random split"


def _make_pipeline(
    numeric_features: list[str], categorical_features: list[str], classifier: Any
) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
            ),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def _positive_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    positive_index = list(classifier.classes_).index(1)
    return model.predict_proba(features)[:, positive_index]


def _evaluate(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    scores = _positive_scores(model, features)
    predictions = (scores >= 0.5).astype("int8")
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, float | None] = {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, scores))
            if labels.nunique() == 2
            else None
        ),
        "average_precision": (
            float(average_precision_score(labels, scores))
            if labels.nunique() == 2
            else None
        ),
        "brier_score": float(brier_score_loss(labels, scores)),
    }
    calibration: dict[str, list[float]] = {"mean_predicted": [], "fraction_positive": []}
    if labels.nunique() == 2:
        fraction_positive, mean_predicted = calibration_curve(
            labels, scores, n_bins=10, strategy="quantile"
        )
        calibration = {
            "mean_predicted": [float(value) for value in mean_predicted],
            "fraction_positive": [float(value) for value in fraction_positive],
        }
    return {
        "test_rows": len(labels),
        "class_counts": {
            "did_not_default": int((labels == 0).sum()),
            "defaulted": int((labels == 1).sum()),
        },
        "metrics": metrics,
        "confusion_matrix": {
            "labels": ["Did not default", "Defaulted"],
            "values": matrix.tolist(),
        },
        "calibration": calibration,
        "probability_warning": (
            "Scores are uncalibrated model outputs. Brier score and calibration "
            "curve describe this held-out sample only; scores are not real-world "
            "default probabilities."
        ),
    }


def _feature_schema(
    features: pd.DataFrame,
    details: dict[str, Any],
    training_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training_features = features.iloc[training_indices]
    defaults: dict[str, Any] = {}
    schema: dict[str, Any] = {}
    demo_distributions: dict[str, Any] = {}

    for name in details["feature_columns"]:
        label = FEATURE_LABELS[name]
        if name in details["numeric_features"]:
            values = pd.to_numeric(training_features[name], errors="coerce").dropna()
            if values.empty:
                median = 0.0
                quantiles = [0.0, 0.0]
            else:
                median = float(values.median())
                quantiles = [
                    float(values.quantile(0.10)),
                    float(values.quantile(0.90)),
                ]
            defaults[name] = median
            schema[name] = {
                "type": "number",
                "label": label,
                "integer": name in {
                    "delinq_2yrs",
                    "inq_last_6mths",
                    "open_acc",
                    "pub_rec",
                    "total_acc",
                    "acc_now_delinq",
                },
                "non_negative": name in NON_NEGATIVE_FEATURES,
            }
            demo_distributions[name] = {"quantiles_10_90": quantiles}
        else:
            values = training_features[name].dropna().astype(str)
            counts = values.value_counts()
            options = [str(value) for value in counts.index[:30]]
            if not options:
                options = ["Unknown"]
            defaults[name] = options[0]
            schema[name] = {
                "type": "category",
                "label": label,
                "options": options,
            }
            demo_distributions[name] = {
                "values": [str(value) for value in counts.index[:30]],
                "weights": [int(count) for count in counts.iloc[:30]],
            }
    return defaults, {"schema": schema, "demo_distributions": demo_distributions}


def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    evaluation_path: Path = EVALUATION_REPORT_PATH,
    dataset_report_path: Path = DATASET_REPORT_PATH,
) -> dict[str, Any]:
    """Train candidate classifiers, choose on validation data, and save held-out results."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place the actual LendingClub CSV "
            "at data/loanStats_train.csv or set FINCLARITY_DATA_PATH."
        )
    raw = pd.read_csv(data_path, low_memory=False)
    dataset_report = inspect_dataset(data_path)
    features, labels, details = prepare_training_data(raw)

    sampled = False
    if len(features) > MAX_ROWS:
        indices = np.arange(len(features))
        selected, _ = train_test_split(
            indices,
            train_size=MAX_ROWS,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
        features = features.iloc[selected].copy()
        labels = labels.iloc[selected].copy()
        sampled = True

    train_idx, validation_idx, test_idx, split_method = _split_indices(
        features, labels, raw
    )
    numeric_features = details["numeric_features"]
    categorical_features = details["categorical_features"]

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    fitted: dict[str, Pipeline] = {}
    validation_scores: dict[str, float] = {}
    for name, classifier in candidates.items():
        model = _make_pipeline(numeric_features, categorical_features, classifier)
        model.fit(features.iloc[train_idx], labels.iloc[train_idx])
        fitted[name] = model
        validation_scores[name] = float(
            average_precision_score(
                labels.iloc[validation_idx],
                _positive_scores(model, features.iloc[validation_idx]),
            )
        )

    selected_name = max(validation_scores, key=validation_scores.get)
    selected_model = fitted[selected_name]
    test_metrics = _evaluate(selected_model, features.iloc[test_idx], labels.iloc[test_idx])
    train_defaults, schema_details = _feature_schema(features, details, train_idx)

    importance_sample = min(500, len(test_idx))
    test_features = features.iloc[test_idx]
    test_labels = labels.iloc[test_idx]
    if len(test_idx) > importance_sample:
        sampled_test = test_features.sample(
            n=importance_sample, random_state=RANDOM_STATE
        )
        sampled_labels = test_labels.loc[sampled_test.index]
    else:
        sampled_test = test_features
        sampled_labels = test_labels
    importance = permutation_importance(
        selected_model,
        sampled_test,
        sampled_labels,
        scoring="average_precision",
        n_repeats=2,
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
            details["feature_columns"],
            importance.importances_mean,
            importance.importances_std,
        )
    ]
    global_importance.sort(
        key=lambda item: abs(item["importance_mean"]), reverse=True
    )

    evaluation = {
        "target_column": details["target_column"],
        "target_definition": details["target_definition"],
        "prediction_task": "Historical default risk for previously issued loans",
        "not_approval_prediction": True,
        "selected_model": selected_name,
        "model_selection_metric": "validation average precision",
        "validation_average_precision": validation_scores,
        "split_method": split_method,
        "rows_eligible": details["eligible_rows"],
        "rows_used": len(features),
        "sampled_for_runtime": sampled,
        "feature_columns": details["feature_columns"],
        "test": test_metrics,
        "global_feature_importance": global_importance,
        "limitations": [
            "Historical outcomes cover loans that were issued, not rejected applications.",
            "Historical LendingClub data may not represent current applicants, policies, or economic conditions.",
            "Uncalibrated scores must not be interpreted as real-world probabilities.",
            "Removing protected or identifying columns does not establish fairness.",
            "SHAP describes model behavior and does not establish causation or fairness.",
            "The model and threshold are educational and cannot make actual lending decisions.",
        ],
    }
    metadata = {
        "target_column": details["target_column"],
        "target_definition": details["target_definition"],
        "prediction_task": "Historical default risk for previously issued loans",
        "positive_class_label": "Defaulted",
        "negative_class_label": "Did not default",
        "positive_score_name": "historical default-risk score",
        "score_warning": (
            "Uncalibrated model score for historical issued-loan outcomes; "
            "not a real-world probability or approval decision."
        ),
        "feature_columns": details["feature_columns"],
        "feature_schema": schema_details["schema"],
        "input_defaults": train_defaults,
        "demo_distributions": schema_details["demo_distributions"],
        "selected_model": selected_name,
        "source_file_name": data_path.name,
    }
    artifact = {
        "pipeline": selected_model,
        "metadata": metadata,
        "evaluation": evaluation,
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(evaluation_path).parent.mkdir(parents=True, exist_ok=True)
    Path(dataset_report_path).parent.mkdir(parents=True, exist_ok=True)
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
        description="Train and evaluate a real LendingClub default-risk model."
    )
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=EVALUATION_REPORT_PATH)
    args = parser.parse_args()
    try:
        evaluation = train_model(
            args.csv,
            args.model_output,
            args.report_output,
            DATASET_REPORT_PATH,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()

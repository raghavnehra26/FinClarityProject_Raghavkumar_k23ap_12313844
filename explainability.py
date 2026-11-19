"""SHAP explanations tied to the saved model's actual probability output."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def explain_prediction(
    artifact: dict[str, Any], applicant: dict[str, Any]
) -> dict[str, Any]:
    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]
    feature_names = metadata["feature_columns"]
    row = pd.DataFrame([{name: applicant[name] for name in feature_names}])
    score = float(pipeline.predict_proba(row)[0, list(pipeline.classes_).index(1)])
    prediction = int(score >= 0.5)

    baseline = pd.DataFrame(
        [metadata["input_defaults"]], columns=feature_names
    )
    schema = metadata["feature_schema"]

    def encode(frame: pd.DataFrame) -> np.ndarray:
        encoded = np.empty((len(frame), len(feature_names)), dtype=float)
        for column_index, name in enumerate(feature_names):
            if schema[name]["type"] == "number":
                encoded[:, column_index] = pd.to_numeric(
                    frame[name], errors="coerce"
                ).to_numpy(dtype=float)
            else:
                options = schema[name]["options"]
                categories = {value: index for index, value in enumerate(options)}
                encoded[:, column_index] = [
                    categories.get(str(value), len(options))
                    for value in frame[name]
                ]
        return encoded

    def predict_encoded(values: np.ndarray) -> np.ndarray:
        decoded: list[dict[str, Any]] = []
        for encoded_row in np.atleast_2d(values):
            decoded_row: dict[str, Any] = {}
            for index, name in enumerate(feature_names):
                if schema[name]["type"] == "number":
                    decoded_row[name] = float(encoded_row[index])
                else:
                    options = schema[name]["options"]
                    category_index = int(round(encoded_row[index]))
                    decoded_row[name] = (
                        options[category_index]
                        if 0 <= category_index < len(options)
                        else "__unseen_category__"
                    )
            decoded.append(decoded_row)
        frame = pd.DataFrame(decoded, columns=feature_names)
        positive_index = list(pipeline.classes_).index(1)
        return pipeline.predict_proba(frame)[:, positive_index]

    baseline_values = encode(baseline)
    row_values = encode(row)
    masker = shap.maskers.Independent(baseline_values)
    explainer = shap.Explainer(
        predict_encoded,
        masker,
        algorithm="permutation",
        feature_names=feature_names,
    )
    explanation = explainer(
        row_values, max_evals=max(2 * len(feature_names) + 1, 201)
    )
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        contributions = values[0, :, 0]
    elif values.ndim == 2:
        contributions = values[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    base_value = float(base_values.reshape(-1)[0])

    items = []
    positive_class_label = metadata.get(
        "positive_class_label",
        "Defaulted" if metadata.get("prediction_task", "").startswith("Historical default") else "Positive class",
    )
    negative_class_label = metadata.get(
        "negative_class_label",
        "Did not default" if metadata.get("prediction_task", "").startswith("Historical default") else "Negative class",
    )
    positive_score_name = metadata.get(
        "positive_score_name", "historical default-risk score"
    )
    for name, contribution in zip(feature_names, contributions):
        value = applicant[name]
        items.append(
            {
                "feature": name,
                "label": metadata["feature_schema"][name]["label"],
                "value": str(value),
                "contribution": float(contribution),
                "direction": (
                    f"higher {positive_score_name}"
                    if contribution > 0
                    else f"lower {positive_score_name}"
                ),
            }
        )
    items.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    return {
        "prediction": prediction,
        "prediction_label": (
            positive_class_label if prediction else negative_class_label
        ),
        "positive_class_label": positive_class_label,
        "negative_class_label": negative_class_label,
        "positive_score_name": positive_score_name,
        "target": metadata["target_definition"],
        "model_score": score,
        "score_warning": metadata.get(
            "score_warning",
            "Uncalibrated score for historical issued-loan outcomes; "
            "not a real-world probability or approval decision.",
        ),
        "base_value": base_value,
        "contribution_sum": float(np.sum(contributions)),
        "explanations": items,
        "applicant_inputs": {name: applicant[name] for name in feature_names},
    }


def global_importance(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return artifact["evaluation"]["global_feature_importance"]


def make_synthetic_examples(
    artifact: dict[str, Any], seed: int = 17, candidate_count: int = 128
) -> dict[str, Any]:
    """Sample feature values independently from training summaries, never loan rows."""
    metadata = artifact["metadata"]
    generator = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    for _ in range(candidate_count):
        applicant: dict[str, Any] = {}
        for name, schema in metadata["feature_schema"].items():
            distribution = metadata["demo_distributions"][name]
            if schema["type"] == "number":
                lower, upper = distribution["quantiles_10_90"]
                value = (
                    float(lower)
                    if lower == upper
                    else float(generator.uniform(lower, upper))
                )
                applicant[name] = int(round(value)) if schema["integer"] else value
            else:
                values = distribution["values"]
                weights = np.asarray(distribution["weights"], dtype=float)
                probabilities = weights / weights.sum()
                applicant[name] = str(
                    generator.choice(values, p=probabilities)
                )
        candidates.append(applicant)

    scores = artifact["pipeline"].predict_proba(
        pd.DataFrame(candidates, columns=metadata["feature_columns"])
    )[:, list(artifact["pipeline"].classes_).index(1)]
    low_index = int(np.argmin(scores))
    high_index = int(np.argmax(scores))
    if np.isclose(scores[low_index], scores[high_index]):
        return {
            "examples": [],
            "message": (
                "The model produced no meaningful score contrast across sampled "
                "synthetic profiles; no high/low-score examples are shown."
            ),
        }

    low_prediction = int(scores[low_index] >= 0.5)
    high_prediction = int(scores[high_index] >= 0.5)
    contrast_message = (
        "The generated profiles fall on opposite sides of the model's 0.5 "
        "classification threshold."
        if low_prediction != high_prediction
        else (
            "The profiles have different scores but the same predicted class at "
            "the model's 0.5 threshold; no opposite-class example was found."
        )
    )
    return {
        "examples": [
            {
                "name": "Lower model-score profile",
                "result": explain_prediction(artifact, candidates[low_index]),
            },
            {
                "name": "Higher model-score profile",
                "result": explain_prediction(artifact, candidates[high_index]),
            },
        ],
        "message": (
            "Profiles are independently generated from training-set feature "
            "summaries. The displayed outcomes and explanations come from the "
            "saved model and SHAP; they are not real customer records. "
            f"{contrast_message}"
        ),
    }

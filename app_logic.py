"""Input validation shared by the UI and unit tests."""

from __future__ import annotations

import math
from typing import Any


def validate_applicant_input(
    schema: dict[str, dict[str, Any]], values: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    cleaned: dict[str, Any] = {}
    errors: list[str] = []
    for name, spec in schema.items():
        value = values.get(name)
        label = spec["label"]
        if value is None or value == "":
            errors.append(f"Enter or select a value for {label}.")
            continue
        if spec["type"] == "number":
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{label} must be a number.")
                continue
            if not math.isfinite(number):
                errors.append(f"{label} must be a finite number.")
                continue
            if spec.get("non_negative") and number < 0:
                errors.append(f"{label} cannot be negative.")
                continue
            if spec.get("integer") and not number.is_integer():
                errors.append(f"{label} must be a whole number.")
                continue
            cleaned[name] = int(number) if spec.get("integer") else number
        else:
            options = spec.get("options", [])
            if value not in options:
                errors.append(f"Choose an observed training value for {label}.")
                continue
            cleaned[name] = str(value)
    return cleaned, errors

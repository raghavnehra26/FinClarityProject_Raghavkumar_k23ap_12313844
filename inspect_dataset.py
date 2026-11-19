import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA_PATH, TARGET_COLUMN


def inspect_dataset(path: Path = DATA_PATH) -> dict[str, Any]:
    """Return schema and quality facts without exposing applicant row values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the Kaggle training CSV at "
            "data/loanStats_train.csv or set FINCLARITY_DATA_PATH."
        )

    frame = pd.read_csv(path, low_memory=False)
    row_count = len(frame)
    column_report = []
    for name in frame.columns:
        missing_count = int(frame[name].isna().sum())
        column_report.append(
            {
                "column": str(name),
                "dtype": str(frame[name].dtype),
                "missing_count": missing_count,
                "missing_percent": (
                    round(missing_count * 100 / row_count, 2) if row_count else 0.0
                ),
            }
        )

    target_column = None
    target_definition = None
    target_distribution = None
    target_candidates = []
    if TARGET_COLUMN in frame.columns:
        target_column = TARGET_COLUMN
        counts = frame[TARGET_COLUMN].value_counts(dropna=False)
        target_distribution = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ]
        numeric_labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        observed = set(numeric_labels.dropna().unique())
        if observed and observed.issubset({0, 1}):
            mapping = "0 = did not default, 1 = defaulted"
        elif observed and observed.issubset({1, 2}):
            mapping = "1 = defaulted, 2 = did not default"
        else:
            mapping = None
        if mapping is None:
            target_definition = (
                "A column named bad_loan is present, but its observed labels do "
                "not match a supported binary encoding. Verify its data dictionary."
            )
        else:
            target_definition = (
                f"Historical outcome for previously issued loans ({mapping}); "
                "not an approval/rejection target. Verify this file's data "
                "dictionary before interpreting or training."
            )
        target_candidates.append(
            {"column": TARGET_COLUMN, "meaning": "Historical default outcome"}
        )
    elif "loan_status" in frame.columns:
        target_column = "loan_status"
        counts = frame["loan_status"].value_counts(dropna=False)
        target_distribution = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ]
        target_definition = (
            "Historical status for previously issued loans. Fully Paid and "
            "Charged Off may support a default-risk label; ongoing/other statuses "
            "are not equivalent to rejection. This is not an approval target."
        )
        target_candidates.append(
            {"column": "loan_status", "meaning": "Historical status of issued loans"}
        )

    approval_columns = [
        name
        for name in ("approval_status", "application_status", "approved")
        if name in frame.columns
    ]
    for name in approval_columns:
        target_candidates.append(
            {"column": name, "meaning": "Potential application decision; verify schema"}
        )

    sensitive_name_fragments = (
        "ssn",
        "member_id",
        "emp_title",
        "zip_code",
        "address",
        "email",
        "phone",
        "birth",
        "race",
        "gender",
        "sex",
        "marital",
    )
    sensitive_or_identifying_columns = [
        str(name)
        for name in frame.columns
        if any(fragment in str(name).casefold() for fragment in sensitive_name_fragments)
    ]

    return {
        "file": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "rows": row_count,
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "column_report": column_report,
        "target_candidates": target_candidates,
        "target_column": target_column,
        "target_definition": target_definition,
        "target_distribution": target_distribution,
        "sensitive_or_identifying_columns": sensitive_or_identifying_columns,
        "approval_target_warning": (
            "A dataset of issued loans cannot be treated as rejected applications. "
            "Do not claim approval/rejection predictions unless a verified "
            "application-decision label is present."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a local LendingClub CSV without printing applicant records."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_PATH,
        help="CSV path; defaults to data/loanStats_train.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; no applicant rows are written.",
    )
    args = parser.parse_args()
    try:
        report = inspect_dataset(args.csv)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        parser.error(str(error))
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

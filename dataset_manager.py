"""Safe local registry for explicitly labeled loan CSV datasets."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROJECT_ROOT


MANAGED_ROOT = PROJECT_ROOT / "data" / "managed"
REGISTRY_PATH = MANAGED_ROOT / "registry.json"
FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "loan_amnt": ("loan_amnt", "loan_amount", "amount_requested", "requested_amount", "amount"),
    "annual_inc": ("annual_inc", "annual_income", "income", "yearly_income"),
    "dti": ("dti", "debt_to_income", "debt_to_income_ratio", "dti_ratio"),
    "emp_length": ("emp_length", "employment_length", "years_employed", "employment_tenure"),
    "purpose": ("purpose", "loan_purpose", "loan_title", "purpose_category"),
    "int_rate": ("int_rate", "interest_rate", "apr"),
    "term": ("term", "loan_term", "term_months"),
    "grade": ("grade", "credit_grade", "risk_grade"),
    "home_ownership": ("home_ownership", "homeownership", "housing_status"),
    "verification_status": ("verification_status", "income_verification", "verification"),
    "application_type": ("application_type", "applicationtype"),
    "delinq_2yrs": ("delinq_2yrs", "delinquencies_2yrs", "delinquencies_last_2_years"),
    "inq_last_6mths": ("inq_last_6mths", "inquiries_6mths", "credit_inquiries_6mths"),
    "open_acc": ("open_acc", "open_accounts", "open_credit_accounts"),
    "pub_rec": ("pub_rec", "public_records", "public_record_count"),
    "revol_bal": ("revol_bal", "revolving_balance", "revolving_credit_balance"),
    "total_acc": ("total_acc", "total_accounts", "total_credit_accounts"),
    "acc_now_delinq": ("acc_now_delinq", "accounts_currently_delinquent"),
    "revol_util": ("revol_util", "revolving_utilization", "revol_utilization"),
}
NUMERIC_FEATURES = {"loan_amnt", "annual_inc", "dti", "int_rate", "term", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc", "acc_now_delinq", "revol_util"}
# These values are set or influenced by a lender after selection and are unsafe
# as application-time inputs in a source-membership classifier.
POST_DECISION_FEATURES = {"int_rate", "grade"}
FEATURE_LABELS = {
    "loan_amnt": "Requested loan amount", "annual_inc": "Reported annual income",
    "dti": "Reported debt-to-income ratio", "emp_length": "Reported employment length",
    "purpose": "Reported loan purpose", "int_rate": "Reported interest rate",
    "term": "Reported loan term", "grade": "Reported credit grade",
    "home_ownership": "Reported home ownership", "verification_status": "Income verification status",
    "application_type": "Application type", "delinq_2yrs": "Delinquencies in prior two years",
    "inq_last_6mths": "Credit inquiries in prior six months", "open_acc": "Open credit accounts",
    "pub_rec": "Public record indicators", "revol_bal": "Revolving credit balance",
    "total_acc": "Total credit accounts", "acc_now_delinq": "Currently delinquent accounts",
    "revol_util": "Revolving credit utilization",
}


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _date_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if any(token in _normalized_name(column) for token in ("date", "issue_d", "origination_d", "application_d", "created_d"))]


def _summarize_dates(sample: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    summary = {}
    for column in columns:
        parsed = pd.to_datetime(sample[column], format="mixed", errors="coerce").dropna()
        summary[column] = {
            "sampled_rows": len(sample), "parseable_rows": len(parsed),
            "minimum": parsed.min().isoformat() if not parsed.empty else None,
            "maximum": parsed.max().isoformat() if not parsed.empty else None,
        }
    return summary


def detect_column_mapping(columns: list[str]) -> dict[str, str]:
    normalized = {_normalized_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for feature, aliases in FEATURE_ALIASES.items():
        match = next((normalized[alias] for alias in aliases if alias in normalized), None)
        if match is not None:
            mapping[feature] = match
    return mapping


def load_registry(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the dataset registry: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Dataset registry must contain a list of records.")
    return data


def _save_registry(entries: list[dict[str, Any]], path: Path = REGISTRY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_csv_bytes(contents: bytes, *, chunksize: int = 100_000) -> dict[str, Any]:
    if not contents:
        raise ValueError("The uploaded file is empty.")
    import io

    try:
        columns = list(pd.read_csv(io.BytesIO(contents), nrows=0).columns)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"The uploaded file is not a readable CSV: {exc}") from exc
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("CSV headers must be present and unique.")
    row_count = 0
    missing: dict[str, int] = {column: 0 for column in columns}
    duplicate_hashes: set[int] = set()
    duplicate_rows = 0
    duplicate_check_limit = 250_000
    duplicate_rows_checked = 0
    try:
        for chunk in pd.read_csv(io.BytesIO(contents), chunksize=chunksize, low_memory=False):
            row_count += len(chunk)
            for column in columns:
                missing[column] += int(chunk[column].isna().sum())
            if duplicate_rows_checked < duplicate_check_limit:
                hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
                for row_hash in hashes[:duplicate_check_limit - duplicate_rows_checked]:
                    value = int(row_hash)
                    if value in duplicate_hashes:
                        duplicate_rows += 1
                    else:
                        duplicate_hashes.add(value)
                duplicate_rows_checked += min(len(hashes), duplicate_check_limit - duplicate_rows_checked)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise ValueError(f"CSV validation failed: {exc}") from exc
    if row_count == 0:
        raise ValueError("The uploaded CSV contains headers but no data rows.")
    mapping = detect_column_mapping(columns)
    date_columns = _date_columns(columns)
    date_summary = _summarize_dates(pd.read_csv(io.BytesIO(contents), nrows=1_000, usecols=date_columns) if date_columns else pd.DataFrame(), date_columns)
    return {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "missing_counts": missing,
        "duplicate_rows": duplicate_rows if row_count <= duplicate_check_limit else None,
        "duplicate_rows_in_checked_rows": duplicate_rows,
        "duplicate_check_rows": duplicate_rows_checked,
        "duplicate_check_complete": row_count <= duplicate_check_limit,
        "detected_mapping": mapping,
        "date_columns_detected": date_columns, "date_sample_report": date_summary,
        "unmapped_columns": [column for column in columns if column not in mapping.values()],
        "mapping_note": "Detected mappings are suggestions. Confirm them before training.",
    }


def inspect_csv_path(path: Path, *, chunksize: int = 100_000) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("The source file is missing or empty.")
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise ValueError(f"The source is not a readable CSV: {exc}") from exc
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("CSV headers must be present and unique.")
    row_count = 0
    missing = {column: 0 for column in columns}
    duplicate_hashes: set[int] = set()
    duplicate_rows = 0
    duplicate_rows_checked = 0
    limit = 250_000
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            row_count += len(chunk)
            for column in columns:
                missing[column] += int(chunk[column].isna().sum())
            if duplicate_rows_checked < limit:
                hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
                available = limit - duplicate_rows_checked
                for row_hash in hashes[:available]:
                    value = int(row_hash)
                    if value in duplicate_hashes:
                        duplicate_rows += 1
                    else:
                        duplicate_hashes.add(value)
                duplicate_rows_checked += min(len(hashes), available)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise ValueError(f"CSV validation failed: {exc}") from exc
    if row_count == 0:
        raise ValueError("The uploaded CSV contains headers but no data rows.")
    mapping = detect_column_mapping(columns)
    date_columns = _date_columns(columns)
    date_summary = _summarize_dates(pd.read_csv(path, nrows=1_000, usecols=date_columns) if date_columns else pd.DataFrame(), date_columns)
    return {
        "row_count": row_count, "column_count": len(columns), "columns": columns,
        "missing_counts": missing,
        "duplicate_rows": duplicate_rows if row_count <= limit else None,
        "duplicate_rows_in_checked_rows": duplicate_rows,
        "duplicate_check_rows": duplicate_rows_checked,
        "duplicate_check_complete": row_count <= limit,
        "detected_mapping": mapping,
        "date_columns_detected": date_columns, "date_sample_report": date_summary,
        "unmapped_columns": [column for column in columns if column not in mapping.values()],
        "mapping_note": "Detected mappings are suggestions. Confirm them before training.",
    }


def register_uploaded_csv(
    filename: str,
    contents: bytes,
    label: str,
    *,
    mapping: dict[str, str] | None = None,
    description: str = "",
    source_url: str = "",
    license_name: str = "Not specified by uploader",
    registry_path: Path = REGISTRY_PATH,
    storage_root: Path = MANAGED_ROOT,
) -> dict[str, Any]:
    if label not in {"accepted", "rejected"}:
        raise ValueError("Choose the dataset label explicitly: accepted or rejected.")
    report = inspect_csv_bytes(contents)
    columns = report["columns"]
    selected_mapping = mapping or report["detected_mapping"]
    if any(feature not in FEATURE_ALIASES for feature in selected_mapping):
        raise ValueError("Mapping contains an unsupported canonical feature.")
    if any(column not in columns for column in selected_mapping.values()):
        raise ValueError("Every mapped column must exist in the uploaded CSV.")
    checksum = hashlib.sha256(contents).hexdigest()
    entries = load_registry(registry_path)
    existing = next((entry for entry in entries if entry.get("sha256") == checksum), None)
    if existing:
        return {**existing, "duplicate_file": True}
    storage_root = Path(storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._") or "dataset.csv"
    stored_name = f"{checksum[:12]}_{safe_name}"
    target = storage_root / stored_name
    target.write_bytes(contents)
    entry = {
        "dataset_id": checksum[:16], "name": Path(filename).name,
        "stored_file": str(target.resolve()), "label": label,
        "label_definition": f"User-assigned {label} source membership; not a matched lending decision.",
        "description": description.strip(), "source_url": source_url.strip(),
        "license": license_name.strip() or "Not specified by uploader",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "sha256": checksum, "file_size_bytes": len(contents),
        "row_count": report["row_count"], "column_count": report["column_count"], "columns": columns,
        "date_columns_detected": report["date_columns_detected"], "date_sample_report": report["date_sample_report"],
        "missing_counts": report["missing_counts"], "duplicate_rows": report["duplicate_rows"],
        "duplicate_rows_in_checked_rows": report["duplicate_rows_in_checked_rows"],
        "duplicate_check_rows": report["duplicate_check_rows"],
        "duplicate_check_complete": report["duplicate_check_complete"],
        "mapping": selected_mapping,
        "unmapped_columns": report["unmapped_columns"],
        "normalization_note": "Original uploaded CSV retained. Canonical values are normalized only during model sampling.",
    }
    entries.append(entry)
    _save_registry(entries, registry_path)
    return {**entry, "duplicate_file": False}


def register_existing_csv(
    path: Path,
    label: str,
    *,
    mapping: dict[str, str] | None = None,
    description: str = "",
    source_url: str = "",
    license_name: str = "Not specified by source",
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Register a discovered local CSV/GZip file without copying it into memory."""
    if label not in {"accepted", "rejected"}:
        raise ValueError("Choose the dataset label explicitly: accepted or rejected.")
    path = Path(path).resolve()
    allowed = (PROJECT_ROOT / "data").resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError("Only an existing file under the local data directory can be registered here.")
    report = inspect_csv_path(path)
    selected_mapping = mapping or report["detected_mapping"]
    if any(feature not in FEATURE_ALIASES for feature in selected_mapping):
        raise ValueError("Mapping contains an unsupported canonical feature.")
    if any(column not in report["columns"] for column in selected_mapping.values()):
        raise ValueError("Every mapped column must exist in the source CSV.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    entries = load_registry(registry_path)
    existing = next((entry for entry in entries if entry.get("sha256") == checksum), None)
    if existing:
        return {**existing, "duplicate_file": True}
    entry = {
        "dataset_id": checksum[:16], "name": path.name, "stored_file": str(path),
        "label": label, "label_definition": f"User-assigned {label} source membership; not a matched lending decision.",
        "description": description.strip(), "source_url": source_url.strip(),
        "license": license_name.strip() or "Not specified by source",
        "imported_at": datetime.now(timezone.utc).isoformat(), "sha256": checksum,
        "file_size_bytes": path.stat().st_size, "row_count": report["row_count"],
        "column_count": report["column_count"], "columns": report["columns"],
        "date_columns_detected": report["date_columns_detected"], "date_sample_report": report["date_sample_report"],
        "missing_counts": report["missing_counts"], "duplicate_rows": report["duplicate_rows"],
        "duplicate_rows_in_checked_rows": report["duplicate_rows_in_checked_rows"],
        "duplicate_check_rows": report["duplicate_check_rows"],
        "duplicate_check_complete": report["duplicate_check_complete"],
        "mapping": selected_mapping, "unmapped_columns": report["unmapped_columns"],
        "normalization_note": "Original source retained at its staged path. Canonical values are normalized only during model sampling.",
    }
    entries.append(entry)
    _save_registry(entries, registry_path)
    return {**entry, "duplicate_file": False}


def remove_dataset(dataset_id: str, *, registry_path: Path = REGISTRY_PATH, storage_root: Path = MANAGED_ROOT) -> bool:
    entries = load_registry(registry_path)
    match = next((item for item in entries if item.get("dataset_id") == dataset_id), None)
    if match is None:
        return False
    target = Path(match["stored_file"]).resolve()
    root = Path(storage_root).resolve()
    discovered_root = (PROJECT_ROOT / "data" / "discovered").resolve()
    if not (target.is_relative_to(root) or target.is_relative_to(discovered_root)) or not (target.name.endswith(".csv") or target.name.endswith(".csv.gz")):
        raise ValueError("Refusing to remove a file outside managed dataset storage.")
    target.unlink(missing_ok=True)
    entries = [item for item in entries if item.get("dataset_id") != dataset_id]
    _save_registry(entries, registry_path)
    return True


def compatible_features(entries: list[dict[str, Any]]) -> list[str]:
    """Features mapped in every selected source, in stable canonical order."""
    if not entries:
        return []
    sets = [set(entry.get("mapping", {})) for entry in entries]
    return [feature for feature in FEATURE_ALIASES if feature not in POST_DECISION_FEATURES and all(feature in values for values in sets)]

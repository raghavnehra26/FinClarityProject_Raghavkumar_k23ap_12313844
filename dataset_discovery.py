"""Curated public dataset catalog and opt-in, authenticated Kaggle staging."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kagglehub
from kagglehub.exceptions import UnauthenticatedError
import pandas as pd

from config import PROJECT_ROOT
from download_dataset import list_csv_files, select_training_csv


DISCOVERED_ROOT = PROJECT_ROOT / "data" / "discovered"
CATALOG: list[dict[str, Any]] = [
    {
        "id": "lending-club-accepted-rejected",
        "name": "All Lending Club loan data (accepted and rejected files)",
        "source": "Kaggle / wordsforthewise",
        "url": "https://www.kaggle.com/datasets/wordsforthewise/lending-club",
        "description": "The published bundle contains separate accepted and rejected historical CSV files. The exact file names and schemas are validated after download.",
        "license": "Review the current Kaggle dataset license and source terms before importing or redistributing.",
        "task": "Potentially compatible with the current historical source-membership demonstration, but file labels must still be confirmed by the user and records are unmatched populations.",
        "downloadable": True,
        "kind": "dataset_pair",
    },
    {
        "id": "lending-club-competition",
        "name": "Lending Club historical repayment (Kaggle competition)",
        "source": "Kaggle / Lending Club competition",
        "url": "https://www.kaggle.com/c/lending-club/data",
        "description": "Historical issued-loan records with a bad_loan repayment-outcome label. Authentication and acceptance of competition rules are required.",
        "license": "Subject to Kaggle competition rules; confirm terms before use or redistribution.",
        "task": "Historical repayment outcome. Not accepted/rejected source membership.",
        "downloadable": True,
        "kind": "competition",
        "filename": "loanStats_train.csv",
    },
    {
        "id": "uci-credit-approval",
        "name": "UCI Credit Approval",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/27/credit+approval",
        "description": "690 credit-card application records with a + / - target and anonymized feature names.",
        "license": "CC BY 4.0; attribution required.",
        "task": "Historical application acceptance target. Its anonymized schema and label semantics are not compatible with the current LendingClub source-membership model.",
        "downloadable": False,
        "filename": None,
    },
]


def download_curated_dataset(source_id: str, *, storage_root: Path = DISCOVERED_ROOT) -> dict[str, Any]:
    source = next((item for item in CATALOG if item["id"] == source_id), None)
    if source is None or not source["downloadable"]:
        raise ValueError("This curated source is informational only; no verified compatible download is configured.")
    storage_root = Path(storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    manifest_path = storage_root / "downloads.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        manifest = []
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Download manifest could not be read: {exc}") from exc
    existing = next((row for row in manifest if row.get("source_id") == source_id and all(Path(item.get("file_path", "")).is_file() for item in row.get("files", [])) and row.get("files")), None)
    if existing:
        return {**existing, "already_downloaded": True}
    if shutil.disk_usage(storage_root).free < 2_000_000_000:
        raise RuntimeError("At least 2 GB of free disk space is required before staging this curated Kaggle download and cache copy.")
    try:
        if source["kind"] == "competition":
            download_path = Path(kagglehub.competition_download("lending-club"))
        else:
            download_path = Path(kagglehub.dataset_download("wordsforthewise/lending-club"))
    except UnauthenticatedError as exc:
        raise RuntimeError("Kaggle authentication is required. Accept the competition rules and configure Kaggle credentials before retrying.") from exc
    csv_files = list_csv_files(download_path)
    staged_files = []
    if source["kind"] == "competition":
        source_csv = select_training_csv(csv_files, source["filename"])
        candidates = [source_csv] if source_csv is not None else []
    else:
        expected = ("accepted_2007_to_2018Q4.csv", "accepted_2007_to_2018Q4.csv.gz", "rejected_2007_to_2018Q4.csv", "rejected_2007_to_2018Q4.csv.gz")
        by_name = {path.name.casefold(): path for path in Path(download_path).rglob("*") if path.is_file()}
        candidates = [by_name[name.casefold()] for name in expected if name.casefold() in by_name]
        present = {path.name.casefold().replace(".gz", "") for path in candidates}
        if not {"accepted_2007_to_2018q4.csv", "rejected_2007_to_2018q4.csv"}.issubset(present):
            raise ValueError("The downloaded bundle did not contain both documented accepted and rejected 2007–2018Q4 CSV files.")
    for source_csv in candidates:
        if source_csv is None or not source_csv.is_file():
            raise ValueError("The expected labeled CSV was not present in the Kaggle download.")
        digest = hashlib.sha256()
        with source_csv.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        destination = storage_root / f"{checksum[:12]}_{source_csv.name}"
        if not destination.exists():
            shutil.copy2(source_csv, destination)
        try:
            preview = pd.read_csv(destination, nrows=5, low_memory=False)
        except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ValueError(f"CSV integrity/schema preflight failed for {source_csv.name}: {exc}") from exc
        if preview.empty or len(preview.columns) == 0:
            raise ValueError(f"CSV integrity/schema preflight found no data rows in {source_csv.name}.")
        staged_files.append({"filename": source_csv.name, "file_path": str(destination.resolve()), "file_size_bytes": destination.stat().st_size, "sha256": checksum, "columns": list(preview.columns)})
    if not staged_files:
        raise ValueError("No documented CSV files were available in the curated download.")
    record = {
        "source_id": source_id, "name": source["name"], "source_url": source["url"],
        "description": source["description"], "license": source["license"],
        "task_limitation": source["task"], "files": staged_files,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest.append(record)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return {**record, "already_downloaded": False}

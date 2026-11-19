"""Download Kaggle competition files and optionally select a training CSV."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import kagglehub
from kagglehub.exceptions import UnauthenticatedError

from config import DATA_PATH


COMPETITION_HANDLE = "lending-club"
NON_TRAINING_NAME = re.compile(
    r"(?:^|[_.-])(test|sample|submission)(?:[_.-]|$)", re.IGNORECASE
)


def list_csv_files(download_path: Path) -> list[Path]:
    if download_path.is_file():
        return [download_path] if download_path.suffix.casefold() == ".csv" else []
    if not download_path.is_dir():
        return []
    return sorted(
        (
            path
            for path in download_path.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".csv"
        ),
        key=lambda path: str(path).casefold(),
    )


def select_training_csv(
    csv_files: list[Path], filename: str | None
) -> Path | None:
    if filename is None:
        return None
    requested_name = Path(filename).name
    if requested_name != filename:
        raise ValueError("Provide the CSV filename only, not a path.")
    normalized_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", requested_name)
    if NON_TRAINING_NAME.search(normalized_name):
        raise ValueError(
            "The selected filename looks like a test, sample, or submission file. "
            "Choose the competition's labeled training CSV instead."
        )
    matches = [
        path for path in csv_files if path.name.casefold() == requested_name.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one downloaded CSV named {requested_name!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def copy_training_csv(source: Path, destination: Path = DATA_PATH) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite the existing dataset: {destination}. "
            "Move or rename it yourself if replacement is intentional."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Kaggle Lending Club competition files. First accept "
            "the competition rules; use --login to authenticate with KaggleHub."
        )
    )
    parser.add_argument(
        "--file",
        help=(
            "Exact filename of the labeled training CSV from the downloaded "
            "competition files. Use --select for an interactive prompt."
        ),
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Prompt to select a listed training CSV in this same process.",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help=(
            "Prompt for the Kaggle API token in this process before downloading. "
            "The token is not stored by this project."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PATH,
        help=f"Destination for the selected CSV (default: {DATA_PATH}).",
    )
    args = parser.parse_args()
    if args.select and args.file:
        parser.error("Use either --select or --file, not both.")

    if args.login:
        kagglehub.login()
    try:
        download_path = Path(kagglehub.competition_download(COMPETITION_HANDLE))
    except UnauthenticatedError:
        parser.error(
            "Kaggle authentication is required. Sign in, accept the competition "
            "rules, then rerun this script with `--login` in an interactive "
            "local terminal. Do not share your token."
        )
    print(f"Downloaded competition files: {download_path}")
    csv_files = list_csv_files(download_path)
    if not csv_files:
        parser.error(
            "KaggleHub returned no CSV files. Check the competition download "
            "contents and Kaggle access before selecting a training file."
        )
    print("Available CSV files:")
    for path in csv_files:
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")

    try:
        if args.select:
            selected_name = input(
                "Enter the exact labeled training CSV filename to copy "
                "(or press Enter to skip): "
            ).strip()
        else:
            selected_name = args.file
        source = select_training_csv(csv_files, selected_name)
        if source is not None:
            destination = copy_training_csv(source, args.output)
            print(f"Copied selected training CSV to: {destination}")
            print("Next, inspect the real schema with: python inspect_dataset.py")
        else:
            print(
                "No file was copied. Review the list, confirm the labeled "
                "training CSV, then rerun with --file <exact-filename>."
            )
    except (ValueError, FileExistsError, OSError, EOFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

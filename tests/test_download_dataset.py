from pathlib import Path
import sys

import kagglehub
import pytest
from kagglehub.exceptions import UnauthenticatedError

from download_dataset import (
    copy_training_csv,
    list_csv_files,
    main,
    select_training_csv,
)


def test_lists_csv_files_and_selects_the_exact_training_filename(tmp_path):
    training = tmp_path / "loanStats_train.csv"
    test_file = tmp_path / "loanStats_test.csv"
    training.write_text("bad_loan,loan_amnt\n0,1000\n", encoding="utf-8")
    test_file.write_text("loan_amnt\n2000\n", encoding="utf-8")

    csv_files = list_csv_files(tmp_path)

    assert set(csv_files) == {training, test_file}
    assert select_training_csv(csv_files, training.name) == training
    with pytest.raises(ValueError, match="test, sample, or submission"):
        select_training_csv(csv_files, test_file.name)


def test_rejects_camel_case_sample_submissions_and_path_names(tmp_path):
    with pytest.raises(ValueError, match="test, sample, or submission"):
        select_training_csv([tmp_path / "sampleSubmission.csv"], "sampleSubmission.csv")
    with pytest.raises(ValueError, match="filename only"):
        select_training_csv([tmp_path / "train.csv"], "..\\train.csv")


def test_selected_csv_is_copied_without_overwriting_existing_data(tmp_path):
    source = tmp_path / "training.csv"
    source.write_text("bad_loan\n0\n", encoding="utf-8")
    destination = tmp_path / "data" / "loanStats_train.csv"

    assert copy_training_csv(source, destination) == destination
    assert destination.read_text(encoding="utf-8") == "bad_loan\n0\n"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        copy_training_csv(source, destination)


def test_csv_listing_handles_single_file_and_missing_paths(tmp_path):
    csv_file = tmp_path / "single.csv"
    csv_file.write_text("x\n1\n", encoding="utf-8")
    assert list_csv_files(csv_file) == [csv_file]
    assert list_csv_files(tmp_path / "missing") == []


def test_download_cli_explains_missing_authentication(monkeypatch, capsys):
    def reject_unauthenticated_download(*args, **kwargs):
        raise UnauthenticatedError()

    monkeypatch.setattr(kagglehub, "competition_download", reject_unauthenticated_download)
    monkeypatch.setattr(sys, "argv", ["download_dataset.py"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "Kaggle authentication is required" in capsys.readouterr().err


def test_download_cli_logs_in_and_copies_selected_training_file(
    tmp_path, monkeypatch, capsys
):
    source_dir = tmp_path / "cache"
    source_dir.mkdir()
    source = source_dir / "loanStats_train.csv"
    source.write_text("bad_loan,loan_amnt\n0,1000\n", encoding="utf-8")
    destination = tmp_path / "data" / "loanStats_train.csv"
    login_calls = []
    monkeypatch.setattr(kagglehub, "login", lambda: login_calls.append(True))
    monkeypatch.setattr(
        kagglehub,
        "competition_download",
        lambda handle: str(source_dir),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_dataset.py",
            "--login",
            "--file",
            source.name,
            "--output",
            str(destination),
        ],
    )

    main()

    assert login_calls == [True]
    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert "Copied selected training CSV" in capsys.readouterr().out


def test_download_cli_selects_training_file_interactively(
    tmp_path, monkeypatch, capsys
):
    source_dir = tmp_path / "cache"
    source_dir.mkdir()
    source = source_dir / "loanStats_train.csv"
    source.write_text("bad_loan,loan_amnt\n0,1000\n", encoding="utf-8")
    destination = tmp_path / "data" / "loanStats_train.csv"
    monkeypatch.setattr(kagglehub, "login", lambda: None)
    monkeypatch.setattr(
        kagglehub,
        "competition_download",
        lambda handle: str(source_dir),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_dataset.py",
            "--login",
            "--select",
            "--output",
            str(destination),
        ],
    )
    monkeypatch.setattr("builtins.input", lambda prompt: source.name)

    main()

    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert "Copied selected training CSV" in capsys.readouterr().out

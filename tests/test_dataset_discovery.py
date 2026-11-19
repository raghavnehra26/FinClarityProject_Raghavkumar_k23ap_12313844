import pandas as pd
import pytest

from dataset_discovery import download_curated_dataset


def test_curated_kaggle_download_is_staged_with_checksum_and_deduplicated(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({"loan_amnt": [1000], "bad_loan": [1]}).to_csv(cache / "loanStats_train.csv", index=False)
    calls = []

    def fake_download(handle):
        calls.append(handle)
        return cache

    monkeypatch.setattr("dataset_discovery.kagglehub.competition_download", fake_download)
    monkeypatch.setattr("dataset_discovery.shutil.disk_usage", lambda _path: SimpleNamespace(free=5_000_000_000))
    first = download_curated_dataset("lending-club-competition", storage_root=tmp_path / "staged")
    second = download_curated_dataset("lending-club-competition", storage_root=tmp_path / "staged")
    assert first["already_downloaded"] is False
    assert second["already_downloaded"] is True
    assert first["files"][0]["sha256"]
    assert calls == ["lending-club"]


def test_informational_incompatible_source_cannot_be_downloaded(tmp_path):
    with pytest.raises(ValueError, match="informational only"):
        download_curated_dataset("uci-credit-approval", storage_root=tmp_path)

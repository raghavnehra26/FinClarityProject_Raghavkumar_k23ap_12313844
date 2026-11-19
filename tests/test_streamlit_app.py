from pathlib import Path

from config import MODEL_PATH
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_starts_without_dataset_or_model():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any("FinClarity" in title.value for title in app.title)
    if not MODEL_PATH.is_file():
        assert any("No prediction" in item.value for item in app.info)


def test_applicant_assessment_stays_disabled_until_model_exists():
    app = AppTest.from_file(str(APP_PATH)).run()
    app.radio[0].set_value("Applicant assessment").run()

    assert not app.exception
    if not MODEL_PATH.is_file():
        assert any("Predictions are disabled" in item.value for item in app.info)

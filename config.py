import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
_configured_data_path = Path(
    os.environ.get("FINCLARITY_DATA_PATH", "data/loanStats_train.csv")
).expanduser()
DATA_PATH = (
    _configured_data_path
    if _configured_data_path.is_absolute()
    else PROJECT_ROOT / _configured_data_path
)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "finclarity_model.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"
DATASET_REGISTRY_PATH = PROJECT_ROOT / "data" / "managed" / "registry.json"
MODEL_VERSIONS_DIR = MODEL_DIR / "versions"
DATASET_REPORT_PATH = REPORT_DIR / "dataset_inspection.json"
EVALUATION_REPORT_PATH = REPORT_DIR / "model_evaluation.json"
TARGET_COLUMN = "bad_loan"
TARGET_LABELS = {
    "1": "Defaulted",
    "2": "Did not default",
}

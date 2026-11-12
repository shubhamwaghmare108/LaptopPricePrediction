"""Shared ML/training configuration; no frontend framework dependency."""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

def _setting(name: str, default: str = "") -> str:
    return os.getenv(name, default)

MYSQL = {
    "user": _setting("DB_USER", "root"),
    "password": _setting("DB_PASSWORD", ""),
    "host": _setting("DB_HOST", "localhost"),
    "port": int(_setting("DB_PORT", "3306")),
    "database": _setting("DB_NAME", "laptop_data"),
    "table": _setting("DB_TABLE", "laptop_price"),
}
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_DIR = ARTIFACTS_DIR / "raw"
TRANSFORMED_DATA_DIR = ARTIFACTS_DIR / "transformed"
MODEL_DIR = ARTIFACTS_DIR / "model"
PREDICTION_MODEL_DIR = PROJECT_ROOT / "prediction" / "models"
CSV_FALLBACK_PATH = Path(_setting("LAPTOP_DATA_CSV", str(PROJECT_ROOT / "laptop_data.csv")))
if not CSV_FALLBACK_PATH.is_absolute():
    CSV_FALLBACK_PATH = PROJECT_ROOT / CSV_FALLBACK_PATH
for directory in (RAW_DATA_DIR, TRANSFORMED_DATA_DIR, MODEL_DIR, PREDICTION_MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

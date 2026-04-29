"""
Constants — All fixed paths, column names, and thresholds used across the pipeline.
Central place so nothing is ever hardcoded in component files.
"""
from pathlib import Path

# ── Root Paths ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"

# ── Config File Paths ─────────────────────────────────────────
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"
PARAMS_FILE_PATH = CONFIG_DIR / "params.yaml"
SCHEMA_FILE_PATH = CONFIG_DIR / "schema.yaml"

# ── Data Paths ────────────────────────────────────────────────
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONITORING_DATA_DIR = DATA_DIR / "monitoring"

TRAIN_CSV_PATH = RAW_DATA_DIR / "train.csv"
STORE_CSV_PATH = RAW_DATA_DIR / "store.csv"
TRAIN_PARQUET_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_PARQUET_PATH = PROCESSED_DATA_DIR / "test.parquet"
PREDICTION_LOG_PATH = MONITORING_DATA_DIR / "prediction_logs.parquet"

# ── Model Artifact Paths ──────────────────────────────────────
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
VECTOR_STORE_DIR = MODELS_DIR / "vector_store"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
LGBM_MODEL_PATH = ARTIFACTS_DIR / "lgbm_model.pkl"
XGB_MODEL_PATH = ARTIFACTS_DIR / "xgb_model.pkl"
ENSEMBLE_MODEL_PATH = ARTIFACTS_DIR / "ensemble_model.pkl"

# ── Column Names ─────────────────────────────────────────────
TARGET_COLUMN = "Sales"
DATE_COLUMN = "Date"
STORE_COLUMN = "Store"

CATEGORICAL_COLUMNS = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
NUMERICAL_COLUMNS = [
    "Customers", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2SinceWeek", "Promo2SinceYear",
]

# ── Feature Engineering Constants ────────────────────────────
LAG_DAYS = [7, 14, 28]
ROLLING_WINDOWS = [7, 30]
CYCLICAL_FEATURES = ["DayOfWeek", "Month", "WeekOfYear"]

# ── Evaluation Thresholds ─────────────────────────────────────
MAPE_THRESHOLD = 15.0       # % — model must beat this to be registered
RMSE_THRESHOLD = 1500.0
DRIFT_THRESHOLD = 0.15      # Evidently drift score threshold

# ── MLflow Constants ──────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "retail-sales-forecasting"
MLFLOW_MODEL_NAME = "retail-forecaster"
MLFLOW_PRODUCTION_STAGE = "Production"
MLFLOW_STAGING_STAGE = "Staging"

# ── Logging ──────────────────────────────────────────────────
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

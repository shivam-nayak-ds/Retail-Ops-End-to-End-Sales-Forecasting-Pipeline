"""
Config Loader — Reads config.yaml, params.yaml, schema.yaml
and returns typed dataclasses for each pipeline component.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from Retail_Ops_Pipeline.constants.config_entity import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    VECTOR_STORE_DIR,
    MONITORING_DATA_DIR,
    DOCS_DIR,
    LOG_DIR,
)


# ── Helper ────────────────────────────────────────────────────
def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ── Config Dataclasses ────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    raw_data_path: Path
    store_data_path: Path
    processed_dir: Path
    train_file: Path
    test_file: Path
    train_split_ratio: float
    date_column: str
    target_column: str


@dataclass
class DataValidationConfig:
    validation_report_dir: Path
    drift_report_file: Path
    required_columns: list[str]
    null_threshold: float
    sales_min: float
    sales_max: float


@dataclass
class DataTransformationConfig:
    preprocessor_path: Path
    lag_days: list[int]
    rolling_windows: list[int]
    cyclical_features: list[str]


@dataclass
class ModelTrainerConfig:
    model_dir: Path
    lgbm_model_path: Path
    xgb_model_path: Path
    ensemble_model_path: Path
    cv_splits: int
    random_state: int


@dataclass
class ModelEvaluationConfig:
    metrics_dir: Path
    plots_dir: Path
    mape_threshold: float
    rmse_threshold: float


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    model_name: str
    registered_model_stage: str


@dataclass
class MonitoringConfig:
    reference_data_path: Path
    drift_report_dir: Path
    drift_threshold: float
    prediction_log_path: Path


@dataclass
class GenAIConfig:
    vector_store_path: Path
    embedding_model: str
    llm_model: str
    top_k_results: int
    chunk_size: int
    chunk_overlap: int


# ── Main Config Manager ───────────────────────────────────────
class ConfigurationManager:
    """
    Central manager — reads all YAML files and returns
    typed config objects for each pipeline component.

    Usage:
        cm = ConfigurationManager()
        ingestion_cfg = cm.get_data_ingestion_config()
    """

    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH,
        schema_path: Path = SCHEMA_FILE_PATH,
    ):
        self.config = _read_yaml(config_path)
        self.params = _read_yaml(params_path)
        self.schema = _read_yaml(schema_path)

        # Ensure all required directories exist at startup
        _ensure_dirs(
            PROCESSED_DATA_DIR,
            ARTIFACTS_DIR,
            VECTOR_STORE_DIR,
            MONITORING_DATA_DIR,
            DOCS_DIR,
            LOG_DIR,
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        return DataIngestionConfig(
            raw_data_path=Path(cfg["raw_data_path"]),
            store_data_path=Path(cfg["store_data_path"]),
            processed_dir=Path(cfg["processed_dir"]),
            train_file=Path(cfg["train_file"]),
            test_file=Path(cfg["test_file"]),
            train_split_ratio=cfg["train_split_ratio"],
            date_column=cfg["date_column"],
            target_column=cfg["target_column"],
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self.config["data_validation"]
        return DataValidationConfig(
            validation_report_dir=Path(cfg["validation_report_dir"]),
            drift_report_file=Path(cfg["drift_report_file"]),
            required_columns=cfg["required_columns"],
            null_threshold=cfg["null_threshold"],
            sales_min=cfg["sales_min"],
            sales_max=cfg["sales_max"],
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config["data_transformation"]
        return DataTransformationConfig(
            preprocessor_path=Path(cfg["preprocessor_path"]),
            lag_days=cfg["lag_days"],
            rolling_windows=cfg["rolling_windows"],
            cyclical_features=cfg["cyclical_features"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        return ModelTrainerConfig(
            model_dir=Path(cfg["model_dir"]),
            lgbm_model_path=Path(cfg["lgbm_model_path"]),
            xgb_model_path=Path(cfg["xgb_model_path"]),
            ensemble_model_path=Path(cfg["ensemble_model_path"]),
            cv_splits=cfg["cv_splits"],
            random_state=cfg["random_state"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config["model_evaluation"]
        return ModelEvaluationConfig(
            metrics_dir=Path(cfg["metrics_dir"]),
            plots_dir=Path(cfg["plots_dir"]),
            mape_threshold=cfg["mape_threshold"],
            rmse_threshold=cfg["rmse_threshold"],
        )

    def get_mlflow_config(self) -> MLflowConfig:
        cfg = self.config["mlflow"]
        return MLflowConfig(
            tracking_uri=cfg["tracking_uri"],
            experiment_name=cfg["experiment_name"],
            model_name=cfg["model_name"],
            registered_model_stage=cfg["registered_model_stage"],
        )

    def get_monitoring_config(self) -> MonitoringConfig:
        cfg = self.config["monitoring"]
        return MonitoringConfig(
            reference_data_path=Path(cfg["reference_data_path"]),
            drift_report_dir=Path(cfg["drift_report_dir"]),
            drift_threshold=cfg["drift_threshold"],
            prediction_log_path=Path(cfg["prediction_log_path"]),
        )

    def get_genai_config(self) -> GenAIConfig:
        cfg = self.config["genai"]
        return GenAIConfig(
            vector_store_path=Path(cfg["vector_store_path"]),
            embedding_model=cfg["embedding_model"],
            llm_model=cfg["llm_model"],
            top_k_results=cfg["top_k_results"],
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )

    def get_lgbm_params(self) -> dict[str, Any]:
        return dict(self.params["lightgbm"])

    def get_xgb_params(self) -> dict[str, Any]:
        return dict(self.params["xgboost"])

    def get_ensemble_weights(self) -> dict[str, float]:
        return dict(self.params["ensemble"])

    def get_optuna_config(self) -> dict[str, Any]:
        return dict(self.params["optuna"])

    def get_train_schema(self) -> dict[str, Any]:
        return dict(self.schema["train_schema"])

    def get_store_schema(self) -> dict[str, Any]:
        return dict(self.schema["store_schema"])

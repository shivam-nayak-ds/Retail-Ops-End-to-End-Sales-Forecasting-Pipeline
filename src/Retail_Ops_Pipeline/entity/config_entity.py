from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_csv: Path
    store_csv: Path
    local_data_file: Path
    train_data_file: Path
    test_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    schema_path: Path
    train_data_path: Path
    required_columns: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_train_file: Path
    transformed_test_file: Path
    preprocessor_obj_file: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    base_accuracy_threshold: float
    params: dict
    search_space: dict

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_file: Path
    model_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class ModelRegistryConfig:
    root_dir: Path
    model_name: str
    status_file: Path
    model_path: Path
    metrics_path: Path

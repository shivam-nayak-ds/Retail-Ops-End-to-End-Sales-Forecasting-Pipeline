import os
import sys
from pathlib import Path
from Retail_Ops_Pipeline.constants.config_entity import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Retail_Ops_Pipeline.utils.common import read_yaml, create_directories
from Retail_Ops_Pipeline.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelRegistryConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            train_csv=Path(config.train_csv),
            store_csv=Path(config.store_csv),
            local_data_file=Path(config.local_data_file),
            train_data_file=Path(config.train_data_file),
            test_data_file=Path(config.test_data_file)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            schema_path=Path(config.schema_path),
            train_data_path=Path(config.train_data_path),
            required_columns=config.required_columns
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            transformed_train_file=Path(config.transformed_train_file),
            transformed_test_file=Path(config.transformed_test_file),
            preprocessor_obj_file=Path(config.preprocessor_obj_file)
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        search_space = self.params.search_space

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            base_accuracy_threshold=config.base_accuracy_threshold,
            params=dict(params),
            search_space=dict(search_space)
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        trainer_config = self.config.model_trainer
        transformation_config = self.config.data_transformation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            metrics_file=Path(config.metrics_file),
            model_path=Path(trainer_config.model_path),
            test_data_path=Path(transformation_config.transformed_test_file)
        )

        return model_evaluation_config

    def get_model_registry_config(self) -> ModelRegistryConfig:
        config = self.config.model_registry
        trainer_config = self.config.model_trainer
        evaluation_config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_registry_config = ModelRegistryConfig(
            root_dir=Path(config.root_dir),
            model_name=config.model_name,
            status_file=Path(config.status_file),
            model_path=Path(trainer_config.model_path),
            metrics_path=Path(evaluation_config.metrics_file)
        )

        return model_registry_config

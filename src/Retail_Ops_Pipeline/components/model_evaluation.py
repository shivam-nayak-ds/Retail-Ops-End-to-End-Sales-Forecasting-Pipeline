import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException
from Retail_Ops_Pipeline.entity.config_entity import ModelEvaluationConfig

logger = get_logger(__name__)

class ModelEvaluation:
    """
    Evaluates the performance of the trained model on the test dataset.
    Logs metrics to MLFlow for final performance assessment.
    """
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual: np.ndarray, pred: np.ndarray) -> tuple:
        """
        Calculates key regression metrics.

        Args:
            actual (np.ndarray): True target values.
            pred (np.ndarray): Predicted values.

        Returns:
            tuple: (RMSE, MAE, R2 Score)
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_evaluation(self) -> dict:
        """
        Loads the production model and evaluates it against the test set.
        Saves results locally and logs to MLFlow.

        Returns:
            dict: Calculated evaluation metrics.
        """
        try:
            logger.info("Initiating model evaluation phase.")
            
            # Loading test artifacts
            test_df = pd.read_parquet(self.config.test_data_path)
            X_test = test_df.drop(columns=['Sales'], axis=1)
            y_test = test_df['Sales']

            with open(self.config.model_path, "rb") as f:
                model = pickle.load(f)

            # Execution
            y_pred = model.predict(X_test)
            rmse, mae, r2 = self.eval_metrics(y_test, y_pred)
            
            metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2
            }

            # Local persistence of metrics
            with open(self.config.metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Evaluation metrics persisted at: {self.config.metrics_file}")

            # MLFlow Experiment Management
            mlflow.set_tracking_uri("file:mlruns")
            experiment_name = "retail-sales-forecasting"
            
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                mlflow.create_experiment(experiment_name)
            elif exp.lifecycle_stage == "deleted":
                experiment_name = f"{experiment_name}_prod"
                if not mlflow.get_experiment_by_name(experiment_name):
                    mlflow.create_experiment(experiment_name)
            
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="Final_Performance_Evaluation"):
                mlflow.log_metrics(metrics)
                logger.info("Evaluation metrics successfully logged to MLFlow.")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)

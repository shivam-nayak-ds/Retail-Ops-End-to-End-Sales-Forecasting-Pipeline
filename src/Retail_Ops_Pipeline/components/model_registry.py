import os
import sys
import json
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException
from Retail_Ops_Pipeline.entity.config_entity import ModelRegistryConfig

logger = get_logger(__name__)

class ModelRegistry:
    def __init__(self, config: ModelRegistryConfig):
        self.config = config

    def initiate_model_registry(self):
        try:
            logger.info("Starting Model Registration stage...")
            
            # Load metrics to check if model is worthy
            with open(self.config.metrics_path, "r") as f:
                metrics = json.load(f)
            
            r2_score = metrics.get("r2_score", 0)
            logger.info(f"Current Model R2 Score: {r2_score}")

            # Threshold for registration (Elite Standard)
            # In real world, we might compare with the previous version in MLFlow
            registration_threshold = 0.7 
            
            if r2_score >= registration_threshold:
                logger.info(f"Model passed threshold ({registration_threshold}). Registering to MLFlow...")
                
                mlflow.set_tracking_uri("file:mlruns")
                client = MlflowClient()
                
                # Load the model to log it properly for registration
                with open(self.config.model_path, "rb") as f:
                    model = pickle.load(f)
                
                # Start a run specifically for registration or use the last one
                # For simplicity, we create a 'Registry_Run'
                with mlflow.start_run(run_name="Model_Registration"):
                    mlflow.log_metrics(metrics)
                    # Log and register
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=self.config.model_name
                    )
                
                status = f"SUCCESS: Model registered as {self.config.model_name}"
                logger.info(status)
            else:
                status = f"SKIPPED: Model R2 score {r2_score} is below threshold {registration_threshold}."
                logger.warning(status)

            # Save registry status locally
            with open(self.config.status_file, "w") as f:
                f.write(status)
            
            return status

        except Exception as e:
            raise CustomException(e, sys)

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
    """
    Manages the model governance lifecycle, including registration and automated stage promotion.
    """
    def __init__(self, config: ModelRegistryConfig):
        self.config = config

    def initiate_model_registry(self) -> str:
        """
        Registers the model and promotes it to 'Production' if performance criteria are met.
        
        Returns:
            str: Final registration and promotion status.
        """
        try:
            logger.info("Initiating model governance and registration sequence.")
            
            with open(self.config.metrics_path, "r") as f:
                metrics = json.load(f)
            
            r2_score = metrics.get("r2_score", 0)
            registration_threshold = 0.75 

            if r2_score < registration_threshold:
                status = f"Registration skipped: Performance ({r2_score:.4f}) below threshold ({registration_threshold})."
                logger.warning(status)
                self._save_status(status)
                return status

            # MLFlow Client Interaction
            mlflow.set_tracking_uri("file:mlruns")
            client = MlflowClient()
            
            with open(self.config.model_path, "rb") as f:
                model = pickle.load(f)
            
            with mlflow.start_run(run_name="Automated_Governance_Run"):
                mlflow.log_metrics(metrics)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.config.model_name
                )

            # Automated Stage Promotion Logic
            latest_versions = client.get_latest_versions(self.config.model_name)
            if latest_versions:
                latest_version = latest_versions[0].version
                
                # Promoting to Staging first
                client.transition_model_version_stage(
                    name=self.config.model_name,
                    version=latest_version,
                    stage="Staging"
                )
                logger.info(f"Model version {latest_version} promoted to 'Staging'.")

                # Conditional Promotion to Production
                if r2_score > 0.80:
                    client.transition_model_version_stage(
                        name=self.config.model_name,
                        version=latest_version,
                        stage="Production"
                    )
                    status = f"Success: Model version {latest_version} promoted to 'Production'."
                else:
                    status = f"Success: Model registered and promoted to 'Staging'."
            
            logger.info(status)
            self._save_status(status)
            return status

        except Exception as e:
            raise CustomException(e, sys)

    def _save_status(self, status: str):
        """Persists the registry status to a local file."""
        os.makedirs(os.path.dirname(self.config.status_file), exist_ok=True)
        with open(self.config.status_file, "w") as f:
            f.write(status)

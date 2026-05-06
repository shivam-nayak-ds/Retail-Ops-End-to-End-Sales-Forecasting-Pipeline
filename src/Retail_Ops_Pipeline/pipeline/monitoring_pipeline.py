import os
import sys
from Retail_Ops_Pipeline.config.configuration import ConfigurationManager
from Retail_Ops_Pipeline.components.model_monitoring import ModelMonitoring
from Retail_Ops_Pipeline.pipeline.training_pipeline import TrainPipeline
from Retail_Ops_Pipeline.utils.logger import get_logger

logger = get_logger("MonitoringPipeline")

class MonitoringPipeline:
    """
    Elite Orchestration: Model Monitoring -> Drift Detection -> Automated Retraining.
    This creates a closed-loop MLOps system.
    """
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self) -> bool:
        """
        Runs the monitoring check. If drift is detected, triggers the retraining pipeline.
        """
        try:
            logger.info("Starting Elite Monitoring Loop.")
            
            # 1. Check if prediction logs exist (Current Data)
            monitoring_config = self.config_manager.get_model_monitoring_config()
            if not os.path.exists(monitoring_config.current_data_path):
                logger.warning(f"Prediction logs not found at {monitoring_config.current_data_path}. Bypassing drift check.")
                return False

            # 2. Run Drift Analysis
            monitoring = ModelMonitoring(config=monitoring_config)
            drift_share, drifted_cols = monitoring.get_drift_report()
            
            is_drifted = monitoring.check_drift_status(drift_share)
            
            # 3. SELF-HEALING: Trigger Retraining if Drifted
            if is_drifted:
                logger.warning("ALARM: Significant Data Drift detected! Initiating Automated Retraining Pipeline.")
                trainer = TrainPipeline()
                trainer.run()
                logger.info("SUCCESS: Model retrained and registered due to drift.")
            else:
                logger.info("STABLE: Data distribution within limits. No retraining needed.")
            
            return is_drifted

        except Exception as e:
            logger.error(f"Monitoring Loop Failed: {str(e)}")
            raise e

if __name__ == "__main__":
    try:
        pipeline = MonitoringPipeline()
        pipeline.run()
    except Exception:
        sys.exit(1)

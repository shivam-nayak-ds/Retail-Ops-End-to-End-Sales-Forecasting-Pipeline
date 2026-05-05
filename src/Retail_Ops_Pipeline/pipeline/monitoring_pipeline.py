from Retail_Ops_Pipeline.config.configuration import ConfigurationManager
from Retail_Ops_Pipeline.components.model_monitoring import ModelMonitoring
from Retail_Ops_Pipeline.utils.logger import get_logger
import sys

logger = get_logger("MonitoringPipeline")

class MonitoringPipeline:
    """
    Orchestration pipeline for model monitoring and data drift detection.
    """
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self) -> bool:
        """
        Executes the monitoring lifecycle.
        Returns:
            bool: True if significant drift is detected, False otherwise.
        """
        try:
            logger.info("Executing model monitoring pipeline.")
            monitoring_config = self.config_manager.get_model_monitoring_config()
            
            monitoring = ModelMonitoring(config=monitoring_config)
            drift_share, drifted_cols = monitoring.get_drift_report()
            
            is_drifted = monitoring.check_drift_status(drift_share)
            
            if is_drifted:
                logger.warning("Monitoring alert: Data drift detected above operational threshold.")
            else:
                logger.info("Monitoring check passed: No significant drift detected.")
            
            logger.info("Monitoring pipeline execution completed.")
            return is_drifted

        except Exception as e:
            logger.error(f"Monitoring pipeline failure: {str(e)}")
            raise e

if __name__ == "__main__":
    try:
        pipeline = MonitoringPipeline()
        pipeline.run()
    except Exception:
        sys.exit(1)

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.ui.workspace.cloud import CloudWorkspace
from Retail_Ops_Pipeline.entity.config_entity import ModelMonitoringConfig
from Retail_Ops_Pipeline.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("ModelMonitoring")

class ModelMonitoring:
    """
    Component for detecting data drift using Evidently.ai.
    Supports local report generation and remote synchronization with Evidently Cloud.
    """
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config
        self.api_key = os.getenv("EVIDENTLY_API_KEY")
        self.project_id = os.getenv("EVIDENTLY_PROJECT_ID")

    def get_drift_report(self) -> tuple:
        """
        Executes drift analysis by comparing reference and current data distributions.
        Returns:
            tuple: (drift_share, number_of_drifted_columns)
        """
        try:
            logger.info("Initiating drift detection analysis.")
            reference_df = pd.read_parquet(self.config.reference_data_path)
            current_df = pd.read_parquet(self.config.current_data_path)

            common_cols = list(set(reference_df.columns).intersection(set(current_df.columns)))
            if "Sales" in common_cols:
                common_cols.remove("Sales")
            
            logger.info(f"Analyzing drift across {len(common_cols)} features.")

            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(
                reference_data=reference_df[common_cols], 
                current_data=current_df[common_cols]
            )

            # Cloud Synchronization
            if self.api_key and self.project_id:
                try:
                    logger.info("Synchronizing report with Evidently Cloud.")
                    ws = CloudWorkspace(token=self.api_key, url="https://app.evidently.cloud")
                    ws.add_report(self.project_id, drift_report)
                    logger.info("Cloud synchronization successful.")
                except Exception as cloud_e:
                    logger.error(f"Cloud synchronization failed: {str(cloud_e)}")
            else:
                logger.warning("Cloud credentials missing; bypassing cloud synchronization.")

            # Local Persistence
            report_path = str(self.config.report_file)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            drift_report.save_html(report_path)
            logger.info(f"Local report generated at: {report_path}")
            
            result = drift_report.as_dict()
            drift_share = result['metrics'][0]['result']['drift_share']
            drifted_cols = result['metrics'][0]['result']['number_of_drifted_columns']
            
            logger.info(f"Analysis Complete: Drift Share = {drift_share:.4f}, Drifted Columns = {drifted_cols}")
            return drift_share, drifted_cols

        except Exception as e:
            logger.error(f"Critical error during drift analysis: {str(e)}")
            raise e

    def check_drift_status(self, drift_share: float) -> bool:
        """
        Evaluates if the drift share exceeds the configured threshold.
        """
        if drift_share > self.config.drift_threshold:
            logger.warning(f"Significant drift detected: {drift_share:.4f} exceeds threshold {self.config.drift_threshold}")
            return True
        
        logger.info(f"System stable: Drift share {drift_share:.4f} is within acceptable limits.")
        return False

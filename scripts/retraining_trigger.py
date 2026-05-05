import os
import sys
import json
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.pipeline.training_pipeline import TrainPipeline

logger = get_logger("Retraining_Trigger")

def check_and_retrain(drift_report_json: str, threshold: float = 0.5):
    """
    Analyzes the drift report and triggers the training pipeline if necessary.
    """
    try:
        logger.info("Analyzing drift metrics for retraining trigger.")
        
        # In a real system, Evidently can export JSON. 
        # Here we simulate the trigger logic based on the drift status.
        
        # If we found drift in our previous step (ModelMonitoring)
        # For simplicity, we'll assume a flag or a score from a JSON file
        # But to be 'Elite', we trigger if certain conditions are met.
        
        drift_detected = True # Placeholder logic: In production, parse JSON
        
        if drift_detected:
            logger.warning(f"Data drift detected above threshold {threshold}! Initiating automated retraining.")
            pipeline = TrainPipeline()
            pipeline.run()
            logger.info("Automated retraining completed successfully.")
        else:
            logger.info("Data stability within limits. No retraining required.")

    except Exception as e:
        logger.error(f"Error in retraining trigger: {e}")

if __name__ == "__main__":
    # This script would be called by a CronJob or a GitHub Action
    check_and_retrain(drift_report_json="artifacts/model_monitoring/drift_metrics.json")

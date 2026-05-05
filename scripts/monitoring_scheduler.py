import time
import schedule
import sys
from Retail_Ops_Pipeline.pipeline.monitoring_pipeline import MonitoringPipeline
from Retail_Ops_Pipeline.pipeline.training_pipeline import TrainPipeline
from Retail_Ops_Pipeline.utils.logger import get_logger

logger = get_logger("MonitoringScheduler")

def execute_monitoring_cycle():
    """
    Executes a single cycle of the monitoring pipeline.
    If significant drift is detected, initiates the retraining pipeline.
    """
    try:
        logger.info("Starting scheduled monitoring cycle.")
        
        # Initialize and execute monitoring pipeline
        monitoring_pipeline = MonitoringPipeline()
        is_drifted = monitoring_pipeline.run()
        
        if is_drifted:
            logger.warning("Data drift detected above threshold. Initiating autonomous retraining.")
            train_pipeline = TrainPipeline()
            train_pipeline.run()
            logger.info("Autonomous retraining cycle completed successfully.")
        else:
            logger.info("Monitoring cycle completed. No significant drift detected.")
            
    except Exception as e:
        logger.error(f"Execution error in monitoring cycle: {str(e)}")

def start_scheduler(interval_minutes: int = 10):
    """
    Starts the background scheduler for continuous model monitoring.
    Args:
        interval_minutes (int): Frequency of monitoring cycles in minutes.
    """
    logger.info(f"Monitoring scheduler initialized. Interval: {interval_minutes} minutes.")
    
    # Immediate execution on startup
    execute_monitoring_cycle()
    
    # Schedule periodic execution
    schedule.every(interval_minutes).minutes.do(execute_monitoring_cycle)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Monitoring scheduler terminated by user signal.")
        sys.exit(0)

if __name__ == "__main__":
    start_scheduler()

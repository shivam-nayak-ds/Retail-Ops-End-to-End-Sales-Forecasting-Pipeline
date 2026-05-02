import sys
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException
from Retail_Ops_Pipeline.pipeline.training_pipeline import TrainPipeline

logger = get_logger("Pipeline_Orchestrator")

def main():
    """
    Main entry point for the training pipeline.
    Initializes and executes the end-to-end ML workflow.
    """
    try:
        logger.info("Starting execution of the end-to-end training pipeline.")
        pipeline = TrainPipeline()
        pipeline.run()
        logger.info("End-to-end training pipeline execution completed successfully.")

    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()

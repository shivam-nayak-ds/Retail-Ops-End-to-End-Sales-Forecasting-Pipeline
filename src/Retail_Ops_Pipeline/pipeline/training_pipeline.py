import sys
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException
from Retail_Ops_Pipeline.config.configuration import ConfigurationManager
from Retail_Ops_Pipeline.components.data_ingestion import DataIngestion
from Retail_Ops_Pipeline.components.data_validation import DataValidation
from Retail_Ops_Pipeline.components.data_transformation import DataTransformation
from Retail_Ops_Pipeline.components.model_trainer import ModelTrainer
from Retail_Ops_Pipeline.components.model_evaluation import ModelEvaluation
from Retail_Ops_Pipeline.components.model_registry import ModelRegistry
from Retail_Ops_Pipeline.genai.embeddings import create_vector_db

logger = get_logger(__name__)

class TrainPipeline:
    """
    Orchestrates the sequence of stages in the retail operations sales forecasting training pipeline.
    """
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self):
        """
        Executes all stages of the training pipeline sequentially.
        """
        try:
            # STAGE 1: Data Ingestion
            logger.info("Data Ingestion stage initiated.")
            ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=ingestion_config)
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logger.info("Data Ingestion stage completed.")

            # STAGE 2: Data Validation
            logger.info("Data Validation stage initiated.")
            validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=validation_config)
            validation_status = data_validation.validate_dataset()
            
            if not validation_status:
                raise Exception("Data validation failed. Terminating pipeline.")
            
            logger.info("Data Validation stage completed successfully.")

            # STAGE 3: Data Transformation
            logger.info("Data Transformation stage initiated.")
            transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=transformation_config)
            transformed_train_path, transformed_test_path = data_transformation.initiate_data_transformation(train_path, test_path)
            logger.info("Data Transformation stage completed.")

            # STAGE 4: Model Training
            logger.info("Model Training stage initiated.")
            trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=trainer_config)
            model_path = model_trainer.initiate_model_trainer(transformed_train_path, transformed_test_path)
            logger.info("Model Training stage completed.")

            # STAGE 5: Model Evaluation
            logger.info("Model Evaluation stage initiated.")
            evaluation_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=evaluation_config)
            metrics = model_evaluation.initiate_model_evaluation()
            logger.info("Model Evaluation stage completed.")

            # STAGE 6: Model Registry
            logger.info("Model Registry stage initiated.")
            registry_config = self.config_manager.get_model_registry_config()
            model_registry = ModelRegistry(config=registry_config)
            registry_status = model_registry.initiate_model_registry()
            logger.info(f"Model Registry stage completed with status: {registry_status}")

            # STAGE 7: Vector Database (RAG) Auto-Update
            logger.info("RAG Vector DB Auto-Update initiated.")
            create_vector_db()
            logger.info("Vector DB updated with latest tabular context.")

            logger.info("Pipeline Execution Successful!")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()

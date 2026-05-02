import os
import sys
import pandas as pd
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException
from Retail_Ops_Pipeline.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion stage from CSV sources...")
        try:
            # Loading CSVs
            logger.info(f"Reading train data: {self.config.train_csv}")
            train_df = pd.read_csv(self.config.train_csv, low_memory=False)
            
            logger.info(f"Reading store data: {self.config.store_csv}")
            store_df = pd.read_csv(self.config.store_csv)

            # Merging
            logger.info("Merging train and store data...")
            df = pd.merge(train_df, store_df, on='Store', how='left')

            # Save raw merged data for audit
            df.to_parquet(self.config.local_data_file, index=False)
            logger.info(f"Saved merged raw data to {self.config.local_data_file}")

            # Splitting
            logger.info("Splitting data into train and test sets...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_parquet(self.config.train_data_file, index=False)
            test_set.to_parquet(self.config.test_data_file, index=False)

            logger.info(f"Ingestion complete. Train and Test artifacts created in {self.config.root_dir}")
            
            return (
                self.config.train_data_file,
                self.config.test_data_file
            )

        except Exception as e:
            raise CustomException(e, sys)
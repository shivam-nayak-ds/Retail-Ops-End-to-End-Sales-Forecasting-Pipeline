import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException

from Retail_Ops_Pipeline.entity.config_entity import DataValidationConfig

logger = get_logger(__name__)

class DataValidation:
    """
    Performs comprehensive data validation against a predefined schema.
    Checks for structural consistency, data types, null values, and range constraints.
    """
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self) -> bool:
        """
        Executes the data validation logic for the ingested dataset.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        try:
            validation_status = True
            error_messages = []

            # 1. Load Schema metadata
            if not self.config.schema_path.exists():
                logger.error(f"Schema definition not found at: {self.config.schema_path}")
                return False

            with open(self.config.schema_path, 'r') as f:
                schema = yaml.safe_load(f)
            
            # Combine train and store schema as ingestion merges them
            full_schema = {**schema['train_schema'], **schema['store_schema']}
            
            # 2. Read Ingested Data
            if not self.config.train_data_path.exists():
                logger.error(f"Ingested data file missing: {self.config.train_data_path}")
                return False
            
            df = pd.read_parquet(self.config.train_data_path)
            logger.info(f"Initiating validation for dataset with {len(df)} rows and {len(df.columns)} columns.")

            # --- Validation Logic ---

            # A. Duplicate Check
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(f"Detected {duplicate_count} duplicate rows in the dataset.")
            
            # B. Schema Compliance Check
            for col, meta in full_schema.items():
                # 1. Column Existence
                if col not in df.columns:
                    validation_status = False
                    msg = f"Schema consistency error: Column [{col}] is missing."
                    error_messages.append(msg)
                    logger.error(msg)
                    continue
                
                # 2. Data Type Check
                actual_dtype = str(df[col].dtype)
                expected_dtype = meta['dtype']
                if expected_dtype not in actual_dtype and not (expected_dtype == 'float64' and 'int' in actual_dtype):
                     logger.warning(f"Data type mismatch for [{col}]: Expected {expected_dtype}, found {actual_dtype}")

                # 3. Null Value Constraint
                null_count = df[col].isnull().sum()
                is_nullable = meta.get('nullable', True)
                if not is_nullable and null_count > 0:
                    validation_status = False
                    msg = f"Null constraint violation: [{col}] contains {null_count} null values."
                    error_messages.append(msg)
                    logger.error(msg)

                # 4. Range Validation
                if 'min' in meta:
                    actual_min = df[col].min()
                    if actual_min < meta['min']:
                        validation_status = False
                        msg = f"Range error: [{col}] minimum value {actual_min} is below threshold {meta['min']}."
                        error_messages.append(msg)
                        logger.error(msg)
                
                if 'max' in meta:
                    actual_max = df[col].max()
                    if actual_max > meta['max']:
                        validation_status = False
                        msg = f"Range error: [{col}] maximum value {actual_max} is above threshold {meta['max']}."
                        error_messages.append(msg)
                        logger.error(msg)

                # 5. Categorical Integrity
                if 'allowed_values' in meta:
                    unique_values = df[col].dropna().unique()
                    invalid_vals = set(unique_values) - set(meta['allowed_values'])
                    if invalid_vals:
                        validation_status = False
                        msg = f"Data integrity error: [{col}] contains undefined categories: {invalid_vals}"
                        error_messages.append(msg)
                        logger.error(msg)

            # --- Status Reporting ---
            os.makedirs(self.config.status_file.parent, exist_ok=True)
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}\n")
                if not validation_status:
                    f.write("Critical validation errors identified:\n")
                    for err in error_messages:
                        f.write(f"- {err}\n")

            if validation_status:
                logger.info("Data validation pipeline completed successfully.")
            else:
                logger.error(f"Data validation failed. Detailed report available at: {self.config.status_file}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test block
    validator = DataValidation()
    res = validator.validate_dataset()
    print(f"Validation Success: {res}")

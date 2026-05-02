import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException

from Retail_Ops_Pipeline.config.configuration import ConfigurationManager

logger = get_logger(__name__)

class PredictionPipeline:
    """
    Orchestrates the prediction process using the trained model and preprocessor.
    """
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.transformation_config = self.config_manager.get_data_transformation_config()
        self.trainer_config = self.config_manager.get_model_trainer_config()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Transforms input features and returns model predictions.

        Args:
            features (pd.DataFrame): Raw input features.

        Returns:
            np.ndarray: Predicted sales values.
        """
        try:
            logger.info("Initiating prediction sequence.")
            
            features = self._apply_feature_engineering(features)

            with open(self.transformation_config.preprocessor_obj_file, "rb") as f:
                preprocessor = pickle.load(f)
            
            with open(self.trainer_config.model_path, "rb") as f:
                model = pickle.load(f)

            transformed_data = preprocessor.transform(features)
            log_preds = model.predict(transformed_data)
            final_preds = np.expm1(log_preds)

            logger.info("Prediction sequence completed successfully.")
            return final_preds

        except Exception as e:
            raise CustomException(e, sys)

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies necessary feature engineering transformations to input data.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe with engineered features.
        """
        try:
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

            required_cols = [
                'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 
                'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_roll_mean_7',
                'Day', 'Month', 'Year', 'WeekOfYear', 'StoreType', 'Assortment', 'StateHoliday'
            ]
            
            return df[required_cols]

        except Exception as e:
            raise CustomException(e, sys)

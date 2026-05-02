import os
import sys
import pandas as pd
import numpy as np
import pickle
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException
from Retail_Ops_Pipeline.entity.config_entity import DataTransformationConfig

logger = get_logger(__name__)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class DataTransformation:
    """
    Handles data preprocessing and feature engineering.
    Prepares raw datasets for model training by applying transformations and scaling.
    """
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer_object(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer object for automated feature scaling and encoding.

        Returns:
            ColumnTransformer: Fitted preprocessing object.
        """
        try:
            num_cols = [
                'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 
                'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_roll_mean_7',
                'Day', 'Month', 'Year', 'WeekOfYear'
            ]
            cat_cols = ['StoreType', 'Assortment', 'StateHoliday']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def get_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies manual feature engineering including lags, rolling means, and date decomposition.

        Args:
            df (pd.DataFrame): Raw dataframe.

        Returns:
            pd.DataFrame: Dataframe with engineered features.
        """
        try:
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(['Store', 'Date'], inplace=True)

            # Lag and Rolling window features
            for lag in [1, 7, 30]:
                df[f'sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
            df['sales_roll_mean_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(window=7).mean())
            
            # Temporal features
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

            df.drop(columns=['Date'], inplace=True)
            df.dropna(inplace=True)
            
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
        Executes the end-to-end data transformation process.

        Args:
            train_path (str): Path to raw training data.
            test_path (str): Path to raw test data.

        Returns:
            tuple: (Transformed train file path, Transformed test file path)
        """
        try:
            logger.info("Initiating data transformation sequence.")
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)

            logger.info("Applying feature engineering transformations.")
            train_df = self.get_feature_engineering(train_df)
            test_df = self.get_feature_engineering(test_df)

            target_column_name = "Sales"
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = np.log1p(train_df[target_column_name])

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = np.log1p(test_df[target_column_name])

            preprocessor = self.get_transformer_object()
            
            logger.info("Applying ColumnTransformer to training and test sets.")
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Recovering feature names for structured storage
            ohe_cols = preprocessor.named_transformers_['cat_pipeline'].named_steps['onehot'].get_feature_names_out(['StoreType', 'Assortment', 'StateHoliday'])
            all_cols = [
                'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance', 
                'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_roll_mean_7',
                'Day', 'Month', 'Year', 'WeekOfYear'
            ] + list(ohe_cols)
            
            train_final_df = pd.DataFrame(X_train_arr, columns=all_cols)
            train_final_df['Sales'] = y_train.values
            
            test_final_df = pd.DataFrame(X_test_arr, columns=all_cols)
            test_final_df['Sales'] = y_test.values

            train_final_df.to_parquet(self.config.transformed_train_file, index=False)
            test_final_df.to_parquet(self.config.transformed_test_file, index=False)

            # Exporting preprocessor object
            with open(self.config.preprocessor_obj_file, "wb") as f:
                pickle.dump(preprocessor, f)

            logger.info(f"Data transformation completed. Preprocessor saved at: {self.config.preprocessor_obj_file}")
            
            return (
                self.config.transformed_train_file,
                self.config.transformed_test_file
            )

        except Exception as e:
            raise CustomException(e, sys)

        except Exception as e:
            raise CustomException(e, sys)

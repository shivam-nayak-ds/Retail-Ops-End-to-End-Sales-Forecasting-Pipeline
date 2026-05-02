import os
import sys
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.utils.exception import CustomException
from Retail_Ops_Pipeline.entity.config_entity import ModelTrainerConfig

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

logger = get_logger(__name__)

class ModelTrainer:
    """
    Handles the training, selection, and fine-tuning of different regression models.
    Logs metrics, parameters, and feature importance to MLFlow.
    """
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def eval_metrics(self, actual: np.ndarray, pred: np.ndarray) -> tuple:
        """Calculates RMSE and R2 Score."""
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        return rmse, r2

    def initiate_model_trainer(self, train_array_path: str, test_array_path: str) -> str:
        """
        Executes the model training sequence including hyperparameter tuning.
        """
        try:
            # MLFlow Configuration
            mlflow.set_tracking_uri("file:mlruns")
            experiment_name = "retail-sales-forecasting"
            exp = mlflow.get_experiment_by_name(experiment_name)
            
            if exp is None:
                mlflow.create_experiment(experiment_name)
            elif exp.lifecycle_stage == "deleted":
                experiment_name = f"{experiment_name}_prod"
                if not mlflow.get_experiment_by_name(experiment_name):
                    mlflow.create_experiment(experiment_name)
            
            mlflow.set_experiment(experiment_name)

            logger.info("Loading transformed datasets.")
            train_df = pd.read_parquet(train_array_path)
            test_df = pd.read_parquet(test_array_path)

            X_train, y_train = train_df.drop(columns=['Sales'], axis=1), train_df['Sales']
            X_test, y_test = test_df.drop(columns=['Sales'], axis=1), test_df['Sales']

            models = {
                "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=42),
                "XGBoost": XGBRegressor(n_jobs=-1, random_state=42),
                "LightGBM": lgb.LGBMRegressor(n_jobs=-1, random_state=42, verbosity=-1)
            }

            search_spaces = self.config.search_space
            model_report = {}
            trained_models = {}

            for model_name, model in models.items():
                logger.info(f"Initiating hyperparameter tuning for {model_name}.")
                
                param_grid = search_spaces.get(model_name, {})
                
                # GridSearchCV for automated tuning
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=1
                )
                
                with mlflow.start_run(run_name=f"{model_name}_Tuning", nested=True):
                    grid_search.fit(X_train, y_train)
                    
                    best_model = grid_search.best_estimator_
                    trained_models[model_name] = best_model
                    
                    y_pred = best_model.predict(X_test)
                    rmse, r2 = self.eval_metrics(y_test, y_pred)
                    
                    model_report[model_name] = r2
                    
                    # Logging best parameters and metrics
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2_score", r2)
                    
                    # Feature Importance Logging
                    self._log_feature_importance(best_model, X_train.columns, model_name)
                    
                    mlflow.sklearn.log_model(best_model, "model")

            # Selection of the best performing model
            best_score = max(model_report.values())
            best_model_name = [n for n, s in model_report.items() if s == best_score][0]
            best_model = trained_models[best_model_name]

            if best_score < self.config.base_accuracy_threshold:
                raise Exception(f"Best model performance ({best_score}) failed to meet threshold.")

            logger.info(f"Optimal model selected: {best_model_name} (R2: {best_score:.4f})")

            # Final persistence
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            with open(self.config.model_path, "wb") as f:
                pickle.dump(best_model, f)

            return self.config.model_path

        except Exception as e:
            raise CustomException(e, sys)

    def _log_feature_importance(self, model, feature_names, model_name):
        """Generates and logs feature importance plot to MLFlow."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(12, 8))
                plt.title(f"Feature Importance - {model_name}")
                sns.barplot(x=importances[indices][:15], y=feature_names[indices][:15])
                plt.tight_layout()
                
                plot_path = os.path.join(self.config.root_dir, f"feat_imp_{model_name}.png")
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path)
                plt.close()
                logger.info(f"Logged feature importance for {model_name}.")
        except Exception as e:
            logger.warning(f"Could not log feature importance for {model_name}: {e}")

        except Exception as e:
            raise CustomException(e, sys)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
from typing import List, Optional
from Retail_Ops_Pipeline.pipeline.prediction_pipeline import PredictionPipeline
from Retail_Ops_Pipeline.pipeline.training_pipeline import TrainPipeline
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.constants.config_entity import PREDICTION_LOG_PATH
from prometheus_fastapi_instrumentator import Instrumentator

# Configuration and Initialization
logger = get_logger("Retail_Ops_API")
app = FastAPI(
    title="Retail Operations Sales Forecasting API",
    description="Enterprise-grade inference service for store sales prediction using gradient boosting ensembles.",
    version="1.0.0"
)

# Step 2: Initialize Instrumentator
Instrumentator().instrument(app).expose(app)

class SalesInferenceInput(BaseModel):
    """Data model for sales inference requests."""
    Store: int
    DayOfWeek: int
    Date: str
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    sales_lag_1: float
    sales_lag_7: float
    sales_lag_30: float
    sales_roll_mean_7: float

class PredictionResponse(BaseModel):
    """Data model for API response."""
    status: str
    predictions: List[float]

@app.get("/")
async def health_check():
    """Service health check endpoint."""
    return {
        "service": "Retail Ops Forecasting API",
        "status": "online",
        "documentation": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: List[SalesInferenceInput]):
    """
    Inference endpoint for generating sales forecasts.
    Logs input features asynchronously for drift monitoring.
    """
    try:
        logger.info(f"Received inference request for {len(data)} records.")
        
        # Convert request body to DataFrame
        input_dicts = [item.dict() for item in data]
        df = pd.DataFrame(input_dicts)
        
        # Execute Prediction Pipeline
        pipeline = PredictionPipeline()
        predictions = pipeline.predict(df)
        
        # Asynchronous logging for monitoring
        try:
            os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
            if os.path.exists(PREDICTION_LOG_PATH):
                existing_logs = pd.read_parquet(PREDICTION_LOG_PATH)
                updated_logs = pd.concat([existing_logs, df], ignore_index=True)
                updated_logs.to_parquet(PREDICTION_LOG_PATH, index=False)
            else:
                df.to_parquet(PREDICTION_LOG_PATH, index=False)
            logger.info("Input features logged successfully for drift analysis.")
        except Exception as log_error:
            logger.error(f"Monitoring log failure: {str(log_error)}")

        return PredictionResponse(
            status="success",
            predictions=predictions.tolist()
        )

    except Exception as e:
        logger.error(f"Inference Failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers the model retraining pipeline as a background task.
    """
    try:
        logger.info("Manual retraining trigger initiated.")
        
        def execute_training():
            try:
                pipeline = TrainPipeline()
                pipeline.run()
                logger.info("Background retraining cycle completed successfully.")
            except Exception as train_error:
                logger.error(f"Background training failed: {str(train_error)}")

        background_tasks.add_task(execute_training)
        
        return {
            "status": "initiated",
            "message": "Retraining pipeline is executing in the background."
        }

    except Exception as e:
        logger.error(f"Training Trigger Failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate training pipeline.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

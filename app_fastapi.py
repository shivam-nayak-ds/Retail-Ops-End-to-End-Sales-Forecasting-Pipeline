import time
import os
import asyncio
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager

from Retail_Ops_Pipeline.pipeline.prediction_pipeline import PredictionPipeline
from Retail_Ops_Pipeline.genai.rag_pipeline import RetailRAGPipeline
from Retail_Ops_Pipeline.utils.logger import get_logger
from Retail_Ops_Pipeline.constants.config_entity import PREDICTION_LOG_PATH, TRAIN_PARQUET_PATH

# Initialize logger 
logger = get_logger("Retail_Ops_API")

# --- 1. Elite Schemas ---
class SalesForecastInput(BaseModel):
    store: int = Field(..., gt=0, description="Unique Store ID", example=1)
    DayOfWeek: int = Field(..., ge=1, le=7, description="Day of the week (1-7)")
    Date: str = Field(..., description="Date in the format YYYY-MM-DD")
    open: int = Field(..., ge=0, le=1, description="Is store open? (0 or 1)")
    Promo: int = Field(..., ge=0, le=1, description="Is there a promotion? (0 or 1)")
    StateHoliday: str = Field(..., description="State holiday indicator (a, b, c or 0)")
    SchoolHoliday: int = Field(..., ge=0, le=1, description="School holiday indicator")
    StoreType: str = Field(..., description="Type of store (a, b, c, d)")
    Assortment: str = Field(..., description="Assortment level (a, b, c)")
    CompetitionDistance: float = Field(..., description="Distance to nearest competitor")

    @field_validator('Date')
    @classmethod
    def validate_date(cls, value):
        try:
            datetime.strptime(value, '%Y-%m-%d')
            return value
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

class ForecastResponse(BaseModel):
    status: str = "success"
    store_id: int
    prediction: float
    analysis: Optional[str] = None
    request_timestamp: datetime = Field(default_factory=datetime.now)
    model_info: Dict[str, str] = {"version": "2.1.0", "algorithm": "LightGBM"}
    latency: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "store_id": 1,
                "prediction": 15420.50,
                "analysis": "Sales are predicted to be strong...",
                "model_info": {"version": "2.1.0", "algorithm": "LightGBM"},
                "latency": "0.45s"
            }
        }
    }

# --- 2. Service Layer (Lifecycle Management) ---
def load_latest_lags():
    """Helper to create a lookup table for the latest lag features per store."""
    try:
        # Convert Path object to string for pandas
        df = pd.read_parquet(str(TRAIN_PARQUET_PATH))
        # Get the latest record for each store
        latest_lags = df.sort_values('Date').groupby('Store').tail(1)
        # Columns to keep for inference
        lag_cols = ['sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_roll_mean_7']
        lookup = latest_lags.set_index('Store')[lag_cols].to_dict('index')
        return lookup
    except Exception as e:
        logger.error(f"Feature Store Initialization Failed: {e}")
        return {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan event for loading models and pipelines once on startup.
    """
    try:
        app.state.predictor = PredictionPipeline()
        app.state.rag_pipeline = RetailRAGPipeline()
        app.state.feature_store = load_latest_lags()
        logger.info("Service Layer: Prediction, RAG, and Feature Store initialized successfully.")
        yield
    except Exception as e:
        logger.error(f"Startup Failure: {str(e)}")
        yield

# Initialize FastAPI App
app = FastAPI(
    title="Retail Intelligence API",
    description="Enterprise-grade sales forecasting with AI-powered RAG analysis.",
    version="2.1.0",
    lifespan=lifespan
)

# Initialize Prometheus Monitoring
Instrumentator().instrument(app).expose(app)

# --- 3. Business Logic Helper ---
def log_prediction(data: pd.DataFrame):
    """Background task to log predictions for drift monitoring."""
    try:
        os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
        if os.path.exists(PREDICTION_LOG_PATH):
            existing = pd.read_parquet(PREDICTION_LOG_PATH)
            pd.concat([existing, data]).to_parquet(PREDICTION_LOG_PATH, index=False)
        else:
            data.to_parquet(PREDICTION_LOG_PATH, index=False)
    except Exception as e:
        logger.error(f"Failed to log prediction data: {e}")

# --- 4. The Elite Endpoint ---
@app.post("/forecast", response_model=ForecastResponse)
async def forecast(data: SalesForecastInput, background_tasks: BackgroundTasks):
    """
    Unified Endpoint: Performs ML Inference and RAG Analysis in Parallel.
    """
    start_time = time.time()
    logger.info(f"Received forecast request for Store: {data.store}")

    try:
        # Prepare data dictionary
        input_data = data.model_dump()
        
        # --- Automated Feature Engineering & Mapping ---
        store_id = input_data['store']
        
        # Mapping: Rename to match model's expected column names (Title Case)
        input_data['Store'] = input_data.pop('store')
        input_data['Open'] = input_data.pop('open')
        # DayOfWeek is already capital in schema now
        
        # Inject lags from our 'Feature Store'
        if store_id in app.state.feature_store:
            lags = app.state.feature_store[store_id]
            input_data.update(lags)
            logger.info(f"Successfully injected historical lags for Store {store_id}")
        else:
            input_data.update({
                'sales_lag_1': 0.0, 'sales_lag_7': 0.0, 
                'sales_lag_30': 0.0, 'sales_roll_mean_7': 0.0
            })
            logger.warning(f"No history found for Store {store_id}. Defaulting lags.")

        input_df = pd.DataFrame([input_data])
        
        # 1. Run Prediction first (so AI knows the value)
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(None, app.state.predictor.predict, input_df)
        prediction_value = float(predictions[0])
        
        # 2. Run AI Analysis with the ACTUAL prediction
        analysis = await loop.run_in_executor(None, app.state.rag_pipeline.explain_forecast, input_data, prediction_value)

        # Logging in background
        background_tasks.add_task(log_prediction, input_df)

        latency = f"{time.time() - start_time:.2f}s"

        return ForecastResponse(
            store_id=store_id,
            prediction=round(prediction_value, 2),
            analysis=analysis,
            latency=latency
        )

    except Exception as e:
        logger.error(f"Inference Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal AI Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

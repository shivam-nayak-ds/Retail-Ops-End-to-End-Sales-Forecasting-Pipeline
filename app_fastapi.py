import os
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import os
import asyncio
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager

from Retail_Ops_Pipeline.pipeline.prediction_pipeline import PredictionPipeline
from Retail_Ops_Pipeline.genai.rag_pipeline import RetailRAGPipeline
from Retail_Ops_Pipeline.utils.logger import get_logger
import shap
from Retail_Ops_Pipeline.constants.config_entity import PREDICTION_LOG_PATH, TRAIN_PARQUET_PATH, PREPROCESSOR_PATH
from prometheus_client import Gauge
from Retail_Ops_Pipeline.components.model_monitoring import ModelMonitoring
from Retail_Ops_Pipeline.entity.config_entity import ModelMonitoringConfig

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
            dt = datetime.strptime(value, '%Y-%m-%d')
            if dt.year < 2013 or dt.year > 2026:
                raise ValueError('Date must be between 2013 and 2026 for reliable forecasting.')
            return value
        except ValueError as e:
            if 'between' in str(e): raise e
            raise ValueError('Date must be in YYYY-MM-DD format')

class ForecastResponse(BaseModel):
    status: str = "success"
    store_id: int
    prediction: float
    analysis: Optional[str] = None
    explanation: Optional[Dict[str, float]] = None
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

class ForecastRangeInput(BaseModel):
    store: int = Field(..., gt=0, description="Unique Store ID", example=1)
    start_date: str = Field(..., description="Start Date (YYYY-MM-DD)")
    days: int = Field(7, ge=1, le=30, description="Number of days to forecast (1-30)")
    open_status: int = Field(1, ge=0, le=1, description="Default open status for the period")
    promo_active: int = Field(0, ge=0, le=1, description="Default promo status for the period")
    store_type: str = Field('a', description="Store type (a, b, c, d)")
    assortment: str = Field('a', description="Assortment level (a, b, c)")
    competition_distance: float = Field(1270.0, description="Distance to nearest competitor (meters)")

    @field_validator('start_date')
    @classmethod
    def validate_start_date(cls, value):
        try:
            dt = datetime.strptime(value, '%Y-%m-%d')
            if dt.year < 2013 or dt.year > 2026:
                raise ValueError('Start Date must be between 2013 and 2026.')
            return value
        except ValueError as e:
            if 'between' in str(e): raise e
            raise ValueError('Date must be in YYYY-MM-DD format')

class RangeForecastResponse(BaseModel):
    status: str = "success"
    store_id: int
    start_date: str
    end_date: str
    daily_forecasts: List[Dict[str, Any]]
    total_estimated_sales: float
    strategic_analysis: Optional[str] = None
    latency: Optional[str] = None

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
        # 1. Predictor
        app.state.predictor = PredictionPipeline()
        
        # 2. RAG Pipeline
        app.state.rag_pipeline = RetailRAGPipeline()
        
        # 3. Feature Store
        app.state.feature_store = load_latest_lags()
        
        # 4. SHAP Explainer (Optional/Robust)
        try:
            model = app.state.predictor.model
            app.state.explainer = shap.TreeExplainer(model)
            logger.info("SHAP Explainer initialized.")
        except Exception as se:
            logger.warning(f"SHAP Initialization skipped: {se}")
            app.state.explainer = None
            
        logger.info("Service Layer: All components initialized.")
        yield
    except Exception as e:
        logger.error(f"Critical Startup Failure: {str(e)}")
        # Provide null objects to prevent API crashes on attribute access
        app.state.predictor = None
        app.state.rag_pipeline = None
        app.state.explainer = None
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

# --- SECURITY: API Key Authentication ---
API_KEY_NAME = "X-API-KEY"
API_KEY = os.environ.get("RETAIL_API_KEY", "retail-ops-elite-key-2024")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Forbidden: Invalid or Missing API Key")


# ── Layer 1: FastAPI / HTTP Metrics (auto via Instrumentator above) ────────
# (http_requests_total, http_request_duration_seconds auto-generated)

# ── Layer 2: Model Metrics ─────────────────────────────────────────────────
PREDICTION_VALUE    = Gauge('model_prediction_value',      'Latest sales prediction (€)',           ['store_id'])
PREDICTION_LATENCY  = Gauge('model_inference_latency_seconds', 'Time taken for ML model inference', ['endpoint'])
PREDICTION_COUNTER  = Gauge('model_total_predictions',      'Total predictions served',             [])

# ── Layer 3: Drift Metrics ─────────────────────────────────────────────────
DRIFT_SHARE         = Gauge('drift_share',                  'Fraction of features with drift (0-1)', [])
DRIFTED_COLS        = Gauge('drift_drifted_columns_count',  'Number of features with drift',         [])
DRIFT_RETRAIN_FLAG  = Gauge('drift_retraining_required',    '1 if retraining is required, else 0',  [])

# ── Layer 4: RAG Metrics ───────────────────────────────────────────────────
RAG_LATENCY         = Gauge('rag_inference_latency_seconds','Time taken for RAG + LLM call',        [])
RAG_SUCCESS_COUNTER = Gauge('rag_success_total',            'Total successful RAG responses',        [])
RAG_FAILURE_COUNTER = Gauge('rag_failure_total',            'Total failed RAG responses',            [])

# Running counters
_prediction_count = 0

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

# --- 4. The Elite Endpoints ---
@app.post("/forecast/range", response_model=RangeForecastResponse)
async def forecast_range(data: ForecastRangeInput, api_key: str = Depends(get_api_key)):
    """
    Strategic Endpoint: Performs Recursive Multi-Step Forecasting for a range of days.
    """
    start_time = time.time()
    logger.info(f"Strategic request for Store: {data.store} for {data.days} days.")

    try:
        current_date = datetime.strptime(data.start_date, '%Y-%m-%d')
        store_id = data.store
        
        # Initial Lags from Feature Store
        if store_id not in app.state.feature_store:
            current_lags = {'sales_lag_1': 0.0, 'sales_lag_7': 0.0, 'sales_lag_30': 0.0, 'sales_roll_mean_7': 0.0}
        else:
            current_lags = app.state.feature_store[store_id].copy()

        daily_results = []
        total_sales = 0.0
        
        # Recursive Forecasting Loop
        for i in range(data.days):
            iter_date = current_date + pd.Timedelta(days=i)
            
            # Prepare Input Features for this step
            step_input = {
                'Store': store_id,
                'DayOfWeek': iter_date.weekday() + 1,
                'Date': iter_date.strftime('%Y-%m-%d'),
                'Open': data.open_status,
                'Promo': data.promo_active,
                'StateHoliday': '0',
                'SchoolHoliday': 0,
                'StoreType': data.store_type,
                'Assortment': data.assortment,
                'CompetitionDistance': data.competition_distance
            }
            step_input.update(current_lags)
            
            # Predict this step
            input_df = pd.DataFrame([step_input])
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(None, app.state.predictor.predict, input_df)
            pred_val = float(prediction[0])
            
            # Recursive Update
            current_lags['sales_lag_1'] = pred_val
            
            daily_results.append({
                "date": step_input['Date'],
                "prediction": round(pred_val, 2)
            })
            total_sales += pred_val

        # Strategic Analysis
        analysis = await loop.run_in_executor(None, app.state.rag_pipeline.explain_forecast, {"Store": store_id, "period": f"{data.days} days"}, total_sales)

        latency = f"{time.time() - start_time:.2f}s"
        
        return RangeForecastResponse(
            store_id=store_id,
            start_date=data.start_date,
            end_date=daily_results[-1]['date'],
            daily_forecasts=daily_results,
            total_estimated_sales=round(total_sales, 2),
            strategic_analysis=analysis,
            latency=latency
        )

    except Exception as e:
        logger.error(f"Range Inference Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Strategic Engine Error: {str(e)}")

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(data: SalesForecastInput, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
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
        rag_start = time.time()
        try:
            analysis = await loop.run_in_executor(None, app.state.rag_pipeline.explain_forecast, input_data, prediction_value)
            RAG_LATENCY.set(time.time() - rag_start)
            RAG_SUCCESS_COUNTER.set(RAG_SUCCESS_COUNTER._value.get() + 1)
        except Exception as rag_e:
            logger.warning(f"RAG analysis failed: {rag_e}")
            analysis = "AI analysis unavailable."
            RAG_FAILURE_COUNTER.set(RAG_FAILURE_COUNTER._value.get() + 1)
        
        # 3. Calculate SHAP Explanation (Robust check)
        top_features = {}
        if app.state.explainer:
            try:
                shap_values = app.state.explainer.shap_values(input_df)
                if isinstance(shap_values, list): shap_values = shap_values[0]
                feature_names = input_df.columns.tolist()
                importances = dict(zip(feature_names, shap_values[0]))
                top_features = dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
        else:
            logger.warning("SHAP Explainer not available.")

        # ── Record all Prometheus Metrics ──
        global _prediction_count
        elapsed = time.time() - start_time
        _prediction_count += 1

        # Layer 2: Model metrics
        PREDICTION_VALUE.labels(store_id=str(store_id)).set(prediction_value)
        PREDICTION_LATENCY.labels(endpoint='/forecast').set(elapsed)
        PREDICTION_COUNTER.set(_prediction_count)

        # Layer 4: RAG metrics (track separately)
        # Logging in background
        background_tasks.add_task(log_prediction, input_df)

        latency = f"{elapsed:.2f}s"

        return ForecastResponse(
            store_id=store_id,
            prediction=round(prediction_value, 2),
            analysis=analysis,
            explanation=top_features,
            latency=latency
        )

    except Exception as e:
        logger.error(f"Inference Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal AI Error: {str(e)}")

@app.get("/monitor/drift")
async def check_drift():
    """
    Evaluates statistical data drift between training baseline and live inference logs.
    """
    try:
        if not os.path.exists(PREDICTION_LOG_PATH):
            return {"status": "skipped", "message": "No live prediction logs available yet to calculate drift."}

        # Setup Config
        config = ModelMonitoringConfig(
            root_dir=Path("artifacts/monitoring"),
            reference_data_path=str(TRAIN_PARQUET_PATH),
            current_data_path=str(PREDICTION_LOG_PATH),
            report_file="artifacts/monitoring/drift_report.html",
            drift_threshold=0.15
        )
        
        monitor = ModelMonitoring(config)
        
        # Run calculation in background thread to avoid blocking API
        loop = asyncio.get_event_loop()
        drift_share, drifted_cols = await loop.run_in_executor(None, monitor.get_drift_report)
        is_drifting = monitor.check_drift_status(drift_share)

        # Layer 3: Record drift Prometheus metrics
        DRIFT_SHARE.set(drift_share)
        DRIFTED_COLS.set(drifted_cols)
        DRIFT_RETRAIN_FLAG.set(1 if is_drifting else 0)

        return {
            "status": "success",
            "drift_share": round(drift_share, 4),
            "drifted_columns_count": drifted_cols,
            "requires_retraining": is_drifting
        }
    except Exception as e:
        logger.error(f"Drift Calculation Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Monitoring Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

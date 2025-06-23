"""
FastAPI application for Boston Housing Price Prediction.
"""

import os
import sys
import uuid
import time
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.schemas import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, MetricsResponse, ErrorResponse,
    HousingFeatures
)
from models.trainer import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Prometheus metrics
PREDICTION_COUNTER = Counter('housing_predictions_total', 'Total number of predictions made')
BATCH_COUNTER = Counter('housing_batch_predictions_total', 'Total number of batch predictions made')
REQUEST_DURATION = Histogram('housing_request_duration_seconds', 'Request duration in seconds')
ERROR_COUNTER = Counter('housing_errors_total', 'Total number of errors', ['error_type'])
MODEL_LOAD_GAUGE = Gauge('housing_model_loaded', 'Whether model is loaded (1=loaded, 0=not loaded)')
ACTIVE_REQUESTS = Gauge('housing_active_requests', 'Number of active requests')

# Global variables
model = None
model_version = "unknown"
start_time = time.time()
prediction_count = 0
last_prediction_time = None

# Initialize FastAPI app
app = FastAPI(
    title=config['api']['title'],
    description=config['api']['description'],
    version=config['api']['version'],
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelPredictor:
    """Handle model predictions with monitoring."""
    
    def __init__(self):
        self.trainer = ModelTrainer()
        self.scaler = None
        
    def load_model(self):
        """Load the latest model."""
        global model, model_version
        try:
            model = self.trainer.load_latest_model()
            model_version = "1.0.0"  # This should come from MLflow in production
            MODEL_LOAD_GAUGE.set(1)
            logger.info(f"Model loaded successfully, version: {model_version}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            MODEL_LOAD_GAUGE.set(0)
            raise
    
    def preprocess_features(self, features: HousingFeatures) -> pd.DataFrame:
        """Preprocess input features."""
        # Convert to DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        # Add engineered features (same as training)
        df['rm_squared'] = df['rm'] ** 2
        df['age_per_room'] = df['age'] / df['rm']
        
        return df
    
    def predict_single(self, features: HousingFeatures) -> float:
        """Make single prediction."""
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess features
        df = self.preprocess_features(features)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return float(prediction)
    
    def predict_batch(self, features_list: List[HousingFeatures]) -> List[float]:
        """Make batch predictions."""
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess all features
        feature_dicts = [f.dict() for f in features_list]
        df = pd.DataFrame(feature_dicts)
        
        # Add engineered features
        df['rm_squared'] = df['rm'] ** 2
        df['age_per_room'] = df['age'] / df['rm']
        
        # Make predictions
        predictions = model.predict(df)
        
        return [float(p) for p in predictions]


# Initialize predictor
predictor = ModelPredictor()


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting Boston Housing API...")
    
    # Load model
    try:
        predictor.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    
    # Start Prometheus metrics server
    metrics_port = config['monitoring']['metrics_port']
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time to headers and metrics."""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        REQUEST_DURATION.observe(process_time)
        return response
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Boston Housing Price Prediction API",
        "version": config['api']['version'],
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global start_time, model, model_version
    
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        version=config['api']['version'],
        model_loaded=model is not None,
        model_version=model_version if model is not None else None,
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    global prediction_count, last_prediction_time
    
    try:
        # Make prediction
        prediction = predictor.predict_single(request.features)
        
        # Update metrics
        PREDICTION_COUNTER.inc()
        prediction_count += 1
        last_prediction_time = datetime.utcnow()
        
        # Generate response
        response = PredictionResponse(
            prediction=prediction,
            model_version=model_version,
            prediction_id=f"pred_{uuid.uuid4().hex[:12]}",
            timestamp=last_prediction_time
        )
        
        logger.info(f"Prediction made: {prediction:.2f}")
        return response
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    global prediction_count, last_prediction_time
    
    try:
        # Make predictions
        predictions = predictor.predict_batch(request.features)
        
        # Update metrics
        BATCH_COUNTER.inc()
        prediction_count += len(predictions)
        last_prediction_time = datetime.utcnow()
        
        # Generate response
        response = BatchPredictionResponse(
            predictions=predictions,
            model_version=model_version,
            batch_id=f"batch_{uuid.uuid4().hex[:12]}",
            timestamp=last_prediction_time,
            count=len(predictions)
        )
        
        logger.info(f"Batch prediction made: {len(predictions)} predictions")
        return response
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.get("/metrics/summary", response_model=MetricsResponse)
async def get_metrics_summary():
    """Human-readable metrics summary."""
    global prediction_count, last_prediction_time, start_time
    
    uptime = time.time() - start_time
    predictions_per_minute = (prediction_count / uptime) * 60 if uptime > 0 else 0
    
    return MetricsResponse(
        total_predictions=prediction_count,
        predictions_per_minute=predictions_per_minute,
        average_response_time_ms=50.0,  # This should be calculated from actual metrics
        error_rate=0.1,  # This should be calculated from actual error count
        model_version=model_version,
        last_prediction_time=last_prediction_time
    )


@app.post("/model/reload")
async def reload_model():
    """Reload model endpoint."""
    try:
        predictor.load_model()
        return {"message": "Model reloaded successfully", "version": model_version}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.utcnow(),
            request_id=f"req_{uuid.uuid4().hex[:12]}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=type(exc).__name__,
            message="Internal server error",
            timestamp=datetime.utcnow(),
            request_id=f"req_{uuid.uuid4().hex[:12]}"
        ).dict()
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=True,
        log_level=config['monitoring']['log_level'].lower()
    )
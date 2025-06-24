# API con métricas integradas
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import json

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Métricas Prometheus
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total ML predictions made'
)

PREDICTION_DURATION = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction duration'
)

MODEL_ACCURACY = Gauge(
    'model_accuracy_score',
    'Current model accuracy score'
)

PREDICTION_DRIFT = Gauge(
    'prediction_drift_score',
    'Prediction drift score'
)

ACTIVE_USERS = Gauge(
    'active_users_count',
    'Number of active users'
)

ERROR_COUNT = Counter(
    'api_errors_total',
    'Total API errors',
    ['error_type']
)

# Modelo
class HouseFeatures(BaseModel):
    crim: float = Field(..., description="Per capita crime rate")
    zn: float = Field(..., description="Proportion of residential land zoned for lots over 25,000 sq.ft")
    indus: float = Field(..., description="Proportion of non-retail business acres")
    chas: int = Field(..., description="Charles River dummy variable")
    nox: float = Field(..., description="Nitric oxides concentration")
    rm: float = Field(..., description="Average number of rooms per dwelling")
    age: float = Field(..., description="Proportion of owner-occupied units built prior to 1940")
    dis: float = Field(..., description="Weighted distances to employment centres")
    rad: int = Field(..., description="Index of accessibility to radial highways")
    tax: float = Field(..., description="Property tax rate per $10,000")
    ptratio: float = Field(..., description="Pupil-teacher ratio")
    b: float = Field(..., description="Proportion of blacks")
    lstat: float = Field(..., description="% lower status of the population")

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted house price")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str

# Inicializar FastAPI
app = FastAPI(
    title="Housing Price Prediction API",
    description="ML API for predicting house prices using Boston Housing dataset",
    version="1.0.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
scaler = None
feature_names = None
model_version = "1.0.0"
baseline_predictions = []
current_predictions = []

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metrics = {}
        
    def load_model(self, model_path: str = "models/"):
        """Cargar modelo y scaler"""
        try:
            model_file = os.path.join(model_path, "housing_model.pkl")
            scaler_file = os.path.join(model_path, "scaler.pkl")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                
                # Cargar métricas del modelo si existen
                metrics_file = os.path.join(model_path, "model_metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        self.model_metrics = json.load(f)
                        MODEL_ACCURACY.set(self.model_metrics.get('r2_score', 0))
                
                logger.info("Modelo y scaler cargados exitosamente")
                return True
            else:
                logger.error(f"Archivos de modelo no encontrados en {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            ERROR_COUNT.labels(error_type="model_loading").inc()
            return False
    
    def predict(self, features: np.ndarray) -> tuple:
        """Realizar predicción con intervalo de confianza"""
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo no cargado")
        
        start_time = time.time()
        
        try:
            features_scaled = self.scaler.transform(features)
            
            # Predicción
            prediction = self.model.predict(features_scaled)[0]
            
            # Simulación del intervalo
            # En un caso real, usara modelos bayesianos
            std_error = self.model_metrics.get('mae', 3.0)  # Usar MAE como proxy
            confidence_interval = {
                "lower": float(prediction - 1.96 * std_error),
                "upper": float(prediction + 1.96 * std_error)
            }
            
            duration = time.time() - start_time
            PREDICTION_DURATION.observe(duration)
            PREDICTION_COUNT.inc()
            
            # Almacenar predicción para detección de drift
            current_predictions.append(prediction)
            if len(current_predictions) > 1000:
                current_predictions.pop(0)
            
            return prediction, confidence_interval
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="prediction").inc()
            raise e

# Inicializar manager del modelo
model_manager = ModelManager()

# Middleware para métricas
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Registrar métricas
    duration = time.time() - start_time
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

@app.on_event("startup")
async def startup_event():
    """Inicializar aplicación"""
    logger.info("Iniciando aplicación...")
    
    # Cargar modelo
    if not model_manager.load_model():
        logger.warning("No se pudo cargar el modelo. Algunas funciones no estarán disponibles.")
    
    # Cargar datos baseline para detección de drift
    try:
        baseline_file = "models/baseline_predictions.json"
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                global baseline_predictions
                baseline_predictions = json.load(f)
    except Exception as e:
        logger.warning(f"No se pudieron cargar predicciones baseline: {str(e)}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz"""
    return {
        "message": "Housing Price Prediction API",
        "version": model_version,
        "status": "active"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_manager.model is not None,
        version=model_version
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predecir precio de vivienda"""
    if model_manager.model is None:
        ERROR_COUNT.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_array = np.array([[
            features.crim, features.zn, features.indus, features.chas,
            features.nox, features.rm, features.age, features.dis,
            features.rad, features.tax, features.ptratio, features.b, features.lstat
        ]])
        
        prediction, confidence_interval = model_manager.predict(feature_array)
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence_interval=confidence_interval,
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction_error").inc()
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(features_list: List[HouseFeatures]):
    """Predicción por lotes"""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        for features in features_list:
            feature_array = np.array([[
                features.crim, features.zn, features.indus, features.chas,
                features.nox, features.rm, features.age, features.dis,
                features.rad, features.tax, features.ptratio, features.b, features.lstat
            ]])
            
            prediction, confidence_interval = model_manager.predict(feature_array)
            
            predictions.append({
                "prediction": float(prediction),
                "confidence_interval": confidence_interval
            })
        
        return {
            "predictions": predictions,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="batch_prediction_error").inc()
        logger.error(f"Error en predicción por lotes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Endpoint para métricas de Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
async def get_model_info():
    """Información del modelo"""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": model_version,
        "model_type": str(type(model_manager.model).__name__),
        "metrics": model_manager.model_metrics,
        "feature_count": 13,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/model/drift")
async def check_drift():
    """Verificar drift del modelo"""
    if not baseline_predictions or not current_predictions:
        return {"drift_score": 0.0, "status": "insufficient_data"}
    
    try:
        baseline_mean = np.mean(baseline_predictions)
        current_mean = np.mean(current_predictions[-100:])  # Últimas 100 predicciones
        
        baseline_std = np.std(baseline_predictions)
        current_std = np.std(current_predictions[-100:])
        
        mean_drift = abs(current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        std_drift = abs(current_std - baseline_std) / baseline_std if baseline_std > 0 else 0
        
        drift_score = (mean_drift + std_drift) / 2
        PREDICTION_DRIFT.set(drift_score)
        
        status = "normal"
        if drift_score > 0.5:
            status = "warning"
        if drift_score > 0.8:
            status = "critical"
        
        return {
            "drift_score": float(drift_score),
            "status": status,
            "baseline_mean": float(baseline_mean),
            "current_mean": float(current_mean),
            "baseline_std": float(baseline_std),
            "current_std": float(current_std),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="drift_calculation").inc()
        logger.error(f"Error calculando drift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift calculation error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Estadísticas de la API"""
    return {
        "total_predictions": PREDICTION_COUNT._value._value,
        "current_predictions_count": len(current_predictions),
        "baseline_predictions_count": len(baseline_predictions),
        "model_version": model_version,
        "uptime": time.time() - start_time if 'start_time' in globals() else 0
    }

start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
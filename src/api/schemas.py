from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class HousingFeatures(BaseModel):
    
    crim: float = Field(..., description="Per capita crime rate by town", ge=0)
    zn: float = Field(..., description="Proportion of residential land zoned for lots over 25,000 sq.ft", ge=0, le=100)
    indus: float = Field(..., description="Proportion of non-retail business acres per town", ge=0, le=100)
    chas: int = Field(..., description="Charles River dummy variable (1 if tract bounds river; 0 otherwise)", ge=0, le=1)
    nox: float = Field(..., description="Nitric oxides concentration (parts per 10 million)", ge=0, le=1)
    rm: float = Field(..., description="Average number of rooms per dwelling", ge=0, le=15)
    age: float = Field(..., description="Proportion of owner-occupied units built prior to 1940", ge=0, le=100)
    dis: float = Field(..., description="Weighted distances to employment centres", ge=0)
    rad: int = Field(..., description="Index of accessibility to radial highways", ge=1, le=24)
    tax: float = Field(..., description="Full-value property-tax rate per $10,000", ge=0)
    ptratio: float = Field(..., description="Pupil-teacher ratio by town", ge=0, le=50)
    b: float = Field(..., description="1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town", ge=0)
    lstat: float = Field(..., description="% lower status of the population", ge=0, le=100)
    
    @validator('rm')
    def validate_rooms(cls, v):
        if v < 3 or v > 10:
            raise ValueError('Number of rooms should be between 3 and 10 for realistic housing')
        return v
    
    @validator('nox')
    def validate_nox(cls, v):
        if v < 0.3 or v > 1.0:
            raise ValueError('NOX concentration should be between 0.3 and 1.0')
        return v

    class Config:
        schema_extra = {
            "example": {
                "crim": 0.00632,
                "zn": 18.0,
                "indus": 2.31,
                "chas": 0,
                "nox": 0.538,
                "rm": 6.575,
                "age": 65.2,
                "dis": 4.0900,
                "rad": 1,
                "tax": 296.0,
                "ptratio": 15.3,
                "b": 396.90,
                "lstat": 4.98
            }
        }


class PredictionRequest(BaseModel):
    
    features: HousingFeatures
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "crim": 0.00632,
                    "zn": 18.0,
                    "indus": 2.31,
                    "chas": 0,
                    "nox": 0.538,
                    "rm": 6.575,
                    "age": 65.2,
                    "dis": 4.0900,
                    "rad": 1,
                    "tax": 296.0,
                    "ptratio": 15.3,
                    "b": 396.90,
                    "lstat": 4.98
                },
                "model_version": "latest"
            }
        }


class BatchPredictionRequest(BaseModel):
    
    features: List[HousingFeatures]
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    
    @validator('features')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 predictions')
        if len(v) == 0:
            raise ValueError('At least one feature set is required')
        return v


class PredictionResponse(BaseModel):
    
    prediction: float = Field(..., description="Predicted housing price in thousands of dollars")
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    timestamp: datetime = Field(..., description="Timestamp of prediction")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="95% confidence interval")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 24.0,
                "model_version": "1.0.0",
                "prediction_id": "pred_123456789",
                "timestamp": "2024-01-01T12:00:00Z",
                "confidence_interval": {
                    "lower": 22.5,
                    "upper": 25.5
                }
            }
        }


class BatchPredictionResponse(BaseModel):
    
    predictions: List[float] = Field(..., description="List of predicted housing prices")
    model_version: str = Field(..., description="Model version used for predictions")
    batch_id: str = Field(..., description="Unique identifier for this batch")
    timestamp: datetime = Field(..., description="Timestamp of batch prediction")
    count: int = Field(..., description="Number of predictions in batch")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [24.0, 21.6, 34.7],
                "model_version": "1.0.0",
                "batch_id": "batch_123456789",
                "timestamp": "2024-01-01T12:00:00Z",
                "count": 3
            }
        }


class HealthResponse(BaseModel):
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "model_version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00Z",
                "uptime_seconds": 3600.0
            }
        }


class MetricsResponse(BaseModel):
    
    total_predictions: int = Field(..., description="Total number of predictions served")
    predictions_per_minute: float = Field(..., description="Recent predictions per minute")
    average_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    error_rate: float = Field(..., description="Error rate as percentage")
    model_version: str = Field(..., description="Current model version")
    last_prediction_time: Optional[datetime] = Field(None, description="Timestamp of last prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 1500,
                "predictions_per_minute": 25.5,
                "average_response_time_ms": 45.2,
                "error_rate": 0.1,
                "model_version": "1.0.0",
                "last_prediction_time": "2024-01-01T12:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input features",
                "details": {
                    "field": "rm",
                    "issue": "Number of rooms should be between 3 and 10"
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789"
            }
        }
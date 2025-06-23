"""
Model training module with MLflow integration.
"""

import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple
import logging
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training with MLflow tracking."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ModelTrainer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
    def create_model(self) -> RandomForestRegressor:
        """Create model with configured hyperparameters."""
        model_config = self.config['model']
        
        if model_config['type'] == 'RandomForestRegressor':
            model = RandomForestRegressor(**model_config['hyperparameters'])
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
            
        return model
    
    def evaluate_model(self, model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return metrics
    
    def cross_validate_model(self, model: RandomForestRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Perform cross-validation."""
        cv_folds = self.config['training']['cross_validation_folds']
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        cv_mae = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_absolute_error')
        
        cv_metrics = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_mae_mean': -cv_mae.mean(),
            'cv_mae_std': cv_mae.std()
        }
        
        return cv_metrics
    
    def feature_importance(self, model: RandomForestRegressor, feature_names: list) -> Dict[str, float]:
        """Get feature importances."""
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        return {}
    
    def validate_model_performance(self, metrics: Dict[str, float]) -> bool:
        """Validate if model meets performance thresholds."""
        thresholds = self.config['training']['metric_threshold']
        
        if metrics['r2_score'] < thresholds['r2_score']:
            logger.warning(f"R2 score {metrics['r2_score']} below threshold {thresholds['r2_score']}")
            return False
            
        if metrics['mae'] > thresholds['mae']:
            logger.warning(f"MAE {metrics['mae']} above threshold {thresholds['mae']}")
            return False
            
        return True
    
    def save_model(self, model: RandomForestRegressor, model_name: str = None) -> str:
        """Save model to local storage."""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"boston_housing_model_{timestamp}"
        
        model_path = os.path.join(self.config['storage']['model_path'], f"{model_name}.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def train_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, y_test: pd.Series) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
        """Complete model training pipeline with MLflow tracking."""
        
        with mlflow.start_run():
            logger.info("Starting model training")
            
            # Create model
            model = self.create_model()
            
            # Log parameters
            mlflow.log_params(self.config['model']['hyperparameters'])
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_metrics = self.cross_validate_model(model, X_train, y_train)
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            # Combine all metrics
            all_metrics = {**cv_metrics, **test_metrics}
            
            # Log metrics to MLflow
            mlflow.log_metrics(all_metrics)
            
            # Feature importance
            feature_importance = self.feature_importance(model, X_train.columns.tolist())
            
            # Log feature importance as artifacts
            importance_df = pd.DataFrame(list(feature_importance.items()), 
                                       columns=['feature', 'importance'])
            importance_path = "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            # Validate model performance
            is_valid = self.validate_model_performance(test_metrics)
            mlflow.log_param("model_valid", is_valid)
            
            if is_valid:
                logger.info("Model meets performance thresholds")
                
                # Save model locally
                model_path = self.save_model(model)
                
                # Log model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.config['mlflow']['model_name']
                )
                
                # Log model file as artifact
                mlflow.log_artifact(model_path)
                
            else:
                logger.warning("Model does not meet performance thresholds")
            
            # Prepare results
            results = {
                'model': model,
                'metrics': all_metrics,
                'feature_importance': feature_importance,
                'is_valid': is_valid,
                'run_id': mlflow.active_run().info.run_id
            }
            
            logger.info("Model training completed")
            logger.info(f"Test R2 Score: {test_metrics['r2_score']:.4f}")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            
            return model, results
    
    def load_model(self, model_path: str) -> RandomForestRegressor:
        """Load model from file."""
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def load_latest_model(self) -> RandomForestRegressor:
        """Load the latest model from MLflow."""
        client = mlflow.tracking.MlflowClient()
        model_name = self.config['mlflow']['model_name']
        
        try:
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model version {latest_version.version}")
            return model
        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            raise
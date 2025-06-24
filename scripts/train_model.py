import sys
import os
import logging
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataLoader
from models.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Boston Housing Price Prediction Model')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', default='models/', help='Output directory for model')
    parser.add_argument('--experiment-name', help='MLflow experiment name override')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting Boston Housing model training pipeline")
        
        # Initialize
        data_loader = DataLoader(args.config)
        trainer = ModelTrainer(args.config)
        
        # Override
        if args.experiment_name:
            import mlflow
            mlflow.set_experiment(args.experiment_name)
        
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test, data_stats = data_loader.prepare_data_pipeline()
        
        logger.info(f"Dataset shape: {data_stats['shape']}")
        logger.info(f"Target mean: {data_stats['target_stats']['mean']:.2f}")
        logger.info(f"Target std: {data_stats['target_stats']['std']:.2f}")
        
        # Train model
        logger.info("Training model...")
        model, results = trainer.train_model(X_train, X_test, y_train, y_test)
        
        # Print results
        metrics = results['metrics']
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"Cross-validation R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        print(f"Cross-validation MAE (mean ± std): {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")
        print(f"Test R² Score: {metrics['r2_score']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
        print(f"Test MAE: {metrics['mae']:.4f}")
        print(f"Model Valid: {results['is_valid']}")
        print(f"MLflow Run ID: {results['run_id']}")
        
        # Feature
        print("\nTOP 10 IMPORTANT FEATURES:")
        print("-" * 30)
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
            print(f"{i+1:2d}. {feature:15s}: {importance:.4f}")
        
        if results['is_valid']:
            print(f"\n✅ Model training successful!")
            print(f"Model meets performance thresholds and is ready for deployment.")
        else:
            print(f"\n❌ Model training completed but doesn't meet performance thresholds.")
            print(f"Review hyperparameters and data preprocessing.")
            
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
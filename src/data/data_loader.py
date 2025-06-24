"""
Data loading and preprocessing for Boston Housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import requests
import logging
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        
    def download_data(self) -> pd.DataFrame:
        try:
            url = self.config['data']['https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv']
            logger.info(f"Downloading data from {url}")
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Save locally
            with open(self.config['data']['local_path'], 'wb') as f:
                f.write(response.content)
                
            df = pd.read_csv(self.config['data']['local_path'])
            logger.info(f"Downloaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Load data from local file or download if not exists."""
        try:
            df = pd.read_csv(self.config['data']['local_path'])
            logger.info(f"Loaded local data with shape: {df.shape}")
        except FileNotFoundError:
            logger.info("Local file not found, downloading...")
            df = self.download_data()
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:

        df_processed = df.copy()
        
        if df_processed.isnull().sum().any():
            logger.warning("Missing values found, filling with median")
            df_processed = df_processed.fillna(df_processed.median())
        
        df_processed['rm_squared'] = df_processed['rm'] ** 2
        df_processed['age_per_room'] = df_processed['age'] / df_processed['rm']
        
        # Remove outliers (simple approach using IQR)
        target_col = self.config['data']['target_column']
        Q1 = df_processed[target_col].quantile(0.25)
        Q3 = df_processed[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        original_size = len(df_processed)
        df_processed = df_processed[
            (df_processed[target_col] >= lower_bound) & 
            (df_processed[target_col] <= upper_bound)
        ]
        
        logger.info(f"Removed {original_size - len(df_processed)} outliers")
        
        return df_processed
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        target_col = self.config['data']['target_column']
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        numerical_features = self.config['features']['numerical_features']
        
        # Add engineered features to numerical list
        numerical_features.extend(['rm_squared', 'age_per_room'])
        
        # Filter only existing columns
        existing_numerical = [col for col in numerical_features if col in X_train.columns]
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Fit scaler on train data and transform both sets
        X_train_scaled[existing_numerical] = self.scaler.fit_transform(X_train[existing_numerical])
        X_test_scaled[existing_numerical] = self.scaler.transform(X_test[existing_numerical])
        
        logger.info(f"Scaled {len(existing_numerical)} numerical features")
        
        return X_train_scaled, X_test_scaled
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Get basic statistics about the dataset."""
        target_col = self.config['data']['target_column']
        
        stats = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'target_stats': {
                'mean': df[target_col].mean(),
                'std': df[target_col].std(),
                'min': df[target_col].min(),
                'max': df[target_col].max(),
                'median': df[target_col].median()
            },
            'feature_correlation': df.corr()[target_col].abs().sort_values(ascending=False).to_dict()
        }
        
        return stats
    
    def prepare_data_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:

        logger.info("Starting data preparation pipeline")
        
        # Load data
        df = self.load_data()
        
        # Get initial statistics
        initial_stats = self.get_data_statistics(df)
        
        # Preprocess
        df_processed = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df_processed)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        logger.info("Data preparation pipeline completed")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, initial_stats
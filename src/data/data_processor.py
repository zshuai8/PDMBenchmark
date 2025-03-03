#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processor for predictive maintenance benchmark.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to handle data loading, preprocessing, and feature engineering."""
    
    def __init__(self, config):
        self.config = config
        self.raw_data_path = config['data']['raw_path']
        self.processed_data_path = config['data']['processed_path']
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        
    def load_data(self, filename):
        """
        Load data from CSV or other formats.
        
        Args:
            filename (str): Name of the file to load
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {filename}")
        file_path = os.path.join(self.raw_data_path, filename)
        
        if filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif filename.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            
        logger.info(f"Loaded data with shape {data.shape}")
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess data - handle missing values, outliers, etc.
        
        Args:
            data (pandas.DataFrame): Raw data to preprocess
            
        Returns:
            pandas.DataFrame: Preprocessed data
        """
        logger.info("Preprocessing data")
        
        # Create a copy to avoid modifying the original
        preprocessed_data = data.copy()
        
        # Handle missing values
        for col in preprocessed_data.columns:
            if preprocessed_data[col].isnull().sum() > 0:
                if preprocessed_data[col].dtype == 'object':
                    preprocessed_data[col] = preprocessed_data[col].fillna('unknown')
                else:
                    preprocessed_data[col] = preprocessed_data[col].fillna(preprocessed_data[col].median())
        
        # Handle categorical features
        for col in preprocessed_data.select_dtypes(include=['object']).columns:
            preprocessed_data = pd.get_dummies(preprocessed_data, columns=[col], drop_first=True)
            
        # Handle outliers (using IQR method for numerical columns)
        numeric_cols = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            Q1 = preprocessed_data[col].quantile(0.25)
            Q3 = preprocessed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the outliers instead of removing them
            preprocessed_data[col] = np.where(
                preprocessed_data[col] < lower_bound,
                lower_bound,
                np.where(
                    preprocessed_data[col] > upper_bound,
                    upper_bound,
                    preprocessed_data[col]
                )
            )
        
        return preprocessed_data
    
    def feature_engineering(self, data):
        """
        Create new features to improve model performance.
        
        Args:
            data (pandas.DataFrame): Preprocessed data
            
        Returns:
            pandas.DataFrame: Data with engineered features
        """
        logger.info("Performing feature engineering")
        
        # Create a copy to avoid modifying the original
        engineered_data = data.copy()
        
        # Example: Create time-based features if timestamp exists
        if 'timestamp' in engineered_data.columns:
            engineered_data['timestamp'] = pd.to_datetime(engineered_data['timestamp'])
            engineered_data['hour'] = engineered_data['timestamp'].dt.hour
            engineered_data['day_of_week'] = engineered_data['timestamp'].dt.dayofweek
            engineered_data['month'] = engineered_data['timestamp'].dt.month
            engineered_data['day'] = engineered_data['timestamp'].dt.day
            engineered_data['year'] = engineered_data['timestamp'].dt.year
            engineered_data['is_weekend'] = engineered_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Example: Create rolling window features for sensor data
        window_size = self.config['feature_engineering'].get('window_size', 24)
        sensor_cols = [col for col in engineered_data.columns if col.startswith('sensor_')]
        
        for col in sensor_cols:
            if self.config['feature_engineering']['use_rolling_features']:
                engineered_data[f"{col}_rolling_mean"] = engineered_data[col].rolling(
                    window=window_size, min_periods=1).mean()
                engineered_data[f"{col}_rolling_std"] = engineered_data[col].rolling(
                    window=window_size, min_periods=1).std().fillna(0)
                engineered_data[f"{col}_rolling_min"] = engineered_data[col].rolling(
                    window=window_size, min_periods=1).min()
                engineered_data[f"{col}_rolling_max"] = engineered_data[col].rolling(
                    window=window_size, min_periods=1).max()
        
        # Create interaction features between sensors if multiple exist
        if len(sensor_cols) > 1:
            for i, col1 in enumerate(sensor_cols):
                for col2 in sensor_cols[i+1:]:
                    engineered_data[f"{col1}_{col2}_ratio"] = (
                        engineered_data[col1] / engineered_data[col2].replace(0, np.nan)
                    ).fillna(0)
                    engineered_data[f"{col1}_{col2}_diff"] = engineered_data[col1] - engineered_data[col2]
        
        # Optional: Create polynomial features if configured
        if self.config['feature_engineering'].get('create_polynomial_features', False):
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            
            # Select numerical columns for polynomial features
            num_cols = engineered_data.select_dtypes(include=['float64', 'int64']).columns
            # Limit to a reasonable number of columns to avoid explosion of features
            if len(num_cols) > 5:  # If more than 5 numerical columns
                # Select the 5 most correlated with target if target is available
                target_col = self.config.get('target_column', None)
                if target_col in engineered_data.columns:
                    correlations = engineered_data[num_cols].corrwith(engineered_data[target_col]).abs()
                    num_cols = correlations.sort_values(ascending=False).head(5).index.tolist()
                else:
                    # Otherwise just take first 5
                    num_cols = num_cols[:5]
            
            # Apply polynomial features
            poly_features = poly.fit_transform(engineered_data[num_cols])
            poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=engineered_data.index)
            
            # Combine with original data
            engineered_data = pd.concat([engineered_data, poly_df], axis=1)
        
        return engineered_data
    
    def split_data(self, data, target_col):
        """
        Split data into features and target, then into train and test sets.
        
        Args:
            data (pandas.DataFrame): Data with engineered features
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) split datasets
        """
        logger.info("Splitting data into train and test sets")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state']
        )
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale numerical features.
        
        Args:
            X_train (pandas.DataFrame): Training features
            X_test (pandas.DataFrame): Testing features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled) scaled datasets
        """
        logger.info("Scaling features")
        
        scaler = StandardScaler()
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        # Save the scaler for future use
        os.makedirs(os.path.dirname(self.config['model']['saved_models_path']), exist_ok=True)
        scaler_path = os.path.join(self.config['model']['saved_models_path'], 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled
    
    def process_and_save(self, filename, target_col):
        """
        Complete data processing pipeline.
        
        Args:
            filename (str): Name of the file to process
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test) processed datasets
        """
        # Load data
        data = self.load_data(filename)
        
        # Preprocess
        data = self.preprocess_data(data)
        
        # Feature engineering
        data = self.feature_engineering(data)
        
        # Save processed data
        processed_file_path = os.path.join(self.processed_data_path, f"processed_{filename}")
        data.to_csv(processed_file_path, index=False)
        logger.info(f"Saved processed data to {processed_file_path}")
        
        # Split and scale data
        X_train, X_test, y_train, y_test = self.split_data(data, target_col)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save train/test splits
        train_data = pd.concat([X_train_scaled, pd.DataFrame(y_train, columns=[target_col])], axis=1)
        test_data = pd.concat([X_test_scaled, pd.DataFrame(y_test, columns=[target_col])], axis=1)
        
        train_data.to_csv(os.path.join(self.processed_data_path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(self.processed_data_path, 'test_data.csv'), index=False)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create processor
    processor = DataProcessor(config)
    
    # Example processing
    print("Data processor module loaded successfully.")
    print("Use processor.process_and_save(filename, target_col) to process data.")
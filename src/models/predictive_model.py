#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive model implementation for maintenance benchmark.
"""

import os
import joblib
import numpy as np
import pandas as pd
import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

logger = logging.getLogger(__name__)

class PredictiveModel:
    """Class to train, evaluate, and save machine learning models."""
    
    def __init__(self, config):
        """
        Initialize the predictive model.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.models_path = config['model']['saved_models_path']
        self.model_type = config['model']['type']
        
        # Create directory for saving models if it doesn't exist
        os.makedirs(self.models_path, exist_ok=True)
        
    def get_model(self):
        """
        Initialize model based on configuration.
        
        Returns:
            object: Initialized model instance
        """
        if self.model_type == 'random_forest':
            # Get parameters from config
            params = self.config['model'].get('params', {})
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            random_state = self.config['model'].get('random_state', 42)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            
        elif self.model_type == 'gradient_boosting':
            # Get parameters from config
            params = self.config['model'].get('params', {})
            n_estimators = params.get('n_estimators', 100)
            learning_rate = params.get('learning_rate', 0.1)
            max_depth = params.get('max_depth', 3)
            random_state = self.config['model'].get('random_state', 42)
            
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return model
    
    def train_model(self, X_train, y_train):
        """
        Train the model on training data.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            
        Returns:
            object: Trained model
        """
        logger.info(f"Training {self.model_type} model")
        
        model = self.get_model()
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            model (object): Trained model
            X_test (pandas.DataFrame): Testing features
            y_test (pandas.Series): Testing target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # For ROC and PR curves, we need probability predictions
        # Check if model supports probability predictions
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
            # Handle binary and multiclass cases
            if y_prob.shape[1] == 2:  # Binary classification
                y_prob = y_prob[:, 1]  # Probability of positive class
            else:  # Multiclass - use one-vs-rest approach
                # This is a placeholder, would need to be adjusted for specific needs
                y_prob = y_prob  
        else:
            y_prob = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Check if binary or multiclass for proper scoring
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            average = 'binary'
        else:
            average = 'weighted'
            
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Log basic metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Store evaluation results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
        
        # Add ROC and PR curves for binary classification
        if len(unique_classes) == 2 and y_prob is not None:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            results['roc'] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
            
            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall_curve, precision_curve)
            results['pr_curve'] = {
                'precision': precision_curve,
                'recall': recall_curve,
                'auc': pr_auc
            }
            
            logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"PR AUC: {pr_auc:.4f}")
        
        return results
    
    def save_model(self, model, model_name=None):
        """
        Save trained model to disk.
        
        Args:
            model (object): Trained model to save
            model_name (str, optional): Name for the saved model. Defaults to None.
            
        Returns:
            str: Path to the saved model
        """
        if model_name is None:
            model_name = f"{self.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
        model_path = os.path.join(self.models_path, model_name)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_name):
        """
        Load saved model from disk.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            object: Loaded model
        """
        model_path = os.path.join(self.models_path, model_name)
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            class_names (list, optional): Names of classes. Defaults to None.
            save_path (str, optional): Path to save the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
            
        # Plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            fpr (numpy.ndarray): False positive rates
            tpr (numpy.ndarray): True positive rates
            roc_auc (float): Area under ROC curve
            save_path (str, optional): Path to save the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        plt.figure(figsize=(10, 8))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model (object): Trained model
            feature_names (list): Names of features
            top_n (int, optional): Number of top features to show. Defaults to 20.
            save_path (str, optional): Path to save the plot. Defaults to None.
            
        Returns:
            dict: Dictionary mapping features to importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Limit to top_n features
            if top_n and top_n < len(indices):
                indices = indices[:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.close()
            
            # Return feature importance as dictionary
            return {feature_names[i]: importances[indices][idx] for idx, i in enumerate(indices)}
        else:
            logger.warning("Model doesn't have feature_importances_ attribute")
            return None


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
    
    # Create model handler
    model_handler = PredictiveModel(config)
    
    # Example usage
    print("Model module loaded successfully.")
    print(f"Configured model type: {model_handler.model_type}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark utilities for predictive maintenance.
"""

import os
import pandas as pd
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

logger = logging.getLogger(__name__)

class BenchmarkUtils:
    """Utilities for benchmarking different models and configurations."""
    
    def __init__(self, config):
        """
        Initialize benchmark utilities.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.results_path = config['benchmark']['results_path']
        os.makedirs(self.results_path, exist_ok=True)
        
    def benchmark_models(self, X_train, X_test, y_train, y_test, models_to_test=None):
        """
        Benchmark multiple models and compare their performance.
        
        Args:
            X_train (pandas.DataFrame): Training features
            X_test (pandas.DataFrame): Testing features
            y_train (pandas.Series): Training target
            y_test (pandas.Series): Testing target
            models_to_test (dict, optional): Dictionary of models to test. Defaults to None.
            
        Returns:
            dict: Dictionary of benchmark results
        """
        # If no models specified, use default models from config
        if models_to_test is None:
            models_to_test = self._get_default_models()
            
        results = {}
        
        for model_name, model_config in models_to_test.items():
            logger.info(f"Benchmarking model: {model_name}")
            
            try:
                # Initialize model based on type
                if model_config['type'] == 'random_forest':
                    model = RandomForestClassifier(**model_config['params'])
                elif model_config['type'] == 'gradient_boosting':
                    model = GradientBoostingClassifier(**model_config['params'])
                else:
                    logger.warning(f"Unsupported model type: {model_config['type']}")
                    continue
                    
                # Train model
                logger.info(f"Training {model_name}...")
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                # Check if binary or multiclass
                unique_classes = np.unique(y_test)
                if len(unique_classes) == 2:
                    average = 'binary'
                else:
                    average = 'weighted'
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average=average, zero_division=0)
                recall = recall_score(y_test, y_pred, average=average, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
                
                # Store results
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'model': model  # Store model for later use
                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {str(e)}")
                continue
            
        return results
    
    def _get_default_models(self):
        """
        Get default models to benchmark from configuration.
        
        Returns:
            dict: Dictionary of default models
        """
        return {
            'RandomForest-Default': {
                'type': 'random_forest',
                'params': {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
            },
            'RandomForest-Tuned': {
                'type': 'random_forest',
                'params': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42}
            },
            'GradientBoosting-Default': {
                'type': 'gradient_boosting',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
            },
            'GradientBoosting-Tuned': {
                'type': 'gradient_boosting',
                'params': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42}
            }
        }
    
    def plot_benchmark_results(self, results, metric='f1', save_path=None):
        """
        Plot benchmark results for visual comparison.
        
        Args:
            results (dict): Dictionary of benchmark results
            metric (str, optional): Metric to plot. Defaults to 'f1'.
            save_path (str, optional): Path to save the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        plt.figure(figsize=(12, 6))
        
        model_names = list(results.keys())
        metric_values = [results[model][metric] for model in model_names]
        
        # Sort by metric value
        sorted_indices = np.argsort(metric_values)[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_values = [metric_values[i] for i in sorted_indices]
        
        bars = plt.bar(sorted_names, sorted_values, color='steelblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Model Comparison by {metric.capitalize()} Score')
        plt.ylim(0, max(metric_values) + 0.1)  # Add some space above highest bar
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Benchmark results plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_metrics_comparison(self, results, save_path=None):
        """
        Plot comparison of multiple metrics across models.
        
        Args:
            results (dict): Dictionary of benchmark results
            save_path (str, optional): Path to save the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(results.keys())
        
        # Sort models by F1 score
        sorted_indices = np.argsort([results[model]['f1'] for model in model_names])[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        
        # Set up the plot
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(sorted_names))  # Label locations
        width = 0.2  # Width of the bars
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in sorted_names]
            plt.bar(x + width * (i - 1.5), values, width, label=metric.capitalize())
        
        plt.ylabel('Score')
        plt.title('Comparison of Model Metrics')
        plt.xticks(x, sorted_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metrics comparison plot saved to {save_path}")
        
        return plt.gcf()
    
    def save_benchmark_results(self, results, filename=None):
        """
        Save benchmark results to a CSV file.
        
        Args:
            results (dict): Dictionary of benchmark results
            filename (str, optional): Name of file to save. Defaults to None.
            
        Returns:
            str: Path to saved results
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_results_{timestamp}.csv"
            
        # Extract metrics, removing model objects
        results_clean = {}
        for model_name, model_metrics in results.items():
            results_clean[model_name] = {k: v for k, v in model_metrics.items() if k != 'model'}
        
        # Convert nested dict to DataFrame
        results_df = pd.DataFrame.from_dict({
            (model, metric): value
            for model, metrics in results_clean.items()
            for metric, value in metrics.items()
        }, orient='index').reset_index()
        
        results_df = results_df.rename(columns={'level_0': 'model', 'level_1': 'metric', 0: 'value'})
        results_df = results_df.pivot(index='model', columns='metric', values='value')
        
        # Add ranking columns
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            results_df[f'{metric}_rank'] = results_df[metric].rank(ascending=False).astype(int)
        
        # Add average rank column
        rank_columns = [col for col in results_df.columns if col.endswith('_rank')]
        results_df['avg_rank'] = results_df[rank_columns].mean(axis=1)
        
        # Sort by average rank
        results_df = results_df.sort_values('avg_rank')
        
        # Save to CSV
        results_path = os.path.join(self.results_path, filename)
        results_df.to_csv(results_path)
        logger.info(f"Benchmark results saved to {results_path}")
        
        return results_path
    
    def save_best_model(self, results, metric='f1'):
        """
        Save the best performing model from benchmark results.
        
        Args:
            results (dict): Dictionary of benchmark results
            metric (str, optional): Metric to use for comparison. Defaults to 'f1'.
            
        Returns:
            tuple: (best_model_name, best_model_path)
        """
        # Find best model
        best_model_name = max(results.items(), key=lambda x: x[1][metric])[0]
        best_model = results[best_model_name]['model']
        
        # Save model
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_path = os.path.join(
            self.config['model']['saved_models_path'], 
            f"best_{best_model_name}_{timestamp}.pkl"
        )
        
        import joblib
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model ({best_model_name}) saved to {best_model_path}")
        
        return best_model_name, best_model_path
    
    def create_benchmark_report(self, results, output_path=None):
        """
        Create a comprehensive benchmark report with visualizations.
        
        Args:
            results (dict): Dictionary of benchmark results
            output_path (str, optional): Path to save report. Defaults to None.
            
        Returns:
            str: Path to saved report
        """
        if output_path is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.results_path, f"benchmark_report_{timestamp}")
            os.makedirs(output_path, exist_ok=True)
        
        # Save results table
        results_path = self.save_benchmark_results(
            results, filename=os.path.basename(output_path) + "_results.csv"
        )
        
        # Plot overall comparison
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            self.plot_benchmark_results(
                results, 
                metric=metric,
                save_path=os.path.join(output_path, f"{metric}_comparison.png")
            )
        
        # Plot metrics comparison
        self.plot_metrics_comparison(
            results,
            save_path=os.path.join(output_path, "metrics_comparison.png")
        )
        
        # Save best model
        best_model_name, best_model_path = self.save_best_model(results)
        
        # Create summary markdown file
        summary_path = os.path.join(output_path, "summary.md")
        with open(summary_path, 'w') as f:
            f.write("# Predictive Maintenance Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("### Best Performing Model\n\n")
            f.write(f"**Model:** {best_model_name}\n\n")
            f.write("**Metrics:**\n\n")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                f.write(f"- {metric.capitalize()}: {results[best_model_name][metric]:.4f}\n")
            
            f.write("\n### All Models Performance\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|----------|-----------|--------|----------|\n")
            
            # Sort models by F1 score
            sorted_models = sorted(
                results.items(), key=lambda x: x[1]['f1'], reverse=True
            )
            
            for model_name, metrics in sorted_models:
                f.write(
                    f"| {model_name} | {metrics['accuracy']:.4f} | "
                    f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['f1']:.4f} |\n"
                )
            
            f.write("\n## Visualizations\n\n")
            f.write("### Metrics Comparison\n\n")
            f.write("![Metrics Comparison](metrics_comparison.png)\n\n")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                f.write(f"### {metric.capitalize()} Comparison\n\n")
                f.write(f"![{metric.capitalize()} Comparison]({metric}_comparison.png)\n\n")
        
        logger.info(f"Benchmark report created at {output_path}")
        return output_path


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create benchmark utility
    benchmark_utils = BenchmarkUtils(config)
    
    # Example usage
    print("Benchmark utilities module loaded successfully.")
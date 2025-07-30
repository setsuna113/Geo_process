#!/usr/bin/env python
"""Utility script for comparing multiple ML models."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.machine_learning import (
    LinearRegressionAnalyzer,
    LightGBMAnalyzer,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV
)
from src.config import config
from src.core.registry import ComponentRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_models():
    """Get all available ML models from registry."""
    models = {}
    
    # Always include base models
    models['linear_regression'] = LinearRegressionAnalyzer
    
    # Check for optional models
    try:
        models['lightgbm'] = LightGBMAnalyzer
    except ImportError:
        logger.warning("LightGBM not available")
    
    # Get any custom registered models
    registry_models = ComponentRegistry.get_available_ml_models()
    for model_type in registry_models:
        if model_type not in models:
            # This would need the actual class retrieval from registry
            logger.info(f"Found registered model: {model_type}")
    
    return models


def compare_models(X, y, models, cv_strategy, metrics=['r2', 'rmse', 'mae']):
    """Compare multiple models using cross-validation."""
    results = {}
    
    for model_name, model_class in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        try:
            # Create model instance
            model = model_class(config)
            
            # Perform cross-validation
            cv_results = model.cross_validate(
                X, y,
                cv_strategy=cv_strategy,
                metrics=metrics
            )
            
            # Store results
            results[model_name] = {
                'cv_results': cv_results,
                'mean_metrics': cv_results.mean_metrics,
                'std_metrics': cv_results.std_metrics,
                'fold_metrics': cv_results.fold_metrics,
                'model': model
            }
            
            # Log results
            logger.info(f"{model_name} results:")
            for metric in metrics:
                mean_val = cv_results.mean_metrics[metric]
                std_val = cv_results.std_metrics[metric]
                logger.info(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")
                
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def plot_model_comparison(results, output_dir):
    """Create visualization of model comparison."""
    try:
        # Prepare data for plotting
        plot_data = []
        for model_name, result in results.items():
            if 'error' not in result:
                for metric, value in result['mean_metrics'].items():
                    plot_data.append({
                        'Model': model_name,
                        'Metric': metric.upper(),
                        'Value': value,
                        'Std': result['std_metrics'][metric]
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots for each metric
        metrics = df['Metric'].unique()
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            metric_df = df[df['Metric'] == metric]
            
            # Bar plot with error bars
            ax = axes[idx]
            bars = ax.bar(metric_df['Model'], metric_df['Value'])
            ax.errorbar(metric_df['Model'], metric_df['Value'], 
                       yerr=metric_df['Std'], fmt='none', color='black', capsize=5)
            
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Color best model
            if metric == 'R2':
                best_idx = metric_df['Value'].idxmax()
            else:  # Lower is better for RMSE, MAE
                best_idx = metric_df['Value'].idxmin()
            
            bars[best_idx].set_color('green')
        
        plt.tight_layout()
        plot_path = output_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to {plot_path}")
        
        # Create detailed fold performance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, result in results.items():
            if 'error' not in result and 'fold_metrics' in result:
                folds = range(1, len(result['fold_metrics']) + 1)
                r2_values = [fold['r2'] for fold in result['fold_metrics']]
                ax.plot(folds, r2_values, marker='o', label=model_name)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance Across CV Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fold_plot_path = output_dir / 'fold_performance.png'
        plt.savefig(fold_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Fold performance plot saved to {fold_plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")


def create_comparison_report(results, output_dir):
    """Create detailed comparison report."""
    report = []
    report.append("# Model Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Summary")
    
    # Find best model for each metric
    best_models = {}
    for metric in ['r2', 'rmse', 'mae']:
        best_model = None
        best_value = None
        
        for model_name, result in results.items():
            if 'error' not in result and metric in result['mean_metrics']:
                value = result['mean_metrics'][metric]
                
                if best_value is None:
                    best_value = value
                    best_model = model_name
                elif metric == 'r2' and value > best_value:
                    best_value = value
                    best_model = model_name
                elif metric in ['rmse', 'mae'] and value < best_value:
                    best_value = value
                    best_model = model_name
        
        if best_model:
            best_models[metric] = (best_model, best_value)
    
    report.append("\n### Best Models by Metric:")
    for metric, (model, value) in best_models.items():
        report.append(f"- **{metric.upper()}**: {model} ({value:.3f})")
    
    # Detailed results
    report.append("\n## Detailed Results")
    
    for model_name, result in results.items():
        report.append(f"\n### {model_name}")
        
        if 'error' in result:
            report.append(f"Error: {result['error']}")
        else:
            report.append("\n#### Cross-Validation Metrics:")
            for metric, value in result['mean_metrics'].items():
                std = result['std_metrics'][metric]
                report.append(f"- {metric.upper()}: {value:.3f} ± {std:.3f}")
            
            if 'fold_metrics' in result:
                report.append("\n#### Per-Fold Performance:")
                for i, fold in enumerate(result['fold_metrics']):
                    metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in fold.items()])
                    report.append(f"- Fold {i+1}: {metrics_str}")
    
    # Recommendations
    report.append("\n## Recommendations")
    
    if 'r2' in best_models:
        best_r2_model = best_models['r2'][0]
        report.append(f"\n1. **Best Overall Model**: {best_r2_model}")
        report.append("   - Highest R² score indicates best predictive performance")
    
    # Check for overfitting
    report.append("\n2. **Overfitting Analysis**:")
    for model_name, result in results.items():
        if 'error' not in result and 'std_metrics' in result:
            r2_std = result['std_metrics'].get('r2', 0)
            if r2_std > 0.1:
                report.append(f"   - {model_name}: High variance (std={r2_std:.3f}), may be overfitting")
    
    # Save report
    report_path = output_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Comparison report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple ML models')
    parser.add_argument('data_path', type=Path, help='Path to training data')
    parser.add_argument('--target', default='total_richness', help='Target column name')
    parser.add_argument('--models', nargs='+', help='Specific models to compare')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--block-size', type=int, default=100, 
                        help='Block size in km for spatial CV')
    parser.add_argument('--output-dir', type=Path, help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Skip creating plots')
    parser.add_argument('--train-best', action='store_true', 
                        help='Train and save the best model')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path('ml_comparisons') / f"comparison_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data_path}")
    if args.data_path.suffix == '.csv':
        data = pd.read_csv(args.data_path)
    elif args.data_path.suffix == '.parquet':
        data = pd.read_parquet(args.data_path)
    else:
        raise ValueError(f"Unsupported file format: {args.data_path.suffix}")
    
    # Impute and engineer features
    logger.info("Preparing data...")
    imputer = SpatialKNNImputer(n_neighbors=10, spatial_weight=0.6)
    data_clean = imputer.fit_transform(data)
    
    feature_builder = CompositeFeatureBuilder(config)
    features = feature_builder.fit_transform(data_clean)
    
    # Prepare target and features
    target_col = args.target
    if target_col not in features.columns:
        target_col = f'richness_{target_col}'
    
    y = features[target_col]
    feature_cols = [
        col for col in features.columns
        if col != target_col and 'richness' not in col.lower()
    ]
    X = features[feature_cols]
    
    logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    # Get models to compare
    available_models = get_available_models()
    if args.models:
        models_to_compare = {
            name: cls for name, cls in available_models.items()
            if name in args.models
        }
    else:
        models_to_compare = available_models
    
    logger.info(f"Comparing models: {list(models_to_compare.keys())}")
    
    # Create CV strategy
    cv_strategy = SpatialBlockCV(
        n_splits=args.cv_folds,
        block_size=args.block_size,
        random_state=42
    )
    
    # Compare models
    results = compare_models(X, y, models_to_compare, cv_strategy)
    
    # Create visualizations
    if not args.no_plots:
        plot_model_comparison(results, output_dir)
    
    # Create report
    create_comparison_report(results, output_dir)
    
    # Save detailed results
    results_data = {}
    for model_name, result in results.items():
        if 'error' not in result:
            results_data[model_name] = {
                'mean_metrics': result['mean_metrics'],
                'std_metrics': result['std_metrics'],
                'fold_metrics': result['fold_metrics']
            }
    
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Train and save best model if requested
    if args.train_best and results:
        # Find best model by R²
        best_model_name = max(
            (k for k, v in results.items() if 'error' not in v),
            key=lambda k: results[k]['mean_metrics']['r2']
        )
        
        logger.info(f"\nTraining best model: {best_model_name}")
        best_model = results[best_model_name]['model']
        best_model.fit(X, y)
        
        model_path = output_dir / f"best_model_{best_model_name}.pkl"
        best_model.save_model(model_path)
        logger.info(f"Best model saved to {model_path}")
        
        # Save preprocessors
        import pickle
        
        imputer_path = output_dir / 'imputer.pkl'
        with open(imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        
        builder_path = output_dir / 'feature_builder.pkl'
        with open(builder_path, 'wb') as f:
            pickle.dump(feature_builder, f)
        
        # Save configuration
        config_data = {
            'best_model': best_model_name,
            'model_path': str(model_path),
            'feature_columns': list(X.columns),
            'target_column': target_col,
            'cv_r2': results[best_model_name]['mean_metrics']['r2']
        }
        
        config_path = output_dir / 'best_model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    logger.info(f"\nComparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
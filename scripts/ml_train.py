#!/usr/bin/env python
"""Utility script for training ML models with various configurations."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.machine_learning import (
    LinearRegressionAnalyzer,
    LightGBMAnalyzer,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV,
    SpatialBufferCV,
    EnvironmentalBlockCV
)
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(model_type: str):
    """Create ML model based on type."""
    if model_type == 'linear_regression':
        return LinearRegressionAnalyzer(config)
    elif model_type == 'lightgbm':
        return LightGBMAnalyzer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_cv_strategy(cv_type: str, n_splits: int = 5, **kwargs):
    """Create cross-validation strategy."""
    if cv_type == 'spatial_block':
        return SpatialBlockCV(
            n_splits=n_splits,
            block_size=kwargs.get('block_size', 100),
            random_state=kwargs.get('random_state', 42)
        )
    elif cv_type == 'spatial_buffer':
        return SpatialBufferCV(
            n_splits=n_splits,
            buffer_distance=kwargs.get('buffer_distance', 50),
            random_state=kwargs.get('random_state', 42)
        )
    elif cv_type == 'environmental':
        return EnvironmentalBlockCV(
            n_splits=n_splits,
            stratify_by=kwargs.get('stratify_by', 'latitude')
        )
    else:
        raise ValueError(f"Unknown CV strategy: {cv_type}")


def prepare_data(data_path: Path, target_column: str, 
                 impute_missing: bool = True, engineer_features: bool = True):
    """Load and prepare data for training."""
    # Load data
    logger.info(f"Loading data from {data_path}")
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    logger.info(f"Loaded {len(data)} records with {len(data.columns)} columns")
    
    # Impute missing values
    imputer = None
    if impute_missing and data.isna().any().any():
        logger.info("Imputing missing values...")
        imputer = SpatialKNNImputer(n_neighbors=10, spatial_weight=0.6)
        data = imputer.fit_transform(data)
    
    # Engineer features
    feature_builder = None
    if engineer_features:
        logger.info("Engineering features...")
        feature_builder = CompositeFeatureBuilder(config)
        data = feature_builder.fit_transform(data)
        
        # Log feature summary
        summary = feature_builder.get_feature_summary()
        for category, info in summary['categories'].items():
            logger.info(f"  {category}: {info['count']} features")
    
    # Prepare target and features
    if target_column not in data.columns:
        # Try with richness prefix
        prefixed_target = f'richness_{target_column}'
        if prefixed_target in data.columns:
            target_column = prefixed_target
        else:
            raise ValueError(f"Target column '{target_column}' not found")
    
    y = data[target_column].copy()
    
    # Remove target-related columns from features
    feature_cols = [
        col for col in data.columns
        if col != target_column and 'richness' not in col.lower()
    ]
    X = data[feature_cols]
    
    logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Target: {target_column} (mean: {y.mean():.2f}, std: {y.std():.2f})")
    
    return X, y, data, imputer, feature_builder


def train_and_evaluate(X, y, model, cv_strategy=None, save_cv_results=False):
    """Train model and optionally perform cross-validation."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model.__class__.__name__,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    # Cross-validation
    if cv_strategy is not None:
        logger.info(f"Performing {cv_strategy.__class__.__name__} cross-validation...")
        cv_results = model.cross_validate(
            X, y,
            cv_strategy=cv_strategy,
            metrics=['r2', 'rmse', 'mae']
        )
        
        results['cv_results'] = {
            'strategy': cv_strategy.__class__.__name__,
            'n_folds': len(cv_results.fold_metrics),
            'mean_metrics': cv_results.mean_metrics,
            'std_metrics': cv_results.std_metrics
        }
        
        logger.info("Cross-validation results:")
        for metric, value in cv_results.mean_metrics.items():
            std = cv_results.std_metrics[metric]
            logger.info(f"  {metric}: {value:.3f} ± {std:.3f}")
        
        if save_cv_results:
            results['fold_metrics'] = cv_results.fold_metrics
    
    # Train final model
    logger.info("Training final model on all data...")
    model.fit(X, y)
    
    # Training metrics
    train_metrics = model.evaluate(X, y, metrics=['r2', 'rmse', 'mae'])
    results['training_metrics'] = train_metrics
    
    logger.info("Training metrics:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        results['feature_importance'] = dict(sorted_importance[:20])  # Top 20
        
        logger.info("\nTop 10 important features:")
        for feat, score in sorted_importance[:10]:
            logger.info(f"  {feat}: {score:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train ML models for biodiversity prediction')
    parser.add_argument('data_path', type=Path, help='Path to training data')
    parser.add_argument('--target', default='total_richness', help='Target column name')
    parser.add_argument('--model-type', choices=['linear_regression', 'lightgbm'],
                        default='linear_regression', help='Type of model to train')
    parser.add_argument('--cv-strategy', choices=['spatial_block', 'spatial_buffer', 'environmental'],
                        help='Cross-validation strategy')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--no-imputation', action='store_true', help='Skip missing value imputation')
    parser.add_argument('--no-features', action='store_true', help='Skip feature engineering')
    parser.add_argument('--output-dir', type=Path, help='Output directory for model and results')
    parser.add_argument('--experiment-name', help='Name for this experiment')
    
    # CV-specific parameters
    parser.add_argument('--block-size', type=int, default=100, 
                        help='Block size in km for spatial_block CV')
    parser.add_argument('--buffer-distance', type=int, default=50,
                        help='Buffer distance in km for spatial_buffer CV')
    parser.add_argument('--stratify-by', default='latitude',
                        help='Variable to stratify by for environmental CV')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path('ml_outputs') / (args.experiment_name or f"experiment_{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Prepare data
    X, y, full_data, imputer, feature_builder = prepare_data(
        args.data_path,
        args.target,
        impute_missing=not args.no_imputation,
        engineer_features=not args.no_features
    )
    
    # Create model
    model = create_model(args.model_type)
    
    # Create CV strategy
    cv_strategy = None
    if args.cv_strategy:
        cv_kwargs = {
            'block_size': args.block_size,
            'buffer_distance': args.buffer_distance,
            'stratify_by': args.stratify_by,
            'random_state': 42
        }
        cv_strategy = create_cv_strategy(args.cv_strategy, args.cv_folds, **cv_kwargs)
    
    # Train and evaluate
    results = train_and_evaluate(X, y, model, cv_strategy, save_cv_results=True)
    
    # Save model
    model_path = output_dir / f"{args.model_type}_model.pkl"
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessors
    if imputer:
        import pickle
        imputer_path = output_dir / 'imputer.pkl'
        with open(imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        logger.info(f"Imputer saved to {imputer_path}")
    
    if feature_builder:
        import pickle
        builder_path = output_dir / 'feature_builder.pkl'
        with open(builder_path, 'wb') as f:
            pickle.dump(feature_builder, f)
        logger.info(f"Feature builder saved to {builder_path}")
    
    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save feature importance as CSV
    if 'feature_importance' in results:
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in results['feature_importance'].items()
        ])
        importance_path = output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Save training configuration
    config_data = {
        'data_path': str(args.data_path),
        'target_column': args.target,
        'model_type': args.model_type,
        'cv_strategy': args.cv_strategy,
        'cv_folds': args.cv_folds,
        'imputation': not args.no_imputation,
        'feature_engineering': not args.no_features,
        'experiment_name': args.experiment_name,
        'feature_columns': list(X.columns)
    }
    
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info("\nTraining complete!")
    logger.info(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Utility script for making predictions with a trained ML model."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.machine_learning import (
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    LinearRegressionAnalyzer,
    LightGBMAnalyzer
)
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: Path, model_type: str):
    """Load a trained ML model."""
    if model_type == 'linear_regression':
        model = LinearRegressionAnalyzer(config)
    elif model_type == 'lightgbm':
        model = LightGBMAnalyzer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_model(model_path)
    return model


def prepare_data(data: pd.DataFrame, imputer_path: Path = None, 
                 feature_builder_path: Path = None):
    """Prepare data for prediction using saved preprocessors."""
    # Handle missing values
    if imputer_path and imputer_path.exists():
        logger.info("Loading saved imputer")
        import pickle
        with open(imputer_path, 'rb') as f:
            imputer = pickle.load(f)
    else:
        logger.info("Creating new imputer")
        imputer = SpatialKNNImputer(n_neighbors=10, spatial_weight=0.6)
        imputer.fit(data)
    
    data_clean = imputer.transform(data)
    
    # Engineer features
    if feature_builder_path and feature_builder_path.exists():
        logger.info("Loading saved feature builder")
        import pickle
        with open(feature_builder_path, 'rb') as f:
            feature_builder = pickle.load(f)
    else:
        logger.info("Creating new feature builder")
        feature_builder = CompositeFeatureBuilder(config)
        feature_builder.fit(data_clean)
    
    features = feature_builder.transform(data_clean)
    
    return features, imputer, feature_builder


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained ML model')
    parser.add_argument('model_path', type=Path, help='Path to saved model')
    parser.add_argument('data_path', type=Path, help='Path to input data (CSV or Parquet)')
    parser.add_argument('--model-type', choices=['linear_regression', 'lightgbm'],
                        default='linear_regression', help='Type of model')
    parser.add_argument('--output', type=Path, help='Output path for predictions')
    parser.add_argument('--imputer-path', type=Path, help='Path to saved imputer')
    parser.add_argument('--feature-builder-path', type=Path, 
                        help='Path to saved feature builder')
    parser.add_argument('--feature-columns', nargs='+', 
                        help='Specific feature columns to use')
    parser.add_argument('--include-uncertainty', action='store_true',
                        help='Include prediction uncertainty if available')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    if args.data_path.suffix == '.csv':
        data = pd.read_csv(args.data_path)
    elif args.data_path.suffix == '.parquet':
        data = pd.read_parquet(args.data_path)
    else:
        raise ValueError(f"Unsupported file format: {args.data_path.suffix}")
    
    logger.info(f"Loaded {len(data)} records")
    
    # Prepare data
    features, imputer, feature_builder = prepare_data(
        data, args.imputer_path, args.feature_builder_path
    )
    
    # Select features
    if args.feature_columns:
        # Use specified columns
        feature_cols = args.feature_columns
        missing_cols = set(feature_cols) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
    else:
        # Auto-detect feature columns (exclude target-like columns)
        exclude_patterns = ['richness', 'target', 'label', 'y']
        feature_cols = [
            col for col in features.columns
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]
    
    X = features[feature_cols]
    logger.info(f"Using {len(feature_cols)} features for prediction")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_type)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X)
    
    # Prepare output
    output_df = data.copy()
    output_df['predicted_richness'] = predictions
    
    # Add uncertainty if requested and available
    if args.include_uncertainty:
        try:
            # This would need to be implemented in the model classes
            uncertainty = model.predict_uncertainty(X)
            output_df['prediction_lower'] = predictions - uncertainty
            output_df['prediction_upper'] = predictions + uncertainty
        except AttributeError:
            logger.warning("Model does not support uncertainty estimation")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = args.data_path.parent / f"{args.data_path.stem}_predictions.csv"
    
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Summary statistics
    logger.info("\nPrediction Summary:")
    logger.info(f"  Mean: {predictions.mean():.2f}")
    logger.info(f"  Std: {predictions.std():.2f}")
    logger.info(f"  Min: {predictions.min():.2f}")
    logger.info(f"  Max: {predictions.max():.2f}")
    
    # Save preprocessors if not already saved
    if not args.imputer_path:
        imputer_path = args.model_path.parent / 'imputer.pkl'
        import pickle
        with open(imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        logger.info(f"Imputer saved to {imputer_path}")
    
    if not args.feature_builder_path:
        builder_path = args.model_path.parent / 'feature_builder.pkl'
        import pickle
        with open(builder_path, 'wb') as f:
            pickle.dump(feature_builder, f)
        logger.info(f"Feature builder saved to {builder_path}")


if __name__ == "__main__":
    main()
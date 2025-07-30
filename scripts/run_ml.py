#!/usr/bin/env python3
"""
Standalone ML Pipeline Runner - Runs machine learning experiments on biodiversity data.

This is a completely standalone runner that doesn't depend on the main pipeline
infrastructure. It reads parquet files and runs ML experiments.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import traceback
import yaml

# Set environment variable to skip database initialization
os.environ['SKIP_DB_INIT'] = 'true'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class StandaloneMLRunner:
    """Standalone ML runner that doesn't depend on pipeline infrastructure."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize ML runner with configuration."""
        self.config_path = config_path or project_root / 'config.yml'
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_dir = Path(self.config.get('paths', {}).get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'ml_standalone_{datetime.now():%Y%m%d_%H%M%S}.log'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment configuration from config.yml."""
        ml_config = self.config.get('machine_learning', {})
        experiments = ml_config.get('experiments', {})
        
        if experiment_name not in experiments:
            available = list(experiments.keys())
            raise ValueError(
                f"Experiment '{experiment_name}' not found. "
                f"Available experiments: {available}"
            )
        
        # Get experiment specific config
        exp_config = experiments[experiment_name].copy()
        
        # Merge with defaults
        defaults = ml_config.get('defaults', {})
        if defaults:
            # Merge defaults (exp_config takes precedence)
            merged_config = defaults.copy()
            merged_config.update(exp_config)
            return merged_config
        
        return exp_config
    
    def run_ml_experiment(self, ml_config: Dict[str, Any]) -> bool:
        """Run ML experiment with given configuration."""
        try:
            # Import ML components (lazy import to avoid early database connections)
            from src.machine_learning import (
                LinearRegressionAnalyzer,
                CompositeFeatureBuilder,
                SpatialKNNImputer,
                SpatialBlockCV,
                SpatialBufferCV,
                EnvironmentalBlockCV
            )
            # Try to import LightGBM if available
            try:
                from src.machine_learning import LightGBMAnalyzer
            except ImportError:
                LightGBMAnalyzer = None
            import pandas as pd
            
            # Validate configuration
            input_parquet = ml_config.get('input_parquet')
            if not input_parquet:
                self.logger.error("No input parquet file specified")
                return False
            
            input_path = Path(input_parquet)
            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return False
            
            # Create output directory (use local directory, not cluster paths)
            output_base_config = self.config.get('output_paths', {}).get('results_dir', 'outputs')
            # Use local outputs directory if cluster path is not accessible
            if output_base_config.startswith('/scratch') and not Path('/scratch').exists():
                output_base = Path('outputs')
            else:
                output_base = Path(output_base_config)
            
            exp_name = ml_config.get('experiment_name', f"ml_{datetime.now():%Y%m%d_%H%M%S}")
            output_dir = output_base / 'ml_results' / exp_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Starting ML experiment: {exp_name}")
            self.logger.info(f"Input data: {input_path}")
            self.logger.info(f"Output directory: {output_dir}")
            
            # Load data directly with pandas
            self.logger.info("Loading dataset...")
            data = pd.read_parquet(input_path)
            
            self.logger.info(f"Loaded {len(data)} samples with {len(data.columns)} columns")
            
            # Handle missing values
            imputation_config = ml_config.get('imputation_strategy', {})
            if imputation_config.get('type') == 'spatial_knn':
                self.logger.info("Imputing missing values...")
                # Select only numeric columns for imputation
                numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
                
                if numeric_cols:
                    imputer = SpatialKNNImputer(
                        n_neighbors=imputation_config.get('n_neighbors', 10),
                        spatial_weight=imputation_config.get('spatial_weight', 0.6)
                    )
                    data_numeric = data[numeric_cols]
                    data_imputed = imputer.fit_transform(data_numeric)
                    # Combine back with non-numeric columns
                    for col in non_numeric_cols:
                        data_imputed[col] = data[col]
                    data = data_imputed
            
            # Feature engineering
            self.logger.info("Engineering features...")
            feature_builder = CompositeFeatureBuilder(self.config)
            engineered_features = feature_builder.fit_transform(data)
            
            # Combine engineered features with original data
            # Keep all original columns plus new features
            features = data.copy()
            for col in engineered_features.columns:
                if col not in features.columns:
                    features[col] = engineered_features[col]
            
            # Prepare target and features
            target_col = ml_config.get('target_column', 'total_richness')
            
            # Handle potential column name variations
            if target_col not in features.columns:
                # Try with richness prefix
                prefixed_target = f'richness_{target_col}'
                if prefixed_target in features.columns:
                    target_col = prefixed_target
                else:
                    self.logger.error(f"Target column '{target_col}' not found")
                    self.logger.info(f"Available columns: {list(features.columns)}")
                    return False
            
            y = features[target_col].copy()
            
            # Get feature columns
            feature_cols = ml_config.get('feature_columns')
            if not feature_cols:
                # Auto-detect numeric feature columns only
                numeric_features = features.select_dtypes(include=['float64', 'int64']).columns.tolist()
                exclude_patterns = [target_col, 'richness', 'lat', 'lon', 'grid_id', 'year']
                feature_cols = [
                    col for col in numeric_features 
                    if not any(pattern in col.lower() for pattern in exclude_patterns)
                ]
            
            X = features[feature_cols]
            self.logger.info(f"Using {len(feature_cols)} features for modeling")
            
            # Create model
            model_type = ml_config.get('model_type', 'linear_regression')
            if model_type == 'linear_regression':
                model = LinearRegressionAnalyzer(self.config)
            elif model_type == 'lightgbm':
                if LightGBMAnalyzer is None:
                    self.logger.error("LightGBM requested but not installed. Install with: pip install lightgbm")
                    return False
                model = LightGBMAnalyzer(self.config)
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return False
            
            # Cross-validation (if requested)
            if ml_config.get('perform_cv', True):
                cv_config = ml_config.get('cv_strategy', {})
                cv_type = cv_config.get('type', 'spatial_block')
                
                if cv_type == 'spatial_block':
                    cv = SpatialBlockCV(
                        n_splits=cv_config.get('n_splits', 5),
                        block_size=cv_config.get('block_size', 100)
                    )
                elif cv_type == 'spatial_buffer':
                    cv = SpatialBufferCV(
                        n_splits=cv_config.get('n_splits', 5),
                        buffer_distance=cv_config.get('buffer_distance', 50)
                    )
                else:
                    cv = None
                
                if cv:
                    self.logger.info(f"Performing {cv_type} cross-validation...")
                    # For spatial CV, we need to pass the full data with coordinates
                    cv_results = model.cross_validate(X, y, cv_strategy=cv, lat_lon=features[['latitude', 'longitude']])
                    self.logger.info(f"CV R²: {cv_results.mean_metrics['r2']:.3f} ± {cv_results.std_metrics['r2']:.3f}")
            
            # Train final model
            self.logger.info("Training final model...")
            model.fit(X, y)
            
            # Evaluate
            metrics = model.evaluate(X, y, metrics=['r2', 'rmse', 'mae'])
            self.logger.info(f"Training metrics: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}")
            
            # Save outputs
            if ml_config.get('save_model', True):
                model_path = output_dir / f"{model_type}_model.pkl"
                # Check if save_model method exists
                if hasattr(model, 'save_model') and callable(getattr(model, 'save_model')):
                    model.save_model(model_path)
                    self.logger.info(f"Model saved to: {model_path}")
                else:
                    self.logger.warning("Model saving not implemented for this model type")
            
            if ml_config.get('save_predictions', True):
                predictions = model.predict(X)
                results_df = data.copy()
                results_df['predicted'] = predictions
                results_df['actual'] = y.values
                results_df['residual'] = y.values - predictions
                
                pred_path = output_dir / "predictions.parquet"
                results_df.to_parquet(pred_path)
                self.logger.info(f"Predictions saved to: {pred_path}")
            
            # Save metrics
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    'model_type': model_type,
                    'target_column': target_col,
                    'n_features': len(feature_cols),
                    'n_samples': len(X),
                    'metrics': metrics
                }, f, indent=2)
            
            self.logger.info("ML experiment completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"ML experiment failed: {e}")
            self.logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run standalone ML experiments on biodiversity data'
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiment', '-e',
        help='Named experiment from config.yml'
    )
    
    # Input data
    parser.add_argument(
        '--input', '-i',
        help='Input parquet file path (overrides experiment config)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-type', '-m',
        choices=['linear_regression', 'lightgbm'],
        help='Model type (overrides experiment config)'
    )
    
    parser.add_argument(
        '--target', '-t',
        help='Target column name (overrides experiment config)'
    )
    
    # Output options
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Skip cross-validation'
    )
    
    parser.add_argument(
        '--no-save-model',
        action='store_true',
        help='Do not save trained model'
    )
    
    # Configuration
    parser.add_argument(
        '--config-file',
        type=Path,
        help='Path to config.yml file'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize runner
    runner = StandaloneMLRunner(config_path=args.config_file)
    
    # Build ML configuration
    ml_config = {}
    
    # Start with experiment config if specified
    if args.experiment:
        try:
            ml_config = runner.load_experiment_config(args.experiment)
            runner.logger.info(f"Loaded experiment config: {args.experiment}")
        except ValueError as e:
            runner.logger.error(str(e))
            sys.exit(1)
    
    # Override with command line arguments
    if args.input:
        ml_config['input_parquet'] = args.input
    
    if args.model_type:
        ml_config['model_type'] = args.model_type
    
    if args.target:
        ml_config['target_column'] = args.target
    
    if args.no_cv:
        ml_config['perform_cv'] = False
    
    if args.no_save_model:
        ml_config['save_model'] = False
    
    # Validate we have input
    if 'input_parquet' not in ml_config:
        runner.logger.error("No input parquet file specified. Use --input or --experiment")
        sys.exit(1)
    
    # Run experiment
    success = runner.run_ml_experiment(ml_config)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
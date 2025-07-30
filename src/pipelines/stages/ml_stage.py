"""Machine Learning stage for biodiversity prediction."""

from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import gc
import json

from .base_stage import PipelineStage, StageResult
from src.infrastructure.logging import get_logger
from src.processors.data_preparation.analysis_data_source import ParquetAnalysisDataset
from src.base.dataset import BaseDataset
from src.machine_learning import (
    LinearRegressionAnalyzer,
    LightGBMAnalyzer,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV,
    SpatialBufferCV,
    EnvironmentalBlockCV
)
from src.core.registry import ComponentRegistry
from src.config import config

logger = get_logger(__name__)


class MLStage(PipelineStage):
    """Stage for machine learning model training and prediction."""
    
    def __init__(self, ml_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML stage.
        
        Args:
            ml_config: Configuration for ML stage with keys:
                - model_type: 'linear_regression' or 'lightgbm'
                - target_column: Name of target column to predict
                - feature_columns: List of feature columns (optional, auto-detected if None)
                - cv_strategy: Cross-validation strategy config
                - imputation_strategy: Missing value imputation config
                - save_predictions: Whether to save predictions
                - save_model: Whether to save trained model
        """
        super().__init__()
        self.ml_config = ml_config or {}
        self._model = None
        self._feature_builder = None
        self._imputer = None
        self._cv_strategy = None
        self._validate_config()
    
    @property
    def name(self) -> str:
        return "machine_learning"
    
    @property
    def dependencies(self) -> List[str]:
        # ML stage is standalone - no dependencies
        return []
    
    @property
    def memory_requirements(self) -> float:
        # ML can be memory intensive, especially with large datasets
        model_type = self.ml_config.get('model_type', 'linear_regression')
        if model_type == 'lightgbm':
            return 16.0  # LightGBM needs more memory
        return 12.0
    
    @property
    def supports_chunking(self) -> bool:
        """ML stage can process data in chunks for prediction."""
        return True
    
    def _validate_config(self):
        """Validate ML configuration."""
        # Set defaults
        self.ml_config.setdefault('input_parquet', None)  # Required - no default
        self.ml_config.setdefault('model_type', 'linear_regression')
        self.ml_config.setdefault('target_column', 'total_richness')
        self.ml_config.setdefault('save_predictions', True)
        self.ml_config.setdefault('save_model', True)
        
        # CV strategy defaults
        if 'cv_strategy' not in self.ml_config:
            self.ml_config['cv_strategy'] = {
                'type': 'spatial_block',
                'n_splits': 5,
                'block_size': 100
            }
        
        # Imputation defaults
        if 'imputation_strategy' not in self.ml_config:
            self.ml_config['imputation_strategy'] = {
                'type': 'spatial_knn',
                'n_neighbors': 10,
                'spatial_weight': 0.6
            }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate ML configuration and dependencies."""
        errors = []
        
        # Validate input parquet is specified
        input_parquet = self.ml_config.get('input_parquet')
        if not input_parquet:
            errors.append("Input parquet file must be specified in ml_config['input_parquet']")
        elif not Path(input_parquet).exists():
            errors.append(f"Input parquet file does not exist: {input_parquet}")
        
        # Validate model type
        model_type = self.ml_config.get('model_type')
        available_models = ComponentRegistry.get_available_ml_models()
        if model_type not in available_models:
            errors.append(f"Unknown model type: {model_type}. Available: {available_models}")
        
        # Validate target column is specified
        if not self.ml_config.get('target_column'):
            errors.append("Target column must be specified")
        
        # Validate CV strategy
        cv_type = self.ml_config.get('cv_strategy', {}).get('type')
        available_cv = ComponentRegistry.get_available_cv_strategies()
        if cv_type and cv_type not in available_cv:
            errors.append(f"Unknown CV strategy: {cv_type}. Available: {available_cv}")
        
        return len(errors) == 0, errors
    
    def execute(self, context) -> StageResult:
        """Execute ML pipeline."""
        logger.info(
            f"Starting ML pipeline with {self.ml_config.get('model_type')} model",
            extra={
                'experiment_id': context.experiment_id,
                'stage': self.name,
                'model_type': self.ml_config.get('model_type')
            }
        )
        
        try:
            # Step 1: Load dataset
            logger.debug("Loading dataset for ML")
            dataset = self._load_dataset(context)
            dataset_info = dataset.load_info()
            
            logger.info(
                f"Dataset loaded: {dataset_info.record_count:,} records, "
                f"{dataset_info.size_mb:.2f} MB"
            )
            
            # Step 2: Load data
            data = self._load_data(dataset)
            logger.info(f"Data shape: {data.shape}")
            
            # Step 3: Handle missing values
            data_imputed = self._impute_missing_values(data)
            
            # Step 4: Feature engineering
            features = self._engineer_features(data_imputed)
            logger.info(f"Created {len(features.columns)} features")
            
            # Step 5: Prepare target and features
            target_col = self.ml_config.get('target_column')
            
            # Handle composite feature builder prefixes
            if target_col not in features.columns:
                # Try with richness prefix
                prefixed_target = f'richness_{target_col}'
                if prefixed_target in features.columns:
                    target_col = prefixed_target
                else:
                    raise ValueError(f"Target column '{target_col}' not found in features")
            
            target = features[target_col].copy()
            
            # Get feature columns (exclude target-related columns)
            feature_cols = self.ml_config.get('feature_columns')
            if not feature_cols:
                # Auto-detect feature columns
                exclude_patterns = [target_col, 'richness', 'lat', 'lon', 'grid_id']
                feature_cols = [
                    col for col in features.columns 
                    if not any(pattern in col.lower() for pattern in exclude_patterns)
                ]
            
            X = features[feature_cols]
            logger.info(f"Using {len(feature_cols)} features for modeling")
            
            # Step 6: Initialize model
            self._model = self._create_model()
            
            # Step 7: Cross-validation (if requested)
            cv_results = None
            if self.ml_config.get('perform_cv', True):
                cv_results = self._perform_cross_validation(X, target, data_imputed)
            
            # Step 8: Train final model
            logger.info("Training final model on all data")
            ml_result = self._model.fit(X, target)
            
            # Step 9: Evaluate on training data
            train_metrics = self._model.evaluate(X, target, metrics=['r2', 'rmse', 'mae'])
            logger.info(f"Training metrics: {train_metrics}")
            
            # Step 10: Feature importance
            importance = self._model.get_feature_importance()
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.info("Top 10 important features:")
                for feat, score in top_features:
                    logger.info(f"  {feat}: {score:.4f}")
            
            # Step 11: Save outputs
            output_files = self._save_outputs(
                context, X, target, features, data_imputed, 
                cv_results, train_metrics, importance
            )
            
            # Calculate metrics
            metrics = {
                'records_processed': len(data),
                'features_created': len(feature_cols),
                'model_type': self.ml_config.get('model_type'),
                'train_r2': train_metrics.get('r2', 0),
                'train_rmse': train_metrics.get('rmse', 0)
            }
            
            if cv_results:
                metrics.update({
                    'cv_r2_mean': cv_results.mean_metrics.get('r2', 0),
                    'cv_r2_std': cv_results.std_metrics.get('r2', 0),
                    'cv_rmse_mean': cv_results.mean_metrics.get('rmse', 0),
                    'cv_rmse_std': cv_results.std_metrics.get('rmse', 0)
                })
            
            return StageResult(
                success=True,
                data={
                    'model_path': output_files.get('model_path'),
                    'predictions_path': output_files.get('predictions_path'),
                    'importance_path': output_files.get('importance_path'),
                    'metrics_path': output_files.get('metrics_path')
                },
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"ML stage failed: {e}", exc_info=True)
            return StageResult(
                success=False,
                data={},
                metrics={},
                warnings=[str(e)]
            )
        finally:
            # Cleanup
            self._cleanup_resources()
    
    def _load_dataset(self, context) -> BaseDataset:
        """Load dataset from configured parquet file."""
        parquet_path = self.ml_config.get('input_parquet')
        
        if not parquet_path:
            raise ValueError("Input parquet file must be specified in ml_config['input_parquet']")
        
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")
        
        logger.info(f"Loading data from parquet: {parquet_path}")
        return ParquetAnalysisDataset(
            file_path=parquet_path,
            experiment_id=context.experiment_id
        )
    
    def _load_data(self, dataset: BaseDataset) -> pd.DataFrame:
        """Load full dataset into DataFrame."""
        # For ML, we typically need all data in memory
        tiles = list(dataset.get_tiles())
        
        if len(tiles) == 1:
            # Single tile, load directly
            return dataset.read_tile(tiles[0])
        
        # Multiple tiles, concatenate
        dfs = []
        for tile in tiles:
            df = dataset.read_tile(tile)
            dfs.append(df)
            
        return pd.concat(dfs, ignore_index=True)
    
    def _impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using configured strategy."""
        imp_config = self.ml_config.get('imputation_strategy', {})
        imp_type = imp_config.get('type', 'spatial_knn')
        
        if imp_type == 'spatial_knn':
            self._imputer = SpatialKNNImputer(
                n_neighbors=imp_config.get('n_neighbors', 10),
                spatial_weight=imp_config.get('spatial_weight', 0.6)
            )
        else:
            # Could add other imputation strategies here
            raise ValueError(f"Unknown imputation strategy: {imp_type}")
        
        logger.info(f"Imputing missing values with {imp_type}")
        missing_before = data.isna().sum().sum()
        data_imputed = self._imputer.fit_transform(data)
        missing_after = data_imputed.isna().sum().sum()
        
        logger.info(
            f"Imputation complete: {missing_before} -> {missing_after} missing values"
        )
        
        return data_imputed
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using composite builder."""
        # Use composite builder to automatically use all registered builders
        self._feature_builder = CompositeFeatureBuilder(config.config)
        
        logger.info("Engineering features with composite builder")
        features = self._feature_builder.fit_transform(data)
        
        # Log feature summary
        summary = self._feature_builder.get_feature_summary()
        for category, info in summary['categories'].items():
            logger.info(f"  {category}: {info['count']} features")
        
        return features
    
    def _create_model(self):
        """Create ML model based on configuration."""
        model_type = self.ml_config.get('model_type')
        
        if model_type == 'linear_regression':
            return LinearRegressionAnalyzer(config.config)
        elif model_type == 'lightgbm':
            return LightGBMAnalyzer(config.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_cv_strategy(self, data: pd.DataFrame):
        """Create cross-validation strategy."""
        cv_config = self.ml_config.get('cv_strategy', {})
        cv_type = cv_config.get('type', 'spatial_block')
        
        if cv_type == 'spatial_block':
            return SpatialBlockCV(
                n_splits=cv_config.get('n_splits', 5),
                block_size=cv_config.get('block_size', 100),
                random_state=cv_config.get('random_state', 42)
            )
        elif cv_type == 'spatial_buffer':
            return SpatialBufferCV(
                n_splits=cv_config.get('n_splits', 5),
                buffer_distance=cv_config.get('buffer_distance', 50),
                random_state=cv_config.get('random_state', 42)
            )
        elif cv_type == 'environmental':
            return EnvironmentalBlockCV(
                n_splits=cv_config.get('n_splits', 5),
                stratify_by=cv_config.get('stratify_by', 'latitude')
            )
        else:
            raise ValueError(f"Unknown CV strategy: {cv_type}")
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                  data: pd.DataFrame):
        """Perform cross-validation."""
        cv_strategy = self._create_cv_strategy(data)
        
        logger.info(f"Performing {cv_strategy.__class__.__name__} cross-validation")
        
        cv_results = self._model.cross_validate(
            X, y,
            cv_strategy=cv_strategy,
            metrics=['r2', 'rmse', 'mae']
        )
        
        logger.info(f"CV Results:")
        logger.info(f"  R²: {cv_results.mean_metrics['r2']:.3f} ± {cv_results.std_metrics['r2']:.3f}")
        logger.info(f"  RMSE: {cv_results.mean_metrics['rmse']:.2f} ± {cv_results.std_metrics['rmse']:.2f}")
        logger.info(f"  MAE: {cv_results.mean_metrics['mae']:.2f} ± {cv_results.std_metrics['mae']:.2f}")
        
        return cv_results
    
    def _save_outputs(self, context, X, y, features, original_data,
                      cv_results, train_metrics, importance):
        """Save model outputs."""
        output_dir = Path(context.output_dir) / "ml_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Save model
        if self.ml_config.get('save_model', True):
            model_path = output_dir / f"{self.ml_config.get('model_type')}_model.pkl"
            self._model.save_model(model_path)
            output_files['model_path'] = str(model_path)
            logger.info(f"Model saved to: {model_path}")
        
        # Save predictions
        if self.ml_config.get('save_predictions', True):
            predictions = self._model.predict(X)
            
            # Create results DataFrame
            results_df = original_data.copy()
            results_df['predicted'] = predictions
            results_df['actual'] = y.values
            results_df['residual'] = y.values - predictions
            
            # Save as parquet
            pred_path = output_dir / "predictions.parquet"
            results_df.to_parquet(pred_path)
            output_files['predictions_path'] = str(pred_path)
            logger.info(f"Predictions saved to: {pred_path}")
        
        # Save feature importance
        if importance:
            imp_df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in importance.items()
            ]).sort_values('importance', ascending=False)
            
            imp_path = output_dir / "feature_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            output_files['importance_path'] = str(imp_path)
        
        # Save metrics summary
        metrics_summary = {
            'model_type': self.ml_config.get('model_type'),
            'target_column': self.ml_config.get('target_column'),
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_metrics': train_metrics
        }
        
        if cv_results:
            metrics_summary['cv_results'] = {
                'mean_metrics': cv_results.mean_metrics,
                'std_metrics': cv_results.std_metrics,
                'n_folds': len(cv_results.fold_metrics)
            }
        
        metrics_path = output_dir / "ml_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        output_files['metrics_path'] = str(metrics_path)
        
        return output_files
    
    def _cleanup_resources(self):
        """Clean up resources."""
        # Garbage collect to free memory
        if hasattr(self, '_model'):
            del self._model
        if hasattr(self, '_feature_builder'):
            del self._feature_builder
        if hasattr(self, '_imputer'):
            del self._imputer
        if hasattr(self, '_cv_strategy'):
            del self._cv_strategy
        
        gc.collect()
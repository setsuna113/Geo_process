#!/usr/bin/env python3
"""
Optimized LightGBM Regressor for EconML/DML pipelines
The BEST version combining all optimizations from both implementations

This is what happens when you actually think about performance instead of
using defaults like a script kiddie.
"""

import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
import psutil
import time
from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd


class OptimizedLGBMRegressor(BaseEstimator, RegressorMixin):
    """
    LightGBM wrapper optimized for MAXIMUM SPEED on large datasets
    
    Combines the best of both implementations:
    - Handles sample weights (critical for DML)
    - No unnecessary pandas conversions
    - Proper 2D array handling
    - All optimization parameters
    - Optional timing reports
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 dataset_size: int,
                 num_threads: int):
        """
        Args:
            config: Full configuration dictionary (REQUIRED - no defaults)
            dataset_size: Number of rows in dataset (REQUIRED)
            num_threads: Number of threads to use (REQUIRED)
        """
        if not config:
            raise ValueError("config is required - no defaults provided")
        if not dataset_size:
            raise ValueError("dataset_size is required")
        if not num_threads:
            raise ValueError("num_threads is required")
        
        self.config = config
        self.dataset_size = dataset_size
        self.n_cores = num_threads
        self.model_ = None
        
        # Extract key parameters from config
        self.objective = config['objective']
        self.metric = config['metric']
        self.boosting_type = config['boosting_type']
        self.verbosity = config['verbosity']
        self.seed = config['seed']
        
        # Tree structure
        self.num_leaves = config['num_leaves']
        self.min_data_in_leaf = config['min_data_in_leaf']
        self.max_depth = config['lgbm_max_depth']
        self.learning_rate = config['learning_rate']
        self.num_iterations = config['num_iterations']
        
        # Sampling
        self.feature_fraction = config['feature_fraction']
        self.feature_fraction_bynode = config['feature_fraction_bynode']
        self.bagging_fraction = config['bagging_fraction']
        self.bagging_freq = config['bagging_freq']
        
        # Other parameters
        self.max_bin = config['max_bin']
        self.early_stopping_rounds = config['early_stopping_rounds']
        self.first_metric_only = config['first_metric_only']
        self.cat_smooth = config['cat_smooth']
        self.max_cat_to_onehot = config['max_cat_to_onehot']
        self.categorical_feature = config['categorical_feature']
        self.use_missing = config['use_missing']
        self.zero_as_missing = config['zero_as_missing']
        self.feature_pre_filter = config['feature_pre_filter']
        self.extra_trees = config['extra_trees']
        
        self.total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        
    def _get_optimized_params(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """
        Build LightGBM parameters from config with data-dependent adjustments
        """
        
        # Base parameters from config
        params = {
            'objective': self.objective,
            'metric': self.metric,
            'num_threads': self.n_cores,
            'verbosity': self.verbosity,
            'boosting_type': self.boosting_type,
            'seed': self.seed,
            
            # SPEED OPTIMIZATIONS - the difference between 1 hour and 1 minute
            # Use row-wise for few features (<50), col-wise for many features
            'force_row_wise': True if n_features < 50 else False,
            'force_col_wise': True if n_features >= 50 else False,
            
            # Histogram optimizations
            'max_bin': self.max_bin,
            'min_data_in_bin': max(3, n_samples // 1_000_000),  # Still auto-scale this
            
            # Memory pool - prevent constant reallocation
            'histogram_pool_size': min(512, self.total_ram_mb // 4),  # MB
            
            # Feature sampling from config with data-dependent adjustments
            'feature_fraction': 0.7 if n_features > 100 else self.feature_fraction,
            'feature_fraction_bynode': 0.7 if n_features > 100 else self.feature_fraction_bynode,
            
            # Bagging from config with data-dependent adjustments
            'bagging_fraction': 0.8 if n_samples > 1_000_000 else self.bagging_fraction,
            'bagging_freq': 1 if n_samples > 1_000_000 else self.bagging_freq,
            
            # Tree parameters from config with data-dependent adjustments
            'num_leaves': 31 if n_samples < 100_000 else self.num_leaves,
            'min_data_in_leaf': max(self.min_data_in_leaf, n_samples // 100_000),
            'learning_rate': self.learning_rate,
            'num_iterations': self.num_iterations,
            
            # Early stopping from config
            'early_stopping_rounds': self.early_stopping_rounds,
            'first_metric_only': self.first_metric_only,
            
            # Categorical features from config
            'cat_smooth': self.cat_smooth,
            'max_cat_to_onehot': self.max_cat_to_onehot,
            
            # Advanced optimizations from config
            'use_missing': self.use_missing,
            'zero_as_missing': self.zero_as_missing,
            'feature_pre_filter': self.feature_pre_filter,
            'extra_trees': self.extra_trees,
            'deterministic': True if self.n_cores == 1 else False,  # Still auto-set this
        }
        
        # Dataset-specific optimizations for large data
        # REMOVED GOSS - using bagging for all large datasets (more stable for DML)
        if n_samples > 1_000_000:
            params.update({
                # Ensure at least 200k samples for histogram construction
                # but cap at 500k to avoid excessive memory/time
                'bin_construct_sample_cnt': max(200_000, min(500_000, n_samples // 10)),
                # Keep bagging for all large datasets - more stable than GOSS
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
            })
        
        # No kwargs override - everything from config
        return params
    
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple] = None,
            categorical_feature: Union[str, list] = 'auto'):
        """
        Fit with optimized parameters
        
        Args:
            X: Features (numpy or pandas)
            y: Target (numpy or pandas)
            sample_weight: Sample weights for DML
            eval_set: Validation set tuple (X_val, y_val)
            categorical_feature: Categorical columns
        """
        # Get dimensions BEFORE any conversion
        if hasattr(X, 'shape'):
            n_samples, n_features = X.shape
        else:
            n_samples = len(X)
            n_features = len(X[0]) if len(X) > 0 else 0
        
        # Handle 2D y arrays (critical for MultiOutputRegressor)
        if hasattr(y, 'shape') and len(y.shape) > 1:
            y = y.ravel()
        
        # Get optimized parameters
        params = self._get_optimized_params(n_samples, n_features)
        
        # Create dataset - NO PANDAS CONVERSION, LightGBM handles it
        start_time = time.time()
        train_data = lgb.Dataset(
            X, label=y,
            categorical_feature=categorical_feature,
            weight=sample_weight,  # Critical for DML
            free_raw_data=False,  # Keep raw data for predictions
            params={'bin_construct_sample_cnt': params.get('bin_construct_sample_cnt', 200000)}
        )
        construct_time = time.time() - start_time
        
        # Validation set if provided
        valid_sets = [train_data]
        if eval_set is not None:
            X_val, y_val = eval_set
            if hasattr(y_val, 'shape') and len(y_val.shape) > 1:
                y_val = y_val.ravel()
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
        
        # Train model
        start_time = time.time()
        self.model_ = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            callbacks=[lgb.log_evaluation(0)] if params['verbosity'] < 0 else None
        )
        train_time = time.time() - start_time
        
        # Report timing if verbose
        if self.verbosity >= 0:
            print(f"[LightGBM] Dataset construction: {construct_time:.2f}s")
            print(f"[LightGBM] Training: {train_time:.2f}s")
            print(f"[LightGBM] Features: {n_features}, Samples: {n_samples:,}")
            print(f"[LightGBM] Iterations: {self.model_.current_iteration()}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fast prediction - no unnecessary conversions
        
        Args:
            X: Features (numpy or pandas)
            
        Returns:
            Predictions as numpy array
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # LightGBM handles both numpy and pandas directly
        return self.model_.predict(X, num_threads=self.n_cores)
    
    def get_params(self, deep=True) -> Dict[str, Any]:
        """For sklearn compatibility - return config"""
        return {
            'config': self.config,
            'dataset_size': self.dataset_size,
            'num_threads': self.n_cores
        }
    
    def set_params(self, **params) -> 'OptimizedLGBMRegressor':
        """For sklearn compatibility - updates not allowed"""
        raise NotImplementedError("Cannot modify parameters after initialization - use new config")
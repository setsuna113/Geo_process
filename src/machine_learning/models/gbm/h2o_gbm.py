"""H2O GBM wrapper for replacing LightGBM with better parallelization.

Optimized for 256-core CPU machines. Drop-in replacement for LightGBM.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

logger = logging.getLogger(__name__)


class H2OGBM:
    """H2O Gradient Boosting Machine with massive parallelization.
    
    Drop-in replacement for LightGBM that utilizes all available CPU cores
    for dramatic speedup on multi-core machines.
    """
    
    def __init__(self,
                 n_threads: int = -1,
                 max_mem_size: str = "32G",
                 **kwargs):
        """Initialize H2O GBM.
        
        Args:
            n_threads: Number of threads (-1 for all available)
            max_mem_size: Maximum memory for H2O cluster
            **kwargs: GBM parameters (similar to LightGBM)
        """
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count()
        self.max_mem_size = max_mem_size
        
        # Map LightGBM params to H2O params
        self.params = self._map_lightgbm_params(kwargs)
        
        # Initialize H2O using centralized utility
        from ...utils.h2o_utils import ensure_h2o_initialized
        ensure_h2o_initialized(
            n_threads=self.n_threads,
            max_mem_size=self.max_mem_size
        )
        
        self.model = None
        self.feature_names = None
        
    def _map_lightgbm_params(self, lgb_params: Dict) -> Dict:
        """Map LightGBM parameters to H2O equivalents."""
        h2o_params = {
            'ntrees': lgb_params.get('num_iterations', 200),
            'max_depth': lgb_params.get('max_depth', 15),
            'min_rows': lgb_params.get('min_data_in_leaf', 20),
            'learn_rate': lgb_params.get('learning_rate', 0.05),
            'sample_rate': lgb_params.get('bagging_fraction', 0.8),
            'col_sample_rate': lgb_params.get('feature_fraction', 0.7),
            'col_sample_rate_per_tree': lgb_params.get('feature_fraction', 0.7),
            'seed': lgb_params.get('random_state', 42),
            'distribution': 'gaussian',  # For regression
            'stopping_rounds': 5,
            'stopping_metric': 'rmse',
            'stopping_tolerance': 0.001,
        }
        
        # Handle objective
        objective = lgb_params.get('objective', 'regression')
        if 'regression' in objective:
            h2o_params['distribution'] = 'gaussian'
        elif 'binary' in objective:
            h2o_params['distribution'] = 'bernoulli'
        elif 'multiclass' in objective:
            h2o_params['distribution'] = 'multinomial'
        
        return h2o_params
    
    def train(self,
             train_set: Union[Dict, h2o.H2OFrame],
             num_boost_round: Optional[int] = None,
             valid_sets: Optional[List] = None,
             callbacks: Optional[List] = None) -> 'H2OGBM':
        """Train the GBM model (LightGBM-compatible interface).
        
        Args:
            train_set: Training data (LightGBM Dataset dict or H2OFrame)
            num_boost_round: Number of boosting iterations
            valid_sets: Validation datasets
            callbacks: Callbacks (ignored for H2O)
            
        Returns:
            Self (trained model)
        """
        # Handle LightGBM Dataset format
        if isinstance(train_set, dict):
            # Assume it's from lgb.Dataset
            X_train = train_set.get('data')
            y_train = train_set.get('label')
            
            # Convert to H2O
            if isinstance(X_train, pd.DataFrame):
                train_df = X_train.copy()
                train_df['target'] = y_train
            else:
                train_df = pd.DataFrame(X_train)
                train_df['target'] = y_train
            
            train_h2o = h2o.H2OFrame(train_df)
            predictors = [col for col in train_h2o.columns if col != 'target']
            response = 'target'
            
        elif isinstance(train_set, h2o.H2OFrame):
            # Direct H2O input
            train_h2o = train_set
            # Assume last column is target
            response = train_h2o.columns[-1]
            predictors = [col for col in train_h2o.columns if col != response]
        else:
            raise ValueError(f"Unsupported train_set type: {type(train_set)}")
        
        # Update number of trees if specified
        if num_boost_round:
            self.params['ntrees'] = num_boost_round
        
        # Train model
        logger.info(f"Training H2O GBM with {self.n_threads} threads...")
        logger.info(f"Parameters: {self.params}")
        
        self.model = H2OGradientBoostingEstimator(**self.params)
        self.model.train(
            x=predictors,
            y=response,
            training_frame=train_h2o
        )
        
        self.feature_names = predictors
        
        logger.info("H2O GBM training complete")
        return self
    
    def predict(self,
               X: Union[np.ndarray, pd.DataFrame, h2o.H2OFrame],
               num_iteration: Optional[int] = None) -> np.ndarray:
        """Make predictions (LightGBM-compatible interface).
        
        Args:
            X: Features to predict on
            num_iteration: Number of iterations to use (ignored)
            
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to H2O if needed
        if isinstance(X, h2o.H2OFrame):
            test_h2o = X
        elif isinstance(X, pd.DataFrame):
            test_h2o = h2o.H2OFrame(X)
        else:
            # Numpy array
            test_df = pd.DataFrame(X, columns=self.feature_names)
            test_h2o = h2o.H2OFrame(test_df)
        
        # Make predictions
        preds_h2o = self.model.predict(test_h2o)
        
        # Convert to numpy
        preds = preds_h2o.as_data_frame().values.flatten()
        
        return preds
    
    def feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """Get feature importance (LightGBM-compatible interface).
        
        Args:
            importance_type: Type of importance (ignored, H2O uses its own)
            
        Returns:
            Feature importance values
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get variable importance from H2O
        importance_df = self.model.varimp(use_pandas=True)
        
        # Return importance values in same order as features
        importance_dict = dict(zip(importance_df['variable'], importance_df['relative_importance']))
        importance = [importance_dict.get(f, 0) for f in self.feature_names]
        
        return np.array(importance)
    
    @property
    def feature_name(self) -> List[str]:
        """Get feature names (LightGBM-compatible property)."""
        return self.feature_names
    
    def __del__(self):
        """Clean up H2O model on deletion."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                h2o.remove(self.model)
            except:
                pass


def train(params: Dict, 
          train_set,
          num_boost_round: int = 100,
          valid_sets=None,
          callbacks=None,
          **kwargs) -> H2OGBM:
    """LightGBM-compatible train function.
    
    This allows drop-in replacement:
    Replace: model = lgb.train(params, train_data, ...)
    With:    model = h2o_gbm.train(params, train_data, ...)
    """
    # Extract H2O-specific params
    n_threads = params.pop('num_threads', os.cpu_count())
    
    # Create and train model
    model = H2OGBM(n_threads=n_threads, **params)
    model.train(
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks
    )
    
    return model


class Dataset:
    """LightGBM Dataset compatibility wrapper."""
    
    def __init__(self, data, label=None, reference=None, **kwargs):
        """Create a dataset compatible with LightGBM interface."""
        self.data = data
        self.label = label
        self.reference = reference
        self.params = kwargs
    
    def construct(self):
        """Construct the dataset (no-op for compatibility)."""
        pass
    
    def __getitem__(self, key):
        """Allow dict-like access."""
        if key == 'data':
            return self.data
        elif key == 'label':
            return self.label
        else:
            return self.params.get(key)
    
    def get(self, key, default=None):
        """Dict-like get method."""
        try:
            return self[key]
        except KeyError:
            return default
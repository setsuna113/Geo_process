"""Method 3: Double Machine Learning (DML) - H2O optimized version.

This module now redirects to the H2O implementation for massive datasets.
The original pandas/LightGBM version has been replaced for 180M+ point efficiency.
"""

import logging
from typing import Dict, Any, Optional

# Import the H2O version
from .method3_dml_h2o import Method3DMLH2O
from ...config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)


class Method3DML:
    """Method 3: Double Machine Learning - Now using H2O implementation.
    
    This class is a wrapper that redirects to the H2O version for efficiency.
    The original pandas/LightGBM implementation has been replaced to handle
    180M+ points without memory duplication.
    
    Key improvements:
    1. All operations in H2O (256-thread parallel)
    2. Weighted bootstrap instead of data copies (avoids 3TB memory ops)
    3. Direct parquet loading (no pandas intermediate)
    4. H2O GBM instead of LightGBM for consistency
    """
    
    def __init__(self, config: Optional[MLRegressionConfig] = None):
        """Initialize Method 3 DML using H2O backend.
        
        Args:
            config: ML regression configuration
        """
        logger.warning("Method3DML now uses H2O backend for 180M+ point efficiency")
        self.config = config or MLRegressionConfig()
        
        # Create H2O implementation
        self.h2o_impl = Method3DMLH2O(self.config)
        
        # Delegate attributes for compatibility
        self.n_folds = self.h2o_impl.n_folds
        self.data_processed = None  # Compatibility
        self.causal_effects = self.h2o_impl.causal_effects
        
        logger.info("Method 3 DML initialized with H2O backend (256-core parallel, weighted bootstrap)")
    
    def prepare_data(self, data: Any) -> Any:
        """Prepare data - now delegates to H2O implementation.
        
        Args:
            data: Can be a parquet path (str) or H2OFrame
        
        Returns:
            H2OFrame ready for DML
        """
        logger.info("Delegating to H2O implementation for data preparation")
        
        # If string path, load directly to H2O
        if isinstance(data, str):
            h2o_data = self.h2o_impl.load_data(data)
        else:
            # Assume it's already H2OFrame
            h2o_data = data
        
        # Process in H2O
        processed = self.h2o_impl.prepare_data_h2o(h2o_data)
        self.data_processed = processed  # Store for compatibility
        return processed
    
    def cross_fit_dml(self, hypothesis: str = 'h1') -> Dict[str, Any]:
        """Perform cross-fitted DML estimation.
        
        Args:
            hypothesis: Which hypothesis to test ('h0', 'h1', 'h2')
            
        Returns:
            Dictionary with causal effect estimates and statistics
        """
        return self.h2o_impl.cross_fit_dml(hypothesis)
    
    def test_robustness(self, hypothesis: str = 'h1') -> Dict[str, Any]:
        """Test robustness using weighted bootstrap.
        
        Uses weighted bootstrap instead of data copies to avoid memory explosion.
        For 180M points: saves 100 × 30GB = 3TB of memory operations!
        
        Args:
            hypothesis: Which hypothesis to test
            
        Returns:
            Robustness test results including bootstrap CI
        """
        logger.info("Using weighted bootstrap (no data copies!)")
        return self.h2o_impl.test_hypothesis(hypothesis)
    
    def run_full_analysis(self, data_path: str) -> Dict[str, Any]:
        """Run complete DML analysis on large dataset.
        
        Args:
            data_path: Path to parquet file
            
        Returns:
            Complete analysis results
        """
        logger.info("Running full DML analysis with H2O backend")
        return self.h2o_impl.process_large_dataset(data_path)
    
    # Backwards compatibility methods
    def fit_nuisance_models(self, *args, **kwargs):
        """Deprecated - use cross_fit_dml instead."""
        logger.warning("fit_nuisance_models is deprecated, using cross_fit_dml")
        return self.cross_fit_dml()
    
    def compute_causal_effect(self, hypothesis: str = 'h1') -> Dict[str, Any]:
        """Compute causal effect for hypothesis."""
        return self.h2o_impl.cross_fit_dml(hypothesis)
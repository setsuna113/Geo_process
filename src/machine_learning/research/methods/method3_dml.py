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
    
    def test_all_hypotheses(self) -> Dict[str, Any]:
        """Test all hypotheses (H0, H1, H2).
        
        Returns:
            Dictionary with results for all hypotheses
        """
        results = {}
        for hyp in ['h0', 'h1', 'h2']:
            try:
                result = self.h2o_impl.test_hypothesis(hyp)
                results[hyp] = {
                    'causal_effect': result.get('causal_effect', 0),
                    'standard_error': result.get('standard_error', 0),
                    'p_value': result.get('p_value', 1),
                    'significant': result.get('p_value', 1) < 0.05,
                    'interpretation': self._interpret_hypothesis(hyp, result)
                }
            except Exception as e:
                logger.warning(f"Failed to test {hyp}: {e}")
                results[hyp] = {
                    'causal_effect': None,
                    'p_value': None,
                    'error': str(e)
                }
        return results
    
    def _interpret_hypothesis(self, hyp: str, result: Dict[str, Any]) -> str:
        """Interpret hypothesis test results."""
        if result.get('p_value') is None:
            return f"Could not test {hyp}"
        
        p_val = result['p_value']
        effect = result.get('causal_effect', 0)
        
        if hyp == 'h0':
            if p_val >= 0.05:
                return "Fungi richness is independent of plant/vertebrate richness (H0 supported)"
            else:
                return "Fungi richness depends on plant/vertebrate richness (H0 rejected)"
        elif hyp == 'h1':
            if p_val < 0.05 and effect > 0:
                return "Positive causal effect: biodiversity begets biodiversity (H1 supported)"
            else:
                return "No positive causal effect detected (H1 not supported)"
        elif hyp == 'h2':
            if p_val < 0.05 and effect < 0:
                return "Temperate mismatch detected: negative interaction (H2 supported)"
            else:
                return "No temperate mismatch detected (H2 not supported)"
        return "Unknown hypothesis"
    
    def run_robustness_checks(self) -> Dict[str, Any]:
        """Run robustness checks using weighted bootstrap.
        
        Returns:
            Robustness test results
        """
        try:
            # Test robustness for main hypothesis (H1)
            return self.h2o_impl.test_robustness('h1')
        except Exception as e:
            logger.error(f"Robustness checks failed: {e}")
            return {'error': str(e)}
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize Method 3 DML results.
        
        Returns:
            Dictionary with summary
        """
        summary = {
            'method': 'Method 3: Double Machine Learning (DML)',
            'implementation': 'H2O backend with weighted bootstrap',
            'n_folds': self.n_folds
        }
        
        # Add hypothesis test results if available
        if hasattr(self.h2o_impl, 'causal_effects') and self.h2o_impl.causal_effects:
            summary['causal_effects'] = self.h2o_impl.causal_effects
        
        return summary
    
    def create_visualization(self, save_path: Optional[str] = None):
        """Create DML visualization.
        
        Args:
            save_path: Path to save visualization
        """
        logger.info(f"DML visualization would be saved to {save_path}")
        # Placeholder for DML-specific visualizations
        pass
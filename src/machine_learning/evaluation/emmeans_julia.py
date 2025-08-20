"""
EMmeans wrapper for Julia GLM results using R emmeans package.

Since Julia provides coefficients and vcov matrix, we can directly use R's emmeans.
No need for bootstrap or vcov calculation - Julia already did that.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .emmeans_core import REMmeansWrapper

logger = logging.getLogger(__name__)


class JuliaEMmeans:
    """EMmeans calculator for Julia GLM results with caching."""
    
    def __init__(self, julia_model: Dict[str, Any], feature_names: List[str]):
        """
        Initialize EMmeans for Julia model results.
        
        Args:
            julia_model: Dictionary containing Julia GLM results with:
                - coefficients: coefficient values
                - std_errors: standard errors  
                - vcov: variance-covariance matrix
                - coef_names: coefficient names
                - df_residual: residual degrees of freedom
            feature_names: List of feature names used in the model
        """
        self.julia_model = julia_model
        self.feature_names = feature_names
        
        # Extract components from Julia results
        self.coefficients = np.array(julia_model['coefficients'])
        self.vcov = np.array(julia_model['vcov'])
        self.coef_names = julia_model['coef_names']
        self.df_residual = julia_model.get('df_residual', len(julia_model['coefficients']))
        
        # Verify vcov shape
        n_coef = len(self.coefficients)
        if self.vcov.shape != (n_coef, n_coef):
            raise ValueError(f"vcov shape {self.vcov.shape} doesn't match coefficients ({n_coef})")
        
        # Cache for computed EMmeans
        self._emmeans_cache = {}
        self._contrasts_cache = {}
        
        logger.info(f"Initialized EMmeans for Julia model with {n_coef} coefficients")
    
    def compute_emmeans(self, factors: Dict[str, List], type: str = 'response') -> pd.DataFrame:
        """
        Compute estimated marginal means using R emmeans package with caching.
        
        Results are cached since EMmeans are deterministic for given coefficients.
        
        Args:
            factors: Dictionary of factor names and their levels
            type: 'response' for back-transformed or 'link' for linear predictor scale
            
        Returns:
            DataFrame with EMmeans results (cached if previously computed)
        """
        # Create cache key from factors and type
        cache_key = self._create_cache_key(factors, type)
        
        # Check cache first
        if cache_key in self._emmeans_cache:
            logger.info(f"Returning cached EMmeans for key: {cache_key}")
            return self._emmeans_cache[cache_key].copy()  # Return copy to prevent modification
        
        # Check R availability
        from .emmeans_core import R_AVAILABLE
        
        if not R_AVAILABLE:
            raise ImportError(
                "R emmeans package is required. Install R and run:\n"
                "R -e \"install.packages('emmeans', repos='https://cran.r-project.org')\""
            )
        
        # Compute and cache
        logger.info(f"Computing new EMmeans for key: {cache_key}")
        result = self._compute_emmeans_r(factors, type)
        self._emmeans_cache[cache_key] = result.copy()  # Store copy
        
        return result
    
    def _compute_emmeans_r(self, factors: Dict[str, List], type: str) -> pd.DataFrame:
        """Compute EMmeans using R emmeans package."""
        # Create dummy data frame for emmeans
        n_dummy = 100
        dummy_data = pd.DataFrame()
        
        # Add factor columns
        for factor, levels in factors.items():
            dummy_data[factor] = np.random.choice(levels, n_dummy)
        
        # Add continuous predictors at their means (zeros for standardized)
        continuous_vars = [name for name in self.feature_names 
                          if name not in factors and name != 'temperate_x_plant_high']
        for var in continuous_vars:
            dummy_data[var] = 0.0  # Standardized variables have mean 0
        
        # Add dummy outcome variable (needed for reference grid creation)
        # The actual values don't matter as we're using qdrg with coefficients
        dummy_data['fungi_richness'] = np.random.poisson(100, n_dummy)
        
        # Create formula
        formula = "fungi_richness ~ " + " + ".join(self.feature_names)
        
        # Initialize R wrapper
        wrapper = REMmeansWrapper(
            coefficients=self.coefficients,
            vcov_matrix=self.vcov,
            coefficient_names=self.coef_names,
            data=dummy_data,
            formula=formula,
            family='nbinom2',  # Negative binomial
            link='log',
            df_residual=self.df_residual
        )
        
        # Compute EMmeans
        # emmeans expects specs in formula format with tilde
        specs = "~ " + " * ".join(factors.keys())
        emmeans_results = wrapper.compute_emmeans(specs=specs, type=type)
        
        return emmeans_results
    
    
    def _create_cache_key(self, factors: Dict[str, List], type: str) -> str:
        """
        Create a unique cache key for EMmeans computation.
        
        Args:
            factors: Dictionary of factor names and their levels
            type: 'response' or 'link'
            
        Returns:
            String cache key
        """
        # Sort factors for consistent key
        sorted_factors = sorted(factors.items())
        factor_str = "_".join([f"{k}={sorted(v)}" for k, v in sorted_factors])
        return f"{factor_str}_{type}"
    
    def compute_contrasts(self, emmeans_df: pd.DataFrame, contrast_type: str = 'pairwise') -> Dict[str, Any]:
        """
        Compute contrasts from EMmeans with caching.
        
        Args:
            emmeans_df: DataFrame with EMmeans results
            contrast_type: Type of contrasts ('pairwise', 'interaction', etc.)
            
        Returns:
            Dictionary with contrast results (cached if previously computed)
        """
        # Create cache key based on EMmeans structure and contrast type
        emmeans_hash = pd.util.hash_pandas_object(emmeans_df).sum()
        cache_key = f"{emmeans_hash}_{contrast_type}"
        
        # Check cache
        if cache_key in self._contrasts_cache:
            logger.info(f"Returning cached contrasts for key: {cache_key}")
            return self._contrasts_cache[cache_key].copy()
        
        # Compute new contrasts
        logger.info(f"Computing new contrasts for key: {cache_key}")
        contrasts = self._compute_contrasts_impl(emmeans_df, contrast_type)
        
        # Cache and return
        self._contrasts_cache[cache_key] = contrasts.copy()
        return contrasts
    
    def _compute_contrasts_impl(self, emmeans_df: pd.DataFrame, contrast_type: str) -> Dict[str, Any]:
        """
        Implementation of contrast computation.
        
        Args:
            emmeans_df: DataFrame with EMmeans results  
            contrast_type: Type of contrasts
            
        Returns:
            Dictionary with contrast results
        """
        contrasts = {}
        
        # For 2x2 design, compute interaction contrast
        if len(emmeans_df) == 4:
            # Assuming order: (0,0), (0,1), (1,0), (1,1)
            emmeans_vals = emmeans_df['emmean'].values
            
            # Interaction contrast: (11 - 10) - (01 - 00)
            if len(emmeans_vals) == 4:
                interaction = (emmeans_vals[3] - emmeans_vals[2]) - (emmeans_vals[1] - emmeans_vals[0])
                
                contrasts['interaction'] = {
                    'estimate': interaction,
                    'interpretation': 'Difference in plant effect between temperate and non-temperate'
                }
                
                # Main effects
                contrasts['temperate_main'] = {
                    'estimate': np.mean([emmeans_vals[2], emmeans_vals[3]]) - np.mean([emmeans_vals[0], emmeans_vals[1]]),
                    'interpretation': 'Average effect of temperate'
                }
                
                contrasts['plant_main'] = {
                    'estimate': np.mean([emmeans_vals[1], emmeans_vals[3]]) - np.mean([emmeans_vals[0], emmeans_vals[2]]),
                    'interpretation': 'Average effect of high plant diversity'
                }
        
        return contrasts
    
    def clear_cache(self) -> None:
        """Clear all cached EMmeans and contrasts results."""
        self._emmeans_cache.clear()
        self._contrasts_cache.clear()
        logger.info("Cleared all EMmeans and contrasts caches")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get information about cached results.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'emmeans_cached': len(self._emmeans_cache),
            'contrasts_cached': len(self._contrasts_cache),
            'emmeans_keys': list(self._emmeans_cache.keys()),
            'contrasts_keys': list(self._contrasts_cache.keys())
        }
    
    def save_cache_to_disk(self, filepath: str) -> None:
        """
        Save cached results to disk for persistence.
        
        Args:
            filepath: Path to save cache (as pickle or JSON)
        """
        import pickle
        cache_data = {
            'emmeans': self._emmeans_cache,
            'contrasts': self._contrasts_cache,
            'coefficients': self.coefficients.tolist(),
            'coef_names': self.coef_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved cache to {filepath}")
    
    def load_cache_from_disk(self, filepath: str) -> None:
        """
        Load cached results from disk.
        
        Args:
            filepath: Path to load cache from
        """
        import pickle
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify coefficients match (ensures cache is for same model)
        loaded_coefs = np.array(cache_data['coefficients'])
        if not np.allclose(loaded_coefs, self.coefficients):
            logger.warning("Cached coefficients don't match current model. Cache not loaded.")
            return
        
        self._emmeans_cache = cache_data['emmeans']
        self._contrasts_cache = cache_data['contrasts']
        logger.info(f"Loaded cache from {filepath}: {len(self._emmeans_cache)} EMmeans, {len(self._contrasts_cache)} contrasts")
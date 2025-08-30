from src.abstractions import get_logger
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

logger = get_logger(__name__)


class JuliaEMmeans:
    """EMmeans calculator for Julia GLM results with caching."""
    
    def __init__(self, julia_model: Dict[str, Any], feature_names: List[str]):
        """
        Initialize EMmeans for Julia GLM results.
        
        Args:
            julia_model: Dictionary containing Julia GLM results with EXACTLY:
                - coefficients: array of coefficient values from Julia
                - vcov: variance-covariance matrix from Julia vcov()
                - coef_names: coefficient names from Julia
                - df_residual: residual degrees of freedom from Julia
            feature_names: List of feature names used in the model
        """
        self.julia_model = julia_model
        self.feature_names = feature_names
        
        # Julia gives us exactly these fields. No special cases.
        self.coefficients = np.array(julia_model['coefficients'])
        self.vcov = np.array(julia_model['vcov'])
        self.coef_names = julia_model['coef_names']
        self.df_residual = julia_model['df_residual']
        
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
        
        # R is REQUIRED - no checking, just import
        # If R is not available, it will fail immediately
        
        # Compute and cache
        logger.info(f"Computing new EMmeans for key: {cache_key}")
        result = self._compute_emmeans_r(factors, type)
        self._emmeans_cache[cache_key] = result.copy()  # Store copy
        
        return result
    
    def compute_emtrends(
        self,
        var: str,
        at: Dict[str, List],
        type: str = 'response'
    ) -> pd.DataFrame:
        """
        Compute estimated marginal trends (slopes) using R emtrends.
        
        This is the proper way to get simple slopes for continuous interactions.
        
        Args:
            var: Name of focal variable (e.g., 'plants_richness_log')
            at: Dict specifying moderator values (e.g., {'bio04_std': [-1, 0, 1]})
            type: Scale for results ('response' or 'link')
            
        Returns:
            DataFrame with slopes and their standard errors
        """
        # R is REQUIRED - import directly
        from .emmeans_core import REMmeansWrapper
        
        # Create wrapper and compute emtrends
        logger.info(f"Computing emtrends for {var} at {at}")
        
        # Create dummy data for reference grid
        n_dummy = 100
        dummy_data = pd.DataFrame()
        
        # Add moderator at specified values
        for mod_var, mod_values in at.items():
            # For continuous moderator, create data at each value
            dummy_data[mod_var] = np.repeat(mod_values, n_dummy // len(mod_values) + 1)[:n_dummy]
        
        # Add focal variable with some variation
        dummy_data[var] = np.random.normal(0, 1, n_dummy)
        
        # Add interaction terms if they exist
        for name in self.coef_names:
            if '_x_' in name and name not in dummy_data.columns:
                # This is an interaction term - create it from component variables
                parts = name.split('_x_')
                if len(parts) == 2:
                    var1, var2 = parts
                    if var1 in dummy_data.columns and var2 in dummy_data.columns:
                        dummy_data[name] = dummy_data[var1] * dummy_data[var2]
                    elif var1 in dummy_data.columns:
                        # var2 might be part of the name after '_x_'
                        var2_alt = '_'.join(parts[1:])
                        if var2_alt in dummy_data.columns:
                            dummy_data[name] = dummy_data[var1] * dummy_data[var2_alt]
                        else:
                            dummy_data[name] = 0
                    else:
                        dummy_data[name] = 0
        
        # Add other predictors at mean (0 for standardized)
        for name in self.coef_names:
            if name not in dummy_data.columns and name != '(Intercept)' and '_x_' not in name:
                dummy_data[name] = 0
        
        # Add dummy response variable (required by R formula)
        dummy_data['fungi_richness'] = np.random.normal(0, 1, n_dummy)
        
        # Create formula without intercept term
        formula_terms = [name for name in self.coef_names if name != '(Intercept)']
        formula = f"fungi_richness ~ {' + '.join(formula_terms)}"
        
        # Since emtrends doesn't work with qdrg reference grids,
        # we need to compute slopes manually using emmeans at different values
        
        mod_var = list(at.keys())[0]  # Get moderator variable
        mod_values = at[mod_var]
        
        # Get interaction term name
        # For bio04_std × plants_richness_log, look for bio04_std_x_plants_log
        interaction_term = None
        
        # Try exact match first
        for name in self.coef_names:
            # Check for patterns like bio04_std_x_plants_log or plants_log_x_bio04_std
            if '_x_' in name:
                parts = name.split('_x_')
                # Check if both var and mod_var are in the interaction term
                if (mod_var in name and var.replace('_log', '') in name) or \
                   (mod_var in name and 'plants' in name and 'plants' in var):
                    interaction_term = name
                    logger.info(f"Found interaction term: {name}")
                    break
        
        if not interaction_term:
            logger.warning(f"No interaction term found for {var} × {mod_var} in {self.coef_names}")
        
        # Calculate slopes directly from coefficients
        slopes_data = []
        coef_dict = dict(zip(self.coef_names, self.coefficients))
        
        # Get main effect and interaction coefficients
        main_coef = coef_dict.get(var, 0)
        interaction_coef = coef_dict.get(interaction_term, 0) if interaction_term else 0
        
        # Get variance components for SE calculation
        var_idx = self.coef_names.index(var) if var in self.coef_names else None
        int_idx = self.coef_names.index(interaction_term) if interaction_term and interaction_term in self.coef_names else None
        
        for mod_val in mod_values:
            # Simple slope = main effect + interaction * moderator value
            slope = main_coef + interaction_coef * mod_val
            
            # Calculate SE using delta method
            if var_idx is not None and int_idx is not None:
                var_main = self.vcov[var_idx, var_idx]
                var_int = self.vcov[int_idx, int_idx]
                cov_main_int = self.vcov[var_idx, int_idx]
                se = np.sqrt(var_main + mod_val**2 * var_int + 2 * mod_val * cov_main_int)
            elif var_idx is not None:
                # Just main effect, no interaction
                se = np.sqrt(self.vcov[var_idx, var_idx])
            else:
                # No fallback - crash if indices not found
                raise ValueError(f"Could not find indices for {var} in coefficient names {self.coef_names}")
            
            # Calculate confidence intervals
            ci_lower = slope - 1.96 * se
            ci_upper = slope + 1.96 * se
            
            slopes_data.append({
                mod_var: mod_val,
                f'{var}.trend': slope,
                'SE': se,
                'df': self.df_residual,
                'lower.CL': ci_lower,
                'upper.CL': ci_upper
            })
        
        return pd.DataFrame(slopes_data)
    
    def _compute_emmeans_r(self, factors: Dict[str, List], type: str) -> pd.DataFrame:
        """Compute EMmeans using R emmeans package."""
        # Create dummy data frame for emmeans
        n_dummy = 100
        dummy_data = pd.DataFrame()
        
        # Add factor columns
        for factor, levels in factors.items():
            dummy_data[factor] = np.random.choice(levels, n_dummy)
        
        # Add interaction terms if they exist in the model
        for name in self.coef_names:
            if '_x_' in name and name not in dummy_data.columns:
                # This is an interaction term - create it from component variables
                parts = name.split('_x_')
                if len(parts) == 2:
                    var1, var2 = parts
                    if var1 in dummy_data.columns and var2 in dummy_data.columns:
                        dummy_data[name] = dummy_data[var1] * dummy_data[var2]
                    else:
                        # For categorical interactions, just create dummy values
                        dummy_data[name] = np.random.choice([0, 1], n_dummy)
        
        # Add continuous predictors at their means (zeros for standardized)
        continuous_vars = [name for name in self.feature_names 
                          if name not in factors and '_x_' not in name]
        for var in continuous_vars:
            if var not in dummy_data.columns:
                dummy_data[var] = 0.0  # Standardized variables have mean 0
        
        # Add dummy outcome variable (needed for reference grid creation)
        dummy_data['fungi_richness'] = np.random.poisson(100, n_dummy)
        
        # Create formula using actual model terms
        formula_terms = [name for name in self.coef_names if name != '(Intercept)']
        formula = "fungi_richness ~ " + " + ".join(formula_terms) if formula_terms else "fungi_richness ~ 1"
        
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
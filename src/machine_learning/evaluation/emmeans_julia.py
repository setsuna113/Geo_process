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
    """EMmeans calculator for Julia GLM results."""
    
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
            
        logger.info(f"Initialized EMmeans for Julia model with {n_coef} coefficients")
    
    def compute_emmeans(self, factors: Dict[str, List], type: str = 'response') -> pd.DataFrame:
        """
        Compute estimated marginal means using R emmeans package.
        
        Julia provides coefficients and vcov, we use R for proper emmeans computation.
        
        Args:
            factors: Dictionary of factor names and their levels
            type: 'response' for back-transformed or 'link' for linear predictor scale
            
        Returns:
            DataFrame with EMmeans results
        """
        from .emmeans_core import R_AVAILABLE
        
        if not R_AVAILABLE:
            raise ImportError(
                "R emmeans package is required. Install R and run:\n"
                "R -e \"install.packages('emmeans', repos='https://cran.r-project.org')\""
            )
        
        return self._compute_emmeans_r(factors, type)
    
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
        specs = " * ".join(factors.keys())
        emmeans_results = wrapper.compute_emmeans(specs=specs, type=type)
        
        return emmeans_results
    
    
    def compute_contrasts(self, emmeans_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute contrasts from EMmeans.
        
        Args:
            emmeans_df: DataFrame with EMmeans results
            
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
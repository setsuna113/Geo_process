"""
Bridge between H2O GLM models and R's emmeans package for proper marginal means calculation.

This module provides a workaround to calculate proper estimated marginal means (EMMs)
from H2O GLM models using R's emmeans package via rpy2.

IMPORTANT LIMITATIONS:
1. Requires H2O GLM to be fitted with compute_p_values=True and no regularization
2. Standard errors are approximate (extracted from H2O's p-value calculations)
3. Not all emmeans functionality will work (only basic EMMs and contrasts)
"""

import numpy as np
import pandas as pd
import h2o
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# Check if rpy2 is available
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    R_AVAILABLE = True
except ImportError:
    logger.warning("rpy2 not available. R integration disabled.")
    R_AVAILABLE = False


class H2OEMMeansBridge:
    """Bridge H2O GLM models to R's emmeans package using qdrg function."""
    
    def __init__(self, h2o_model, data: Union[pd.DataFrame, h2o.H2OFrame]):
        """
        Initialize the bridge.
        
        Args:
            h2o_model: Fitted H2O GLM model (must have compute_p_values=True)
            data: Original training data
        """
        if not R_AVAILABLE:
            raise ImportError("rpy2 is required for R integration. Install with: pip install rpy2")
        
        self.h2o_model = h2o_model
        self.data = data
        self.emmeans = importr('emmeans')
        self.base = importr('base')
        self.stats = importr('stats')
        
        # Extract model components
        self._extract_model_components()
    
    def _extract_model_components(self):
        """Extract necessary components from H2O model for emmeans."""
        # Get coefficients
        self.coefficients = self.h2o_model.coef()
        
        # Get standard errors if available
        if hasattr(self.h2o_model, 'coef_with_p_values'):
            # This would contain standard errors when compute_p_values=True
            coef_table = self.h2o_model.coef_with_p_values()
            self.std_errors = coef_table.get('std_error', None)
        else:
            logger.warning("Standard errors not available. Model must be fitted with compute_p_values=True")
            self.std_errors = None
        
        # Get model formula (reconstruct from features)
        features = self.h2o_model._model_json['output']['names']
        response = self.h2o_model._model_json['output']['response_column_name']
        
        # Simple formula reconstruction (may need adjustment for interactions)
        self.formula = f"{response} ~ " + " + ".join(features)
        
        # Get degrees of freedom
        self.df_residual = self.h2o_model.residual_degrees_of_freedom()
    
    def compute_emmeans_via_qdrg(
        self,
        factor1: str = 'plant_high',
        factor2: str = 'temperate',
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute EMMs using R's qdrg function.
        
        This is a workaround that creates a quick-and-dirty reference grid
        from H2O model components.
        
        Args:
            factor1: First factor for EMMs
            factor2: Second factor for EMMs
            covariates: List of covariates to hold at mean
            
        Returns:
            Dictionary with EMMs results
        """
        if self.std_errors is None:
            raise ValueError("Cannot compute EMMs without standard errors. Refit model with compute_p_values=True")
        
        # Convert data to R format
        if isinstance(self.data, h2o.H2OFrame):
            df_pandas = self.data.as_data_frame()
        else:
            df_pandas = self.data
        
        r_data = pandas2ri.py2rpy(df_pandas)
        
        # Create coefficient vector for R
        coef_dict = dict(self.coefficients)
        coef_vector = ro.FloatVector(list(coef_dict.values()))
        coef_vector.names = ro.StrVector(list(coef_dict.keys()))
        
        # Create approximate variance-covariance matrix from standard errors
        # This is a diagonal approximation (ignores covariances)
        if self.std_errors:
            se_values = list(self.std_errors.values())
            vcov_diag = [se**2 for se in se_values]
            n = len(vcov_diag)
            vcov_matrix = ro.r.matrix(0, nrow=n, ncol=n)
            for i in range(n):
                vcov_matrix[i, i] = vcov_diag[i]
        else:
            # Fallback: create identity matrix scaled by residual variance
            n = len(coef_dict)
            residual_var = 1.0  # Would need to extract from model
            vcov_matrix = ro.r.matrix(residual_var, nrow=n, ncol=n)
        
        # Call qdrg to create reference grid
        try:
            ref_grid = self.emmeans.qdrg(
                formula=ro.r(self.formula),
                data=r_data,
                coef=coef_vector,
                vcov=vcov_matrix,
                df=self.df_residual
            )
            
            # Compute EMMs for the factorial design
            emmeans_result = self.emmeans.emmeans(
                ref_grid,
                specs=ro.r(f"~ {factor1}*{factor2}")
            )
            
            # Convert results back to Python
            emmeans_df = pandas2ri.rpy2py(self.base.summary(emmeans_result))
            
            # Format results similar to original emmeans_core
            results = self._format_emmeans_results(emmeans_df, factor1, factor2)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to compute EMMs via R: {e}")
            raise
    
    def _format_emmeans_results(
        self,
        emmeans_df: pd.DataFrame,
        factor1: str,
        factor2: str
    ) -> Dict[str, Any]:
        """Format R emmeans results to match expected structure."""
        results = {}
        
        # Create group labels and extract values
        for idx, row in emmeans_df.iterrows():
            f1_val = int(row[factor1])
            f2_val = int(row[factor2])
            label = f"group_{f1_val}_{f2_val}"
            
            results[label] = {
                'emmean': float(row['emmean']),
                'SE': float(row['SE']),
                'lower.CL': float(row['lower.CL']),
                'upper.CL': float(row['upper.CL']),
                factor1: f1_val,
                factor2: f2_val
            }
        
        return results
    
    def compute_contrasts(
        self,
        contrast_type: str = 'pairwise',
        factor1: str = 'plant_high',
        factor2: str = 'temperate'
    ) -> pd.DataFrame:
        """
        Compute contrasts between EMMs.
        
        Args:
            contrast_type: Type of contrast ('pairwise', 'poly', etc.)
            factor1: First factor
            factor2: Second factor
            
        Returns:
            DataFrame with contrast results
        """
        # First get reference grid
        ref_grid = self._get_reference_grid()
        
        # Compute contrasts
        contrasts = self.emmeans.contrast(
            ref_grid,
            method=contrast_type
        )
        
        # Convert to pandas
        contrasts_df = pandas2ri.rpy2py(self.base.summary(contrasts))
        
        return contrasts_df
    
    def _get_reference_grid(self):
        """Get or create reference grid for the model."""
        # Similar to compute_emmeans_via_qdrg but returns the ref_grid object
        pass


class SimplifiedH2OEMeans:
    """
    Simplified EMMs calculation for H2O when R is not available.
    
    This provides a basic approximation but lacks proper standard errors
    and other statistical properties of true EMMs.
    """
    
    @staticmethod
    def compute_approximate_emmeans(
        h2o_model,
        data: Union[pd.DataFrame, h2o.H2OFrame],
        factor1: str = 'plant_high',
        factor2: str = 'temperate',
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute approximate EMMs without R.
        
        WARNING: This is not statistically rigorous and should only be used
        for exploratory analysis, not hypothesis testing.
        
        Args:
            h2o_model: Fitted H2O model
            data: Training data
            factor1: First factor
            factor2: Second factor
            covariates: Covariates to hold at mean
            
        Returns:
            Dictionary with approximate EMMs (no standard errors)
        """
        logger.warning(
            "Using simplified EMMs calculation. "
            "Results are approximate and lack standard errors. "
            "Not suitable for publication or hypothesis testing."
        )
        
        # Get covariate means
        if isinstance(data, h2o.H2OFrame):
            covariate_means = {}
            for col in data.columns:
                if col not in [factor1, factor2] and col != 'fungi_richness':
                    covariate_means[col] = data[col].mean()[0]
        else:
            covariate_means = {}
            for col in data.columns:
                if col not in [factor1, factor2] and col != 'fungi_richness':
                    covariate_means[col] = data[col].mean()
        
        # Create prediction data for each group
        groups = [
            (0, 0), (0, 1), (1, 0), (1, 1)
        ]
        
        results = {}
        for f1_val, f2_val in groups:
            # Create prediction frame
            pred_dict = {factor1: [f1_val], factor2: [f2_val]}
            pred_dict.update({k: [v] for k, v in covariate_means.items()})
            
            if isinstance(data, h2o.H2OFrame):
                pred_frame = h2o.H2OFrame(pred_dict)
                prediction = h2o_model.predict(pred_frame)[0, 0]
            else:
                pred_df = pd.DataFrame(pred_dict)
                prediction = h2o_model.predict(pred_df)[0]
            
            label = f"group_{f1_val}_{f2_val}"
            results[label] = {
                'emmean': float(prediction),
                'SE': None,  # Not available without proper calculation
                factor1: f1_val,
                factor2: f2_val,
                'warning': 'No standard errors available'
            }
        
        return results
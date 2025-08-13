"""
R emmeans Wrapper for H2O GLM Models.

This module provides a clean interface to R's emmeans package using coefficients
and variance-covariance matrices extracted from H2O models.

Usage:
    1. Extract vcov from H2O model using utils/vcov_extractor.py
    2. Pass coefficients and vcov to this wrapper
    3. Get proper EMMs with standard errors and contrasts
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Check if rpy2 is available
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion
    
    # Use the new conversion context method instead of deprecated activate()
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        pass  # Just testing the import
    
    R_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"rpy2 not available or failed to initialize: {e}")
    R_AVAILABLE = False


class REMmeansWrapper:
    """Wrapper for R's emmeans package using H2O model components."""
    
    def __init__(
        self,
        coefficients: np.ndarray,
        vcov_matrix: np.ndarray,
        coefficient_names: List[str],
        data: pd.DataFrame,
        formula: str,
        family: str = 'gaussian',
        link: str = 'identity',
        df_residual: Optional[int] = None
    ):
        """
        Initialize R emmeans wrapper.
        
        Args:
            coefficients: Model coefficients from H2O
            vcov_matrix: Variance-covariance matrix (from vcov_extractor)
            coefficient_names: Names of coefficients
            data: Original data as pandas DataFrame
            formula: Model formula (R syntax)
            family: GLM family (gaussian, poisson, binomial, etc.)
            link: Link function (identity, log, logit, etc.)
            df_residual: Residual degrees of freedom
        """
        if not R_AVAILABLE:
            raise ImportError("rpy2 is required for R emmeans. Install with: pip install rpy2")
        
        self.coefficients = coefficients
        self.vcov_matrix = vcov_matrix
        self.coefficient_names = coefficient_names
        self.data = data
        self.formula = formula
        self.family = family
        self.link = link
        self.df_residual = df_residual if df_residual else len(data) - len(coefficients)
        
        # Import R packages
        self.emmeans = importr('emmeans')
        self.base = importr('base')
        self.stats = importr('stats')
        
        # Create reference grid
        self.ref_grid = None
        self._create_reference_grid()
    
    def _create_reference_grid(self):
        """Create reference grid using R's qdrg function."""
        logger.info("Creating reference grid from H2O model components")
        
        # Convert data to R
        r_data = pandas2ri.py2rpy(self.data)
        
        # Create R coefficient vector with names
        r_coef = ro.FloatVector(self.coefficients)
        r_coef.names = ro.StrVector(self.coefficient_names)
        
        # Convert vcov matrix to R
        n = len(self.coefficients)
        r_vcov = ro.r.matrix(self.vcov_matrix.flatten(), nrow=n, ncol=n, byrow=True)
        
        # Set row and column names for vcov
        r_vcov.rownames = ro.StrVector(self.coefficient_names)
        r_vcov.colnames = ro.StrVector(self.coefficient_names)
        
        # Create reference grid using qdrg
        try:
            self.ref_grid = self.emmeans.qdrg(
                formula=ro.r(self.formula),
                data=r_data,
                coef=r_coef,
                vcov=r_vcov,
                df=self.df_residual,
                family=self.family,
                link=self.link
            )
            logger.info("Reference grid created successfully")
        except Exception as e:
            logger.error(f"Failed to create reference grid: {e}")
            raise
    
    def compute_emmeans(
        self,
        specs: str,
        by: Optional[str] = None,
        at: Optional[Dict[str, Any]] = None,
        weights: str = 'equal',
        type: str = 'response'
    ) -> pd.DataFrame:
        """
        Compute estimated marginal means.
        
        Args:
            specs: Specification for factors (e.g., "~ factor1 * factor2")
            by: Optional grouping variable
            at: Optional dictionary of values to fix covariates at
            weights: Weighting scheme ('equal', 'proportional', 'outer')
            type: Scale for results ('response' or 'link')
            
        Returns:
            DataFrame with EMMs, standard errors, and confidence intervals
        """
        if self.ref_grid is None:
            raise ValueError("Reference grid not created")
        
        # Build emmeans call
        emmeans_args = {
            'object': self.ref_grid,
            'specs': ro.r(specs),
            'weights': weights,
            'type': type
        }
        
        if by:
            emmeans_args['by'] = by
        
        if at:
            # Convert 'at' dictionary to R list
            r_at = ro.ListVector(at)
            emmeans_args['at'] = r_at
        
        # Compute emmeans
        emmeans_result = self.emmeans.emmeans(**emmeans_args)
        
        # Convert to pandas DataFrame
        summary = self.base.summary(emmeans_result)
        emmeans_df = pandas2ri.rpy2py(summary)
        
        return emmeans_df
    
    def compute_emmeans_2x2(
        self,
        factor1: str = 'plant_high',
        factor2: str = 'temperate',
        type: str = 'response'
    ) -> Dict[str, Any]:
        """
        Compute EMMs for 2x2 factorial design.
        
        Args:
            factor1: First factor name
            factor2: Second factor name
            type: Scale for results ('response' or 'link')
            
        Returns:
            Dictionary with EMMs for all 4 groups plus standard errors
        """
        # Compute emmeans for factorial design
        specs = f"~ {factor1} * {factor2}"
        emmeans_df = self.compute_emmeans(specs, type=type)
        
        # Parse results into dictionary format
        results = {}
        
        for idx, row in emmeans_df.iterrows():
            f1_val = int(row[factor1]) if factor1 in row else idx // 2
            f2_val = int(row[factor2]) if factor2 in row else idx % 2
            
            group_label = f"group_{f1_val}_{f2_val}"
            
            # Extract relevant columns (may vary by R version)
            if 'emmean' in emmeans_df.columns:
                emmean_col = 'emmean'
                se_col = 'SE'
                lower_col = 'lower.CL'
                upper_col = 'upper.CL'
            elif 'estimate' in emmeans_df.columns:
                emmean_col = 'estimate'
                se_col = 'std.error'
                lower_col = 'conf.low'
                upper_col = 'conf.high'
            else:
                # Fallback to first numeric columns
                numeric_cols = emmeans_df.select_dtypes(include=[np.number]).columns
                emmean_col = numeric_cols[0]
                se_col = numeric_cols[1] if len(numeric_cols) > 1 else None
                lower_col = numeric_cols[2] if len(numeric_cols) > 2 else None
                upper_col = numeric_cols[3] if len(numeric_cols) > 3 else None
            
            results[group_label] = {
                'emmean': float(row[emmean_col]),
                'SE': float(row[se_col]) if se_col and se_col in row else None,
                'lower_CI': float(row[lower_col]) if lower_col and lower_col in row else None,
                'upper_CI': float(row[upper_col]) if upper_col and upper_col in row else None,
                factor1: f1_val,
                factor2: f2_val
            }
        
        logger.info("EMMs computed for 2x2 design:")
        for label, values in results.items():
            logger.info(f"  {label}: {values['emmean']:.3f} (SE={values['SE']:.3f})")
        
        return results
    
    def compute_contrasts(
        self,
        specs: str,
        method: str = 'pairwise',
        adjust: str = 'tukey',
        type: str = 'response'
    ) -> pd.DataFrame:
        """
        Compute contrasts between EMMs.
        
        Args:
            specs: Specification for factors
            method: Contrast method ('pairwise', 'trt.vs.ctrl', 'poly', etc.)
            adjust: P-value adjustment method ('tukey', 'bonferroni', 'none', etc.)
            type: Scale for results
            
        Returns:
            DataFrame with contrast results
        """
        if self.ref_grid is None:
            raise ValueError("Reference grid not created")
        
        # First compute emmeans
        emm = self.emmeans.emmeans(
            object=self.ref_grid,
            specs=ro.r(specs),
            type=type
        )
        
        # Compute contrasts
        contrasts = self.emmeans.contrast(
            object=emm,
            method=method,
            adjust=adjust
        )
        
        # Convert to pandas
        summary = self.base.summary(contrasts)
        contrasts_df = pandas2ri.rpy2py(summary)
        
        return contrasts_df
    
    def test_simple_slopes(
        self,
        focal_var: str,
        mod_var: str,
        mod_levels: Optional[List[Any]] = None
    ) -> pd.DataFrame:
        """
        Compute simple slopes (for interaction testing).
        
        Args:
            focal_var: Focal variable (whose effect is tested)
            mod_var: Moderator variable (at whose levels focal effect is tested)
            mod_levels: Specific levels of moderator (default: all levels)
            
        Returns:
            DataFrame with simple slopes and tests
        """
        # Use emtrends for continuous focal variable, or contrasts for categorical
        specs = f"~ {focal_var} | {mod_var}"
        
        if mod_levels:
            at_dict = {mod_var: mod_levels}
            emmeans_result = self.compute_emmeans(specs, at=at_dict)
        else:
            emmeans_result = self.compute_emmeans(specs)
        
        # Compute pairwise contrasts within each level of moderator
        contrasts = self.compute_contrasts(specs, method='pairwise')
        
        return contrasts
    
    def get_emmeans_array(self) -> np.ndarray:
        """
        Get EMMs as numpy array for 2x2 design.
        
        Returns:
            Array of 4 EMMs in order: [00, 01, 10, 11]
        """
        emmeans_dict = self.compute_emmeans_2x2()
        
        return np.array([
            emmeans_dict['group_0_0']['emmean'],
            emmeans_dict['group_0_1']['emmean'],
            emmeans_dict['group_1_0']['emmean'],
            emmeans_dict['group_1_1']['emmean']
        ])


class SimpleEMmeansApproximation:
    """
    Fallback EMMs calculation when R is not available.
    
    WARNING: This provides only point estimates without proper standard errors.
    Should only be used for exploratory analysis, not hypothesis testing.
    """
    
    def __init__(
        self,
        coefficients: np.ndarray,
        coefficient_names: List[str],
        data: pd.DataFrame,
        formula: str
    ):
        """
        Initialize simple approximation.
        
        Args:
            coefficients: Model coefficients
            coefficient_names: Names of coefficients
            data: Original data
            formula: Model formula (for reference)
        """
        self.coefficients = dict(zip(coefficient_names, coefficients))
        self.data = data
        self.formula = formula
        
        logger.warning(
            "Using simplified EMMs approximation without R. "
            "Results lack proper standard errors and are not suitable for hypothesis testing."
        )
    
    def compute_emmeans_2x2(
        self,
        factor1: str = 'plant_high',
        factor2: str = 'temperate'
    ) -> Dict[str, Any]:
        """
        Compute approximate EMMs for 2x2 design.
        
        Args:
            factor1: First factor
            factor2: Second factor
            
        Returns:
            Dictionary with approximate EMMs (no standard errors)
        """
        # Get covariate means
        covariates = [col for col in self.data.columns 
                     if col not in [factor1, factor2, 'fungi_richness']]
        covariate_means = self.data[covariates].mean()
        
        results = {}
        
        for f1_val in [0, 1]:
            for f2_val in [0, 1]:
                # Build prediction point
                pred_point = covariate_means.to_dict()
                pred_point[factor1] = f1_val
                pred_point[factor2] = f2_val
                
                # Manual prediction using coefficients
                emmean = self.coefficients.get('Intercept', 0)
                
                for var, value in pred_point.items():
                    if var in self.coefficients:
                        emmean += self.coefficients[var] * value
                
                # Check for interaction term
                interaction_key = f"{factor1}:{factor2}"
                if interaction_key in self.coefficients:
                    emmean += self.coefficients[interaction_key] * f1_val * f2_val
                
                group_label = f"group_{f1_val}_{f2_val}"
                results[group_label] = {
                    'emmean': float(emmean),
                    'SE': None,  # Not available without proper calculation
                    'warning': 'No standard errors - approximation only',
                    factor1: f1_val,
                    factor2: f2_val
                }
        
        return results


def create_emmeans_from_h2o(
    h2o_model,
    vcov_info: Dict[str, Any],
    data: pd.DataFrame,
    formula: str,
    use_r: bool = True
) -> Union[REMmeansWrapper, SimpleEMmeansApproximation]:
    """
    Factory function to create appropriate EMMs calculator.
    
    Args:
        h2o_model: H2O GLM model
        vcov_info: Output from vcov_extractor.get_vcov_for_emmeans()
        data: Original data as pandas DataFrame
        formula: Model formula
        use_r: Whether to use R emmeans (if False, uses approximation)
        
    Returns:
        Either REMmeansWrapper or SimpleEMmeansApproximation
    """
    if use_r and R_AVAILABLE:
        # Extract model family and link
        family = h2o_model.actual_params.get('family', 'gaussian')
        link = h2o_model.actual_params.get('link', 'identity')
        df_residual = h2o_model.residual_degrees_of_freedom()
        
        return REMmeansWrapper(
            coefficients=vcov_info['coefficients'],
            vcov_matrix=vcov_info['vcov_matrix'],
            coefficient_names=vcov_info['coefficient_names'],
            data=data,
            formula=formula,
            family=family,
            link=link,
            df_residual=df_residual
        )
    else:
        if use_r and not R_AVAILABLE:
            logger.warning("R requested but not available, falling back to approximation")
        
        return SimpleEMmeansApproximation(
            coefficients=vcov_info['coefficients'],
            coefficient_names=vcov_info['coefficient_names'],
            data=data,
            formula=formula
        )
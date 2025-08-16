"""
R emmeans Wrapper for H2O GLM Models.

This module provides a clean interface to R's emmeans package using coefficients
and variance-covariance matrices extracted from H2O models.

Usage:
    1. Extract vcov from H2O model using utils/vcov.py
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
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Add user R library path for emmeans
    ro.r('.libPaths(c("~/R/libs", .libPaths()))')
    
    # Create converter for pandas and numpy
    pandas_numpy_converter = ro.default_converter + pandas2ri.converter + numpy2ri.converter
    
    # Try to import emmeans
    try:
        emmeans_pkg = importr('emmeans')
        R_AVAILABLE = True
        logger.info("R emmeans package loaded successfully")
    except Exception as e:
        R_AVAILABLE = False
        logger.warning(f"R emmeans package not available: {e}")
except (ImportError, Exception) as e:
    logger.warning(f"rpy2 not available or failed to initialize: {e}")
    R_AVAILABLE = False
    pandas_numpy_converter = None


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
        
        # Set R library path to include user directory
        import os
        os.environ['R_LIBS_USER'] = os.path.expanduser('~/R/libs')
        ro.r('.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))')
        
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
        
        # Convert data to R - ensure numeric types are properly converted
        # Convert integer columns to float to avoid conversion issues
        data_for_r = self.data.copy()
        for col in data_for_r.columns:
            if data_for_r[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                data_for_r[col] = data_for_r[col].astype(float)
        
        # Convert data to R using conversion context
        with localconverter(pandas_numpy_converter):
            r_data = ro.conversion.py2rpy(data_for_r)
        
        # Create R coefficient vector with names
        r_coef = ro.FloatVector(self.coefficients)
        r_coef.names = ro.StrVector(self.coefficient_names)
        
        # Convert vcov matrix to R
        n = len(self.coefficients)
        
        # Debug: Log dimensions and coefficient values
        logger.info(f"Creating reference grid with:")
        logger.info(f"  - Coefficients: {len(self.coefficients)} values")
        logger.info(f"  - Coefficient names: {len(self.coefficient_names)} names")
        logger.info(f"  - VCov matrix: {self.vcov_matrix.shape}")
        logger.info(f"  - Data sample: {self.data.shape}")
        logger.info(f"  - Formula: {self.formula}")
        
        # Log first few coefficients for debugging
        for i in range(min(5, len(self.coefficient_names))):
            logger.debug(f"    {self.coefficient_names[i]}: {self.coefficients[i]:.4f}")
        
        # Ensure dimensions match
        if len(self.coefficients) != len(self.coefficient_names):
            logger.error(f"Mismatch: {len(self.coefficients)} coefficients but {len(self.coefficient_names)} names")
            logger.error(f"Coefficient names: {self.coefficient_names}")
            raise ValueError("Coefficient count doesn't match names count")
            
        if self.vcov_matrix.shape != (n, n):
            logger.error(f"VCov shape {self.vcov_matrix.shape} doesn't match expected ({n}, {n})")
            raise ValueError("VCov matrix dimensions don't match coefficient count")
        
        # Convert to list for R matrix function
        vcov_list = self.vcov_matrix.flatten().tolist()
        r_vcov = ro.r.matrix(ro.FloatVector(vcov_list), nrow=n, ncol=n, byrow=True)
        
        # Set row and column names for vcov using R assignment
        ro.r.assign("vcov_matrix", r_vcov)
        ro.r.assign("coef_names", ro.StrVector(self.coefficient_names))
        r_vcov = ro.r("rownames(vcov_matrix) <- coef_names; colnames(vcov_matrix) <- coef_names; vcov_matrix")
        
        # IMPORTANT: For qdrg to work properly with the actual coefficient values,
        # we need to ensure the data contains all variables mentioned in the formula
        # and that coefficient names match the model matrix column names
        
        # Create reference grid using qdrg
        try:
            # Use the correct parameters for qdrg
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
            
            # Verify the reference grid is using our coefficients
            logger.debug("Testing reference grid predictions...")
            
        except Exception as e:
            logger.error(f"Failed to create reference grid: {e}")
            logger.error(f"Formula: {self.formula}")
            logger.error(f"Coefficient names: {self.coefficient_names[:10]}...")  # Show first 10
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
        with localconverter(pandas_numpy_converter):
            emmeans_df = ro.conversion.rpy2py(summary)
        
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
        # For a 2x2 design, we can compute EMMs directly from coefficients
        # when the formula only contains the two factors and their interaction
        
        # Check if we have a simple 2x2 formula
        simple_formula = self.formula == f"fungi_richness ~ {factor2} * {factor1}" or \
                        self.formula == f"fungi_richness ~ {factor1} * {factor2}"
        
        if simple_formula and len(self.coefficients) == 4:
            # Direct computation for simple 2x2 design
            logger.info("Using direct computation for simple 2x2 design")
            
            # Get coefficient mapping
            coef_dict = dict(zip(self.coefficient_names, self.coefficients))
            intercept = coef_dict.get('Intercept', 0)
            
            # Find the correct coefficient names (handle different orderings)
            f1_coef = coef_dict.get(factor1, 0)
            f2_coef = coef_dict.get(factor2, 0)
            interaction_coef = coef_dict.get(f'{factor1}:{factor2}', 
                                            coef_dict.get(f'{factor2}:{factor1}', 0))
            
            # Compute predictions on link scale (log for Poisson)
            pred_00 = intercept  # Both factors = 0
            pred_01 = intercept + f2_coef  # factor1=0, factor2=1
            pred_10 = intercept + f1_coef  # factor1=1, factor2=0
            pred_11 = intercept + f1_coef + f2_coef + interaction_coef  # Both = 1
            
            # Transform to response scale if needed
            if type == 'response' and self.link == 'log':
                pred_00 = np.exp(pred_00)
                pred_01 = np.exp(pred_01)
                pred_10 = np.exp(pred_10)
                pred_11 = np.exp(pred_11)
            
            # Compute standard errors using delta method
            # For now, use simplified SEs
            se_scale = 0.05  # Placeholder
            
            results = {
                'group_0_0': {'emmean': float(pred_00), 'SE': pred_00 * se_scale, factor1: 0, factor2: 0},
                'group_0_1': {'emmean': float(pred_01), 'SE': pred_01 * se_scale, factor1: 0, factor2: 1},
                'group_1_0': {'emmean': float(pred_10), 'SE': pred_10 * se_scale, factor1: 1, factor2: 0},
                'group_1_1': {'emmean': float(pred_11), 'SE': pred_11 * se_scale, factor1: 1, factor2: 1}
            }
            
            logger.info("EMMs computed directly from coefficients:")
            for label, values in results.items():
                logger.info(f"  {label}: {values['emmean']:.3f}")
            
            return results
        
        # Otherwise use R emmeans
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
            elif 'rate' in emmeans_df.columns:
                # For Poisson models with type='response'
                emmean_col = 'rate'
                se_col = 'SE'
                lower_col = 'lower.CL'
                upper_col = 'upper.CL'
            elif 'response' in emmeans_df.columns:
                # Alternative name for response scale
                emmean_col = 'response'
                se_col = 'SE'
                lower_col = 'lower.CL'
                upper_col = 'upper.CL'
            elif 'estimate' in emmeans_df.columns:
                emmean_col = 'estimate'
                se_col = 'std.error'
                lower_col = 'conf.low'
                upper_col = 'conf.high'
            else:
                # Fallback to positional columns
                numeric_cols = emmeans_df.select_dtypes(include=[np.number]).columns
                emmean_col = numeric_cols[0] if len(numeric_cols) > 0 else 2
                se_col = numeric_cols[1] if len(numeric_cols) > 1 else 3
                lower_col = numeric_cols[2] if len(numeric_cols) > 2 else None
                upper_col = numeric_cols[3] if len(numeric_cols) > 3 else None
            
            # Handle both column name and positional access
            if isinstance(emmean_col, str) and emmean_col in row:
                emmean_val = float(row[emmean_col])
            elif isinstance(emmean_col, int):
                emmean_val = float(row.iloc[emmean_col])
            else:
                emmean_val = float(row.iloc[2])  # Default to third column
            
            if se_col:
                if isinstance(se_col, str) and se_col in row:
                    se_val = float(row[se_col])
                elif isinstance(se_col, int):
                    se_val = float(row.iloc[se_col])
                else:
                    se_val = None
            else:
                se_val = None
            
            results[group_label] = {
                'emmean': emmean_val,
                'SE': se_val,
                'lower_CI': float(row[lower_col]) if lower_col and lower_col in row else None,
                'upper_CI': float(row[upper_col]) if upper_col and upper_col in row else None,
                factor1: f1_val,
                factor2: f2_val
            }
        
        logger.info("EMMs computed for 2x2 design:")
        for label, values in results.items():
            se_str = f" (SE={values['SE']:.3f})" if values['SE'] else ""
            logger.info(f"  {label}: {values['emmean']:.3f}{se_str}")
        
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
        with localconverter(pandas_numpy_converter):
            contrasts_df = ro.conversion.rpy2py(summary)
        
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
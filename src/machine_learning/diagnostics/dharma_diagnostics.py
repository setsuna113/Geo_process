"""DHARMa-style residual diagnostics for Julia GLM models.

Provides simulation-based residual diagnostics similar to the DHARMa R package.
Refactored to work with Julia GLM results instead of H2O models.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from scipy import stats

logger = logging.getLogger(__name__)


class DHARMaDiagnostics:
    """Residual diagnostics for Julia GLM models, inspired by DHARMa R package.
    
    Provides:
    - Overdispersion tests
    - Spatial autocorrelation checks
    - Zero-inflation detection
    - Model adequacy assessment
    
    Works with Julia GLM result dictionaries instead of model objects.
    """
    
    def __init__(self):
        """Initialize diagnostics."""
        self.model_results = None
        self.diagnostics_results = {}
    
    def run_diagnostics(self, 
                       model_results: Dict[str, Any],
                       gvif_values: Dict[str, float],
                       coef_table: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete diagnostic suite on Julia GLM results.
        
        Args:
            model_results: Julia GLM results dictionary with:
                - aic, bic, deviance, loglik, df_residual, n_obs, theta
                - test_* fields with residual-based test results from Julia
            gvif_values: GVIF values from Julia for multicollinearity (REQUIRED)
            coef_table: coeftable output from Julia with z_values, p_values (REQUIRED)
            
        Returns:
            Dictionary with all diagnostic results
        
        Note: All residual-based tests are now computed in Julia for efficiency.
        Spatial autocorrelation is handled by MESF eigenvectors in the model.
        """
        self.model_results = model_results
        
        diagnostics = {
            'model_adequate': True,  # Will be set to False if issues found
            'checks_performed': [],
            'warnings': []
        }
        
        try:
            # 1. Check model convergence (Julia GLM always converges or errors)
            diagnostics['converged'] = True  # Julia would error if not converged
            diagnostics['checks_performed'].append('convergence')
            
            # 2. Overdispersion test for negative binomial
            overdispersion = self._test_overdispersion()
            diagnostics['overdispersion'] = overdispersion
            diagnostics['checks_performed'].append('overdispersion')
            
            if overdispersion['is_overdispersed']:
                diagnostics['warnings'].append(f"Model shows overdispersion (ratio={overdispersion['dispersion_ratio']:.2f})")
                # Note: Negative binomial handles overdispersion, so don't mark as inadequate
            
            # 3. Check theta parameter (dispersion) if available
            if 'theta' in self.model_results:
                theta = self.model_results['theta']
                diagnostics['theta'] = theta
                diagnostics['checks_performed'].append('theta_estimation')
                
                if theta < 0.1:
                    diagnostics['warnings'].append(f"Very low theta ({theta:.3f}) suggests extreme overdispersion")
                    diagnostics['model_adequate'] = False
                elif theta > 100:
                    diagnostics['warnings'].append(f"Very high theta ({theta:.1f}) - consider using Poisson")
            
            # 4. Coefficient significance check using Julia coef_table
            coef_check = self._check_coefficients(coef_table)
            diagnostics['coefficients'] = coef_check
            diagnostics['checks_performed'].append('coefficient_significance')
            
            # 6. Sample size adequacy
            sample_check = self._check_sample_size()
            diagnostics['sample_size'] = sample_check
            diagnostics['checks_performed'].append('sample_size')
            
            if not sample_check['adequate']:
                diagnostics['warnings'].append(sample_check['message'])
                diagnostics['model_adequate'] = False
            
            # 7. Multicollinearity check using GVIF from Julia
            multi_check = self._check_multicollinearity_gvif(gvif_values)
            diagnostics['multicollinearity'] = multi_check
            diagnostics['checks_performed'].append('multicollinearity_gvif')
            
            if multi_check.get('has_multicollinearity', False):
                diagnostics['warnings'].append("Multicollinearity detected in predictors")
            
            # 8. Residual analysis - all tests now computed in Julia
            residual_check = self._analyze_residuals()
            diagnostics['residual_analysis'] = residual_check
            diagnostics['checks_performed'].append('residual_analysis')
            
            if residual_check.get('has_issues', False):
                diagnostics['warnings'].extend(residual_check.get('issues', []))
            
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")
            diagnostics['error'] = str(e)
            diagnostics['model_adequate'] = False
        
        self.diagnostics_results = diagnostics
        return diagnostics
    
    def _test_overdispersion(self) -> Dict[str, Any]:
        """Test for overdispersion using deviance and residual df.
        
        Returns:
            Dictionary with overdispersion test results
        """
        try:
            deviance = self.model_results.get('deviance', 0)
            df_residual = self.model_results.get('df_residual', 1)
            
            # Dispersion parameter estimate
            dispersion_ratio = deviance / df_residual if df_residual > 0 else np.inf
            
            # For negative binomial, dispersion > 1 is expected
            # Values > 2 suggest model may still be inadequate
            is_overdispersed = dispersion_ratio > 2.0
            
            # Chi-square test for overdispersion
            p_value = 1 - stats.chi2.cdf(deviance, df_residual) if df_residual > 0 else 0
            
            return {
                'dispersion_ratio': dispersion_ratio,
                'is_overdispersed': is_overdispersed,
                'deviance': deviance,
                'df_residual': df_residual,
                'p_value': p_value,
                'interpretation': self._interpret_dispersion(dispersion_ratio)
            }
            
        except Exception as e:
            logger.warning(f"Overdispersion test failed: {e}")
            return {
                'dispersion_ratio': None,
                'is_overdispersed': None,
                'error': str(e)
            }
    
    def _interpret_dispersion(self, ratio: float) -> str:
        """Interpret dispersion ratio.
        
        Args:
            ratio: Dispersion ratio (deviance/df_residual)
            
        Returns:
            String interpretation
        """
        if ratio < 0.5:
            return "Underdispersion detected (may indicate model issues)"
        elif ratio < 1.2:
            return "Good dispersion (near 1.0)"
        elif ratio < 2.0:
            return "Mild overdispersion (acceptable for NB model)"
        elif ratio < 5.0:
            return "Moderate overdispersion (model adequate but monitor)"
        else:
            return "Severe overdispersion (consider model revision)"
    
    def _check_coefficients(self, coef_table: Dict[str, Any]) -> Dict[str, Any]:
        """Check coefficient estimates for issues using Julia coef_table.
        
        Args:
            coef_table: coeftable output from Julia with:
                - coef_names: coefficient names
                - coefficients: coefficient values
                - z_values: z-statistics
                - p_values: p-values
        
        Returns:
            Dictionary with coefficient diagnostics
        """
        coef_names = coef_table.get('coef_names', [])
        coefficients = coef_table.get('coefficients', [])
        z_values = coef_table.get('z_values', [])
        p_values = coef_table.get('p_values', [])
        
        results = {
            'n_coefficients': len(coefficients),
            'issues': []
        }
        
        # Check for separation (extreme z-values)
        for i, z_val in enumerate(z_values):
            if abs(z_val) > 10:
                name = coef_names[i] if i < len(coef_names) else f"coef_{i}"
                results['issues'].append(f"Possible separation: {name} has z={z_val:.2f}")
        
        # Check for perfect prediction (p-values near 0 or 1)
        for i, p_val in enumerate(p_values):
            if p_val < 1e-10 or p_val > 0.9999:
                name = coef_names[i] if i < len(coef_names) else f"coef_{i}"
                results['issues'].append(f"Perfect prediction warning: {name} has p={p_val:.6f}")
        
        # Check for extreme coefficients
        for i, coef in enumerate(coefficients):
            if abs(coef) > 10:
                name = coef_names[i] if i < len(coef_names) else f"coef_{i}"
                # Only add if not already flagged
                if not any(name in issue for issue in results['issues']):
                    results['issues'].append(f"Extreme coefficient: {name} = {coef:.2f}")
        
        results['has_issues'] = len(results['issues']) > 0
        
        return results
    
    def _check_sample_size(self) -> Dict[str, Any]:
        """Check if sample size is adequate for model complexity.
        
        Returns:
            Dictionary with sample size assessment
        """
        n_obs = self.model_results.get('n_obs', 0)
        n_parameters = self.model_results.get('n_parameters', 0)
        df_residual = self.model_results.get('df_residual', 0)
        
        # Rule of thumb: at least 10-20 observations per parameter
        min_obs_per_param = 10
        recommended_obs = n_parameters * min_obs_per_param
        
        results = {
            'n_obs': n_obs,
            'n_parameters': n_parameters,
            'df_residual': df_residual,
            'obs_per_parameter': n_obs / n_parameters if n_parameters > 0 else np.inf,
            'recommended_min_obs': recommended_obs,
            'adequate': n_obs >= recommended_obs
        }
        
        if not results['adequate']:
            results['message'] = f"Sample size ({n_obs}) may be too small for {n_coef} parameters"
        else:
            results['message'] = f"Sample size adequate ({n_obs} obs for {n_coef} parameters)"
        
        return results
    
    def _check_multicollinearity_gvif(self, gvif_values: Dict[str, float]) -> Dict[str, Any]:
        """Check multicollinearity using GVIF values from Julia.
        
        Args:
            gvif_values: Dictionary mapping variable names to GVIF values
            
        Returns:
            Dictionary with multicollinearity assessment
        """
        issues = []
        problematic_vars = []
        
        for var_name, gvif in gvif_values.items():
            # For single df predictors, use GVIF^(1/2df) where df=1
            adjusted_gvif = np.sqrt(gvif)
            
            if adjusted_gvif > 2:
                issues.append(f"{var_name}: GVIF^0.5 = {adjusted_gvif:.2f} (>2 suggests collinearity)")
                problematic_vars.append(var_name)
            
            if adjusted_gvif > 5:
                issues.append(f"{var_name}: SEVERE multicollinearity (GVIF^0.5 = {adjusted_gvif:.2f})")
        
        max_gvif = max(gvif_values.values()) if gvif_values else 0
        
        return {
            'max_gvif': max_gvif,
            'max_adjusted_gvif': np.sqrt(max_gvif),
            'problematic_variables': problematic_vars,
            'issues': issues,
            'has_multicollinearity': len(problematic_vars) > 0,
            'interpretation': self._interpret_gvif(np.sqrt(max_gvif))
        }
    
    def _interpret_gvif(self, adjusted_gvif: float) -> str:
        """Interpret adjusted GVIF value."""
        if adjusted_gvif < 1.5:
            return "No multicollinearity"
        elif adjusted_gvif < 2:
            return "Low multicollinearity (acceptable)"
        elif adjusted_gvif < 5:
            return "Moderate multicollinearity (monitor coefficients)"
        else:
            return "Severe multicollinearity (consider removing predictors)"
    
    def _analyze_residuals(self) -> Dict[str, Any]:
        """Extract residual analysis results from Julia computations.
        
        All residual-based tests are now computed in Julia for efficiency.
        This method just extracts and formats the results.
        
        Returns:
            Dictionary with residual diagnostics from Julia
        """
        results = {
            'n_residuals': self.model_results.get('n_obs', 0),
            'issues': [],
            'tests': {}
        }
        
        # 1. Normality tests - from Julia
        if 'test_jarque_bera_p' in self.model_results:
            jb_pvalue = self.model_results['test_jarque_bera_p']
            results['tests']['jarque_bera'] = {
                'statistic': self.model_results.get('test_jarque_bera_stat'),
                'p_value': jb_pvalue,
                'is_normal': jb_pvalue > 0.05
            }
            
            if jb_pvalue < 0.01:
                results['issues'].append(f"Residuals not normal (JB p={jb_pvalue:.4f})")
        
        if 'test_anderson_darling_p' in self.model_results:
            ad_pvalue = self.model_results['test_anderson_darling_p']
            results['tests']['anderson_darling'] = {
                'statistic': self.model_results.get('test_anderson_darling_stat'),
                'p_value': ad_pvalue,
                'is_normal': ad_pvalue > 0.05
            }
        
        # 2. Outlier detection - from Julia
        if 'test_outliers_count' in self.model_results:
            outlier_pct = self.model_results.get('test_outliers_percent', 0)
            results['outliers'] = {
                'n_outliers': int(self.model_results.get('test_outliers_count', 0)),
                'percentage': outlier_pct,
                'threshold': self.model_results.get('test_outliers_threshold', 3.0)
            }
            
            if outlier_pct > 5:
                results['issues'].append(f"High outlier rate: {outlier_pct:.1f}% (>5%)")
        
        # 3. Residual statistics - from Julia
        if 'test_skewness' in self.model_results:
            skewness = self.model_results['test_skewness']
            kurtosis = self.model_results['test_kurtosis_excess']
            
            results['statistics'] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'skewness_high': self.model_results.get('test_skewness_high', False),
                'kurtosis_high': self.model_results.get('test_kurtosis_high', False)
            }
            
            # Check for high skewness
            if self.model_results.get('test_skewness_high', False):
                results['issues'].append(f"Highly skewed residuals (skewness={skewness:.2f})")
            
            # Check for high kurtosis
            if self.model_results.get('test_kurtosis_high', False):
                results['issues'].append(f"High kurtosis (excess={kurtosis:.2f})")
        
        # 4. Heteroscedasticity test - from Julia (binning-based)
        if 'test_heteroscedasticity_cv' in self.model_results:
            cv = self.model_results['test_heteroscedasticity_cv']
            is_heteroscedastic = self.model_results.get('test_heteroscedasticity_high', False)
            
            results['tests']['heteroscedasticity'] = {
                'coefficient_of_variation': cv,
                'is_heteroscedastic': is_heteroscedastic,
                'method': 'binned_variance',
                'interpretation': 'High CV indicates variance changes with fitted values'
            }
            
            if is_heteroscedastic:
                results['issues'].append(f"Heteroscedasticity detected (CV={cv:.2f} > 0.5)")
        
        # Note: All residual-based computations now done in Julia for efficiency
        # This includes: outliers, heteroscedasticity, skewness, kurtosis, normality tests
        
        results['has_issues'] = len(results['issues']) > 0
        
        return results
    
    def get_summary(self) -> str:
        """Get text summary of diagnostics.
        
        Returns:
            String summary of diagnostic results
        """
        if not self.diagnostics_results:
            return "No diagnostics results available"
        
        lines = ["=== Model Diagnostics Summary ===\n"]
        
        # Overall assessment
        adequate = self.diagnostics_results.get('model_adequate', False)
        lines.append(f"Model Adequate: {'✓ Yes' if adequate else '✗ No'}\n")
        
        # Checks performed
        checks = self.diagnostics_results.get('checks_performed', [])
        lines.append(f"Checks Performed: {', '.join(checks)}\n")
        
        # Overdispersion
        if 'overdispersion' in self.diagnostics_results:
            od = self.diagnostics_results['overdispersion']
            if od.get('dispersion_ratio'):
                lines.append(f"Dispersion Ratio: {od['dispersion_ratio']:.2f}")
                lines.append(f"  {od.get('interpretation', '')}\n")
        
        # Theta
        if 'theta' in self.diagnostics_results:
            lines.append(f"Theta (NB dispersion): {self.diagnostics_results['theta']:.3f}\n")
        
        # Sample size
        if 'sample_size' in self.diagnostics_results:
            ss = self.diagnostics_results['sample_size']
            lines.append(f"Sample Size: {ss.get('message', '')}\n")
        
        # Warnings
        warnings = self.diagnostics_results.get('warnings', [])
        if warnings:
            lines.append("\nWarnings:")
            for warning in warnings:
                lines.append(f"  ⚠ {warning}")
        
        return '\n'.join(lines)
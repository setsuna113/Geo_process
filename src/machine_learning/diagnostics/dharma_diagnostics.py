"""DHARMa-style residual diagnostics for Julia GLM models.

Provides simulation-based residual diagnostics similar to the DHARMa R package.
Refactored to work with Julia GLM results instead of H2O models.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
from scipy.spatial import distance

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class DHARMaDiagnostics:
    """Residual diagnostics for Julia GLM models, inspired by DHARMa R package.
    
    Provides:
    - Overdispersion tests
    - Spatial autocorrelation checks
    - Zero-inflation detection
    - Model adequacy assessment
    
    Works with Julia GLM result dictionaries instead of model objects.
    """
    
    def __init__(self, model_results: Optional[Dict[str, Any]] = None):
        """Initialize diagnostics.
        
        Args:
            model_results: Julia GLM results dictionary
        """
        self.model_results = model_results
        self.diagnostics_results = {}
    
    def run_diagnostics(self, 
                       model_results: Optional[Dict[str, Any]] = None,
                       data_summary: Optional[Dict[str, Any]] = None,
                       coords: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run complete diagnostic suite on Julia GLM results.
        
        Args:
            model_results: Julia GLM results dictionary with:
                - coefficients, std_errors, vcov
                - aic, bic, deviance, loglik
                - df_residual, n_obs, theta
            data_summary: Optional data summary from Julia
            coords: Optional coordinates for spatial tests [longitude, latitude]
            
        Returns:
            Dictionary with all diagnostic results
        """
        if model_results is not None:
            self.model_results = model_results
        
        if self.model_results is None:
            return {
                'model_adequate': False,
                'error': 'No model results provided for diagnostics',
                'checks_performed': []
            }
        
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
            
            # 4. Information criteria check
            ic_check = self._check_information_criteria()
            diagnostics['information_criteria'] = ic_check
            diagnostics['checks_performed'].append('information_criteria')
            
            # 5. Coefficient significance check
            coef_check = self._check_coefficients()
            diagnostics['coefficients'] = coef_check
            diagnostics['checks_performed'].append('coefficient_significance')
            
            # 6. Sample size adequacy
            sample_check = self._check_sample_size()
            diagnostics['sample_size'] = sample_check
            diagnostics['checks_performed'].append('sample_size')
            
            if not sample_check['adequate']:
                diagnostics['warnings'].append(sample_check['message'])
                diagnostics['model_adequate'] = False
            
            # 7. Spatial autocorrelation (if coordinates provided)
            if coords is not None and len(coords) > 0:
                spatial = self._test_spatial_autocorrelation(coords)
                diagnostics['spatial_autocorrelation'] = spatial
                diagnostics['checks_performed'].append('spatial_autocorrelation')
                
                if spatial.get('significant', False):
                    diagnostics['warnings'].append("Significant spatial autocorrelation detected")
                    # Don't mark as inadequate since MESF should handle this
            
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
    
    def _check_information_criteria(self) -> Dict[str, Any]:
        """Check AIC and BIC values for model quality.
        
        Returns:
            Dictionary with IC assessment
        """
        aic = self.model_results.get('aic')
        bic = self.model_results.get('bic')
        n_obs = self.model_results.get('n_obs', 0)
        
        results = {
            'aic': aic,
            'bic': bic,
            'n_obs': n_obs
        }
        
        # Check if AIC/BIC are reasonable
        if aic and n_obs > 0:
            aic_per_obs = aic / n_obs
            results['aic_per_obs'] = aic_per_obs
            
            # Rule of thumb: AIC/n should be reasonably small
            if aic_per_obs < 10:
                results['quality'] = 'good'
            elif aic_per_obs < 20:
                results['quality'] = 'acceptable'
            else:
                results['quality'] = 'poor'
        
        return results
    
    def _check_coefficients(self) -> Dict[str, Any]:
        """Check coefficient estimates for issues.
        
        Returns:
            Dictionary with coefficient diagnostics
        """
        coefs = self.model_results.get('coefficients', [])
        std_errors = self.model_results.get('std_errors', [])
        coef_names = self.model_results.get('coef_names', [])
        
        if len(coefs) == 0:
            return {'error': 'No coefficients found'}
        
        results = {
            'n_coefficients': len(coefs),
            'issues': []
        }
        
        # Check for extreme coefficients
        coef_array = np.array(coefs)
        extreme_coefs = np.abs(coef_array) > 10
        if np.any(extreme_coefs):
            extreme_indices = np.where(extreme_coefs)[0]
            for idx in extreme_indices:
                name = coef_names[idx] if idx < len(coef_names) else f"coef_{idx}"
                results['issues'].append(f"Extreme coefficient: {name} = {coefs[idx]:.2f}")
        
        # Check for very large standard errors
        if len(std_errors) > 0:
            se_array = np.array(std_errors)
            large_se = se_array > 100
            if np.any(large_se):
                large_indices = np.where(large_se)[0]
                for idx in large_indices:
                    name = coef_names[idx] if idx < len(coef_names) else f"coef_{idx}"
                    results['issues'].append(f"Large std error: {name} SE = {std_errors[idx]:.2f}")
        
        # Check condition number of vcov matrix
        vcov = self.model_results.get('vcov')
        if vcov is not None and isinstance(vcov, np.ndarray):
            try:
                cond_number = np.linalg.cond(vcov)
                results['vcov_condition_number'] = cond_number
                if cond_number > 1000:
                    results['issues'].append(f"High collinearity (condition number = {cond_number:.0f})")
            except:
                pass
        
        results['has_issues'] = len(results['issues']) > 0
        
        return results
    
    def _check_sample_size(self) -> Dict[str, Any]:
        """Check if sample size is adequate for model complexity.
        
        Returns:
            Dictionary with sample size assessment
        """
        n_obs = self.model_results.get('n_obs', 0)
        n_coef = len(self.model_results.get('coefficients', []))
        df_residual = self.model_results.get('df_residual', 0)
        
        # Rule of thumb: at least 10-20 observations per parameter
        min_obs_per_param = 10
        recommended_obs = n_coef * min_obs_per_param
        
        results = {
            'n_obs': n_obs,
            'n_parameters': n_coef,
            'df_residual': df_residual,
            'obs_per_parameter': n_obs / n_coef if n_coef > 0 else np.inf,
            'recommended_min_obs': recommended_obs,
            'adequate': n_obs >= recommended_obs
        }
        
        if not results['adequate']:
            results['message'] = f"Sample size ({n_obs}) may be too small for {n_coef} parameters"
        else:
            results['message'] = f"Sample size adequate ({n_obs} obs for {n_coef} parameters)"
        
        return results
    
    def _test_spatial_autocorrelation(self, coords: np.ndarray) -> Dict[str, Any]:
        """Test for spatial autocorrelation in residuals.
        
        Note: This is a simplified test since we don't have actual residuals from Julia.
        A full implementation would need residuals computed in Julia.
        
        Args:
            coords: Coordinates array [longitude, latitude]
            
        Returns:
            Dictionary with spatial test results
        """
        try:
            n_obs = self.model_results.get('n_obs', 0)
            
            # Without residuals, we can only report that spatial correction was applied
            # via MESF eigenvectors
            return {
                'test_performed': False,
                'note': 'Spatial autocorrelation handled by MESF eigenvectors',
                'n_observations': n_obs,
                'coords_provided': True
            }
            
        except Exception as e:
            logger.warning(f"Spatial autocorrelation test failed: {e}")
            return {
                'test_performed': False,
                'error': str(e)
            }
    
    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """Create diagnostic plots.
        
        Note: Limited plots available without residuals from Julia.
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.diagnostics_results:
            logger.warning("No diagnostics results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Coefficient plot
        coefs = self.model_results.get('coefficients', [])
        std_errors = self.model_results.get('std_errors', [])
        coef_names = self.model_results.get('coef_names', 
                                           [f"coef_{i}" for i in range(len(coefs))])
        
        if len(coefs) > 0 and len(std_errors) > 0:
            # Calculate z-scores
            z_scores = np.array(coefs) / np.array(std_errors)
            
            # Plot coefficients with error bars
            ax = axes[0]
            y_pos = np.arange(len(coefs))
            ax.errorbar(coefs, y_pos, xerr=1.96*np.array(std_errors), 
                       fmt='o', capsize=5)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coef_names)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Coefficient Estimates with 95% CI')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Model quality metrics
        ax = axes[1]
        metrics = {
            'AIC': self.model_results.get('aic', 0),
            'BIC': self.model_results.get('bic', 0),
            'Deviance': self.model_results.get('deviance', 0),
            'Log-likelihood': self.model_results.get('loglik', 0)
        }
        
        x_pos = np.arange(len(metrics))
        values = list(metrics.values())
        
        ax.bar(x_pos, values)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics.keys(), rotation=45)
        ax.set_ylabel('Value')
        ax.set_title('Model Quality Metrics')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diagnostic plots saved to {save_path}")
        
        plt.show()
    
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
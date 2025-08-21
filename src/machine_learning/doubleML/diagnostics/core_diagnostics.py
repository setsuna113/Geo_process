#!/usr/bin/env python3
"""
Core DML Diagnostics
Implements comprehensive diagnostics for Double Machine Learning models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class DMLDiagnostics:
    """Main diagnostics class for DML models"""
    
    def __init__(self, estimator, data: Optional[Dict] = None):
        """
        Initialize diagnostics
        
        Args:
            estimator: Fitted EconML estimator
            data: Optional dict with 'Y', 'T', 'W' arrays
        """
        self.estimator = estimator
        self.data = data
        self.diagnostics = {}
        
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostic suite"""
        
        print("="*70)
        print("RUNNING DML DIAGNOSTICS")
        print("="*70)
        
        # 1. Basic model info
        self.diagnostics['model_info'] = self._get_model_info()
        
        # 2. Coefficient diagnostics
        self.diagnostics['coefficient_diagnostics'] = self._check_coefficients()
        
        # 3. Cross-fitting diagnostics
        self.diagnostics['cross_fitting'] = self._check_cross_fitting()
        
        # 4. Orthogonality diagnostics
        self.diagnostics['orthogonality'] = self._check_orthogonality()
        
        # 5. Inference diagnostics
        self.diagnostics['inference'] = self._check_inference()
        
        # 6. Sample size diagnostics
        self.diagnostics['sample_size'] = self._check_sample_size()
        
        # 7. Convergence diagnostics
        self.diagnostics['convergence'] = self._check_convergence()
        
        print("\nDiagnostics complete!")
        return self.diagnostics
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Extract basic model information"""
        
        info = {
            'estimator_type': type(self.estimator).__name__,
            'has_intercept': hasattr(self.estimator, 'intercept_'),
            'has_coef': hasattr(self.estimator, 'coef_'),
            'n_treatments': None,
            'n_folds': getattr(self.estimator, 'cv', None)
        }
        
        # Get treatment dimensions
        if hasattr(self.estimator, 'intercept_'):
            info['n_treatments'] = len(self.estimator.intercept_)
        elif hasattr(self.estimator, 'coef_'):
            info['n_treatments'] = self.estimator.coef_.shape[0]
            
        print(f"\nModel Info:")
        print(f"  Type: {info['estimator_type']}")
        print(f"  Treatments: {info['n_treatments']}")
        print(f"  CV Folds: {info['n_folds']}")
        
        return info
    
    def _check_coefficients(self) -> Dict[str, Any]:
        """Diagnose coefficient estimates"""
        
        print("\nCoefficient Diagnostics:")
        
        # Get coefficients
        if hasattr(self.estimator, 'intercept_'):
            coef = self.estimator.intercept_
        elif hasattr(self.estimator, 'coef_'):
            coef = self.estimator.coef_.flatten()
        else:
            return {'status': 'No coefficients found'}
        
        # Get standard errors
        try:
            inf_results = self.estimator.intercept__inference()
            stderr = inf_results.stderr
        except:
            stderr = None
            
        diagnostics = {
            'coefficients': coef.tolist() if hasattr(coef, 'tolist') else coef,
            'n_coefficients': len(coef) if hasattr(coef, '__len__') else 1
        }
        
        if stderr is not None:
            # Calculate t-statistics
            t_stats = coef / stderr
            diagnostics['t_statistics'] = t_stats.tolist()
            diagnostics['std_errors'] = stderr.tolist()
            
            # Check for extreme values
            diagnostics['max_t_stat'] = float(np.max(np.abs(t_stats)))
            diagnostics['min_t_stat'] = float(np.min(np.abs(t_stats)))
            
            # Flag potential issues
            diagnostics['extreme_t_stats'] = np.sum(np.abs(t_stats) > 100).item()
            diagnostics['insignificant_coefs'] = np.sum(np.abs(t_stats) < 2).item()
            
            print(f"  Max |t-stat|: {diagnostics['max_t_stat']:.2f}")
            print(f"  Min |t-stat|: {diagnostics['min_t_stat']:.2f}")
            print(f"  Extreme t-stats (>100): {diagnostics['extreme_t_stats']}")
            print(f"  Insignificant (<2): {diagnostics['insignificant_coefs']}")
            
            # Coefficient stability
            if diagnostics['extreme_t_stats'] > 0:
                diagnostics['warning'] = "Extreme t-statistics detected - possible numerical instability"
                print(f"  ⚠ WARNING: {diagnostics['warning']}")
                
        return diagnostics
    
    def _check_cross_fitting(self) -> Dict[str, Any]:
        """Check cross-fitting diagnostics"""
        
        print("\nCross-Fitting Diagnostics:")
        
        diagnostics = {}
        
        # Check if cross-fitting was used
        if hasattr(self.estimator, 'cv'):
            diagnostics['n_folds'] = self.estimator.cv
            diagnostics['cross_fitting_used'] = True
            print(f"  Folds used: {diagnostics['n_folds']}")
        else:
            diagnostics['cross_fitting_used'] = False
            diagnostics['warning'] = "No cross-fitting detected"
            print(f"  ⚠ WARNING: {diagnostics['warning']}")
            return diagnostics
            
        # Check MC iterations
        if hasattr(self.estimator, 'mc_iters'):
            diagnostics['mc_iterations'] = self.estimator.mc_iters
            print(f"  MC iterations: {diagnostics['mc_iterations']}")
            
            if diagnostics['mc_iterations'] == 1:
                diagnostics['mc_warning'] = "Single MC iteration - consider increasing for stability"
                print(f"  ⚠ WARNING: {diagnostics['mc_warning']}")
                
        # Fold-wise coefficient variation (if available)
        if hasattr(self.estimator, '_models'):
            try:
                n_models = len(self.estimator._models)
                diagnostics['n_fitted_models'] = n_models
                print(f"  Fitted models: {n_models}")
            except:
                pass
                
        return diagnostics
    
    def _check_orthogonality(self) -> Dict[str, Any]:
        """Check Neyman orthogonality conditions"""
        
        print("\nOrthogonality Diagnostics:")
        
        diagnostics = {
            'neyman_orthogonal': True,  # LinearDML satisfies this by construction
            'score_type': 'doubly_robust'
        }
        
        # Check if score functions are available
        if hasattr(self.estimator, 'score_'):
            scores = self.estimator.score_
            diagnostics['score_mean'] = float(np.mean(scores))
            diagnostics['score_std'] = float(np.std(scores))
            
            # Score should have mean close to zero
            diagnostics['score_mean_test'] = abs(diagnostics['score_mean']) < 0.01
            
            print(f"  Score mean: {diagnostics['score_mean']:.6f}")
            print(f"  Score std: {diagnostics['score_std']:.6f}")
            
            if not diagnostics['score_mean_test']:
                diagnostics['warning'] = "Score mean not close to zero - possible misspecification"
                print(f"  ⚠ WARNING: {diagnostics['warning']}")
        else:
            print("  Score functions not available")
            diagnostics['scores_available'] = False
            
        return diagnostics
    
    def _check_inference(self) -> Dict[str, Any]:
        """Check inference validity"""
        
        print("\nInference Diagnostics:")
        
        diagnostics = {}
        
        try:
            # Get inference results
            inf_results = self.estimator.intercept__inference()
            
            # Check confidence interval coverage
            ci_lower = inf_results.conf_int()[0]
            ci_upper = inf_results.conf_int()[1]
            point_est = inf_results.point_estimate
            
            diagnostics['has_valid_inference'] = True
            diagnostics['ci_width'] = (ci_upper - ci_lower).tolist()
            
            # Check for extremely wide CIs
            relative_width = (ci_upper - ci_lower) / np.abs(point_est + 1e-10)
            diagnostics['relative_ci_width'] = relative_width.tolist()
            diagnostics['max_relative_width'] = float(np.max(relative_width))
            
            print(f"  Valid inference: {diagnostics['has_valid_inference']}")
            print(f"  Max relative CI width: {diagnostics['max_relative_width']:.2f}")
            
            if diagnostics['max_relative_width'] > 10:
                diagnostics['warning'] = "Very wide confidence intervals - weak identification"
                print(f"  ⚠ WARNING: {diagnostics['warning']}")
                
        except Exception as e:
            diagnostics['has_valid_inference'] = False
            diagnostics['inference_error'] = str(e)
            print(f"  ⚠ Inference failed: {e}")
            
        return diagnostics
    
    def _check_sample_size(self) -> Dict[str, Any]:
        """Check sample size adequacy"""
        
        print("\nSample Size Diagnostics:")
        
        diagnostics = {}
        
        if self.data and 'Y' in self.data:
            n = len(self.data['Y'])
            n_treatments = len(self.data['T'][0]) if len(self.data['T'].shape) > 1 else 1
            n_confounders = self.data['W'].shape[1] if 'W' in self.data else 0
            
            diagnostics['n_observations'] = n
            diagnostics['n_treatments'] = n_treatments
            diagnostics['n_confounders'] = n_confounders
            
            # Rule of thumb checks
            min_n_per_param = 100
            min_n_required = (n_treatments + n_confounders) * min_n_per_param
            
            diagnostics['min_n_required'] = min_n_required
            diagnostics['n_per_parameter'] = n / (n_treatments + n_confounders) if (n_treatments + n_confounders) > 0 else n
            diagnostics['adequate_sample_size'] = n >= min_n_required
            
            print(f"  N observations: {n:,}")
            print(f"  N per parameter: {diagnostics['n_per_parameter']:.0f}")
            print(f"  Adequate size: {diagnostics['adequate_sample_size']}")
            
            if not diagnostics['adequate_sample_size']:
                diagnostics['warning'] = f"Sample size may be insufficient (recommended: {min_n_required:,})"
                print(f"  ⚠ WARNING: {diagnostics['warning']}")
                
        return diagnostics
    
    def _check_convergence(self) -> Dict[str, Any]:
        """Check model convergence"""
        
        print("\nConvergence Diagnostics:")
        
        diagnostics = {
            'converged': True  # Assume converged unless we find issues
        }
        
        # Check for NaN or infinite values
        if hasattr(self.estimator, 'intercept_'):
            coef = self.estimator.intercept_
            diagnostics['has_nan'] = bool(np.any(np.isnan(coef)))
            diagnostics['has_inf'] = bool(np.any(np.isinf(coef)))
            
            if diagnostics['has_nan'] or diagnostics['has_inf']:
                diagnostics['converged'] = False
                diagnostics['warning'] = "NaN or infinite values in coefficients"
                print(f"  ⚠ WARNING: {diagnostics['warning']}")
            else:
                print("  No numerical issues detected")
                
        # Check condition number if covariance available
        if hasattr(self.estimator, 'cov_'):
            try:
                cond_num = np.linalg.cond(self.estimator.cov_)
                diagnostics['condition_number'] = float(cond_num)
                print(f"  Condition number: {cond_num:.2e}")
                
                if cond_num > 1e10:
                    diagnostics['high_collinearity'] = True
                    diagnostics['warning'] = "Very high condition number - possible multicollinearity"
                    print(f"  ⚠ WARNING: {diagnostics['warning']}")
            except:
                pass
                
        return diagnostics
    
    def generate_summary(self) -> str:
        """Generate text summary of diagnostics"""
        
        if not self.diagnostics:
            self.run_all_diagnostics()
            
        summary = []
        summary.append("="*70)
        summary.append("DML DIAGNOSTICS SUMMARY")
        summary.append("="*70)
        
        # Count warnings
        warnings_found = []
        for section, diag in self.diagnostics.items():
            if isinstance(diag, dict) and 'warning' in diag:
                warnings_found.append(f"{section}: {diag['warning']}")
                
        if warnings_found:
            summary.append("\n⚠ WARNINGS FOUND:")
            for w in warnings_found:
                summary.append(f"  - {w}")
        else:
            summary.append("\n✓ No major issues detected")
            
        # Key metrics
        summary.append("\nKEY METRICS:")
        
        if 'coefficient_diagnostics' in self.diagnostics:
            cd = self.diagnostics['coefficient_diagnostics']
            if 'max_t_stat' in cd:
                summary.append(f"  Max |t-statistic|: {cd['max_t_stat']:.2f}")
                
        if 'cross_fitting' in self.diagnostics:
            cf = self.diagnostics['cross_fitting']
            summary.append(f"  Cross-fitting folds: {cf.get('n_folds', 'N/A')}")
            
        if 'sample_size' in self.diagnostics:
            ss = self.diagnostics['sample_size']
            if 'n_observations' in ss:
                summary.append(f"  Sample size: {ss['n_observations']:,}")
                summary.append(f"  N per parameter: {ss.get('n_per_parameter', 'N/A'):.0f}")
                
        return "\n".join(summary)
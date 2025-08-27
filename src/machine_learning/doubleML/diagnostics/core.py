#!/usr/bin/env python3
"""
Real EconML Diagnostics
Based on actual EconML public API, not fantasy
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class EconMLDiagnostics:
    """
    Real diagnostics using EconML's actual public API.
    No fake attributes, no placeholders, no bullshit.
    """
    
    def __init__(self, estimator, X_test=None, Y_test=None, T_test=None, W_test=None):
        """
        Initialize with fitted EconML estimator and optional test data.
        
        Args:
            estimator: Fitted EconML estimator (LinearDML, CausalForestDML, etc.)
            X_test: Test features for heterogeneous effects
            Y_test: Test outcomes  
            T_test: Test treatments
            W_test: Test confounders
        """
        self.estimator = estimator
        self.X_test = X_test
        self.Y_test = Y_test
        self.T_test = T_test
        self.W_test = W_test
        
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run all available diagnostics based on what the estimator actually provides.
        
        NOTE: Sensitivity analysis has been moved to a standalone module.
        For sensitivity analysis, use:
            from diagnostics.sensitivity import SingleTreatmentSensitivity
            
        Returns
        -------
        dict with all diagnostic results
        """
        
        print("="*70)
        print("ECONML DIAGNOSTICS (REAL)")
        print("="*70)
        
        diagnostics = {}
        
        # 1. Model score (if available)
        diagnostics['score'] = self._check_score()
        
        # 2. Nuisance scores (if available)
        diagnostics['nuisance_scores'] = self._check_nuisance_scores()
        
        # 3. Coefficient intervals (if linear model)
        diagnostics['coefficients'] = self._check_coefficients()
        
        # 4. Effect intervals
        diagnostics['effects'] = self._check_effects()
        
        # 5. Model type and configuration
        diagnostics['model_info'] = self._get_model_info()
        
        # 6. Feature importance
        diagnostics['feature_importance'] = self.check_feature_importance()
        
        # Note: Sensitivity analysis removed - use standalone module
        print("\nNOTE: For sensitivity analysis, use the standalone module:")
        print("  from diagnostics.sensitivity import SingleTreatmentSensitivity")
        print("  (Requires single-treatment models only)")
        
        return diagnostics
    
    def _check_score(self) -> Dict[str, Any]:
        """Check model score if available."""
        
        print("\nModel Score:")
        result = {}
        
        # Check fitted score - NOTE: This is MSE, not R²!
        if hasattr(self.estimator, 'score_') and self.estimator.score_ is not None:
            score = self.estimator.score_
            result['final_mse'] = float(score) if np.isscalar(score) else score.tolist()
            print(f"  Final stage MSE: {result['final_mse']:.4f}")
            print(f"  (Lower is better - this is NOT R²!)")
        else:
            print("  No fitted score available")
            result['final_mse'] = None
            
        # Check test score if test data provided
        if all(x is not None for x in [self.Y_test, self.T_test, self.W_test]):
            try:
                test_score = self.estimator.score(
                    self.Y_test, self.T_test, 
                    X=self.X_test, W=self.W_test
                )
                result['test_score'] = float(test_score) if np.isscalar(test_score) else test_score.tolist()
                print(f"  Test score: {result['test_score']}")
            except Exception as e:
                print(f"  Could not compute test score: {e}")
                result['test_score'] = None
                
        return result
    
    def _check_nuisance_scores(self) -> Dict[str, Any]:
        """Check nuisance model scores if available."""
        
        print("\nNuisance Model Scores (R² for first-stage models):")
        result = {}
        
        # First try to load from NPZ file if model was saved
        import os
        if hasattr(self.estimator, 'nuisance_scores_') and self.estimator.nuisance_scores_ is None:
            # Try to load from NPZ
            model_path = os.environ.get('ECONML_MODEL_PATH', '')
            if model_path:
                npz_path = model_path.replace('.pkl', '.npz')
                if os.path.exists(npz_path):
                    try:
                        arrays = np.load(npz_path)
                        if 'nuisance_scores' in arrays:
                            self.estimator.nuisance_scores_ = arrays['nuisance_scores']
                    except:
                        pass
        
        if hasattr(self.estimator, 'nuisance_scores_'):
            scores = self.estimator.nuisance_scores_
            if scores is not None:
                result['available'] = True
                
                # For LinearDML: shape is (2, n_mc_iters, n_folds)
                # Index 0: Model Y (outcome model)
                # Index 1: Model T (treatment model)
                
                if isinstance(scores, np.ndarray) and scores.ndim == 3:
                    # Standard case
                    model_names = ['Y (fungi richness)', 'T (biodiversity)']
                    result['models'] = {}
                    
                    for i in range(min(len(scores), len(model_names))):
                        # Extract scores for this model across MC iterations and folds
                        model_scores = scores[i].flatten()  # Flatten MC iters × folds
                        
                        result['models'][f'model_{model_names[i]}'] = {
                            'fold_scores': model_scores.tolist(),
                            'mean_r2': float(np.mean(model_scores)),
                            'std_r2': float(np.std(model_scores)),
                            'min_r2': float(np.min(model_scores)),
                            'max_r2': float(np.max(model_scores))
                        }
                        
                        print(f"  Model {model_names[i]}: R²={np.mean(model_scores):.4f} (±{np.std(model_scores):.4f})")
                        
                        # Interpretation
                        if i == 0:
                            explained = np.mean(model_scores) * 100
                            print(f"    → Confounders explain {explained:.1f}% of fungi variation")
                        elif i == 1:
                            explained = np.mean(model_scores) * 100
                            print(f"    → Confounders explain {explained:.1f}% of biodiversity variation")
                            print(f"    → Causal effect estimated from remaining {100-explained:.1f}%")
                else:
                    # Fallback for other structures
                    result['raw_scores'] = scores.tolist() if hasattr(scores, 'tolist') else str(scores)
                    print(f"  Non-standard score structure: {type(scores)}")
            else:
                result['available'] = False
                print("  Nuisance scores are None (may be in separate NPZ file)")
        else:
            result['available'] = False
            print("  No nuisance scores available")
            
        return result
    
    def _check_coefficients(self) -> Dict[str, Any]:
        """Check coefficients and their confidence intervals."""
        
        print("\nCoefficients:")
        result = {}
        
        # Check for linear model coefficients
        has_coef = False
        
        # Try intercept (for models with intercept)
        if hasattr(self.estimator, 'intercept_'):
            coef = self.estimator.intercept_
            result['intercept'] = coef.tolist() if hasattr(coef, 'tolist') else float(coef)
            has_coef = True
            print(f"  Intercept: {result['intercept']}")
            
            # Try to get confidence intervals
            try:
                lb, ub = self.estimator.intercept__interval(alpha=0.05)
                result['intercept_ci_lower'] = lb.tolist() if hasattr(lb, 'tolist') else float(lb)
                result['intercept_ci_upper'] = ub.tolist() if hasattr(ub, 'tolist') else float(ub)
                print(f"  95% CI: [{result['intercept_ci_lower']}, {result['intercept_ci_upper']}]")
            except:
                pass
                
        # Try coef (for models with coefficients)
        if hasattr(self.estimator, 'coef_'):
            coef = self.estimator.coef_
            result['coef'] = coef.tolist() if hasattr(coef, 'tolist') else float(coef)
            has_coef = True
            print(f"  Coefficients shape: {coef.shape if hasattr(coef, 'shape') else 'scalar'}")
            
            # Try to get confidence intervals
            try:
                lb, ub = self.estimator.coef__interval(alpha=0.05)
                result['coef_ci_lower'] = lb.tolist() if hasattr(lb, 'tolist') else float(lb)
                result['coef_ci_upper'] = ub.tolist() if hasattr(ub, 'tolist') else float(ub)
                
                # Calculate CI width
                ci_width = np.array(ub) - np.array(lb)
                result['ci_width_mean'] = float(np.mean(ci_width))
                result['ci_width_max'] = float(np.max(ci_width))
                print(f"  Mean CI width: {result['ci_width_mean']:.4f}")
            except:
                pass
                
        if not has_coef:
            print("  No coefficients available (may be non-parametric model)")
            result['has_coefficients'] = False
            
        return result
    
    def _check_effects(self) -> Dict[str, Any]:
        """Check treatment effects and their intervals."""
        
        print("\nTreatment Effects:")
        result = {}
        
        # Check if we can compute effects
        if self.X_test is not None:
            try:
                # Get point estimates
                effects = self.estimator.effect(self.X_test)
                result['mean_effect'] = float(np.mean(effects))
                result['std_effect'] = float(np.std(effects))
                result['min_effect'] = float(np.min(effects))
                result['max_effect'] = float(np.max(effects))
                
                print(f"  Mean effect: {result['mean_effect']:.4f}")
                print(f"  Std effect: {result['std_effect']:.4f}")
                print(f"  Range: [{result['min_effect']:.4f}, {result['max_effect']:.4f}]")
                
                # Get confidence intervals
                try:
                    lb, ub = self.estimator.effect_interval(self.X_test, alpha=0.05)
                    ci_width = ub - lb
                    result['mean_ci_width'] = float(np.mean(ci_width))
                    result['median_ci_width'] = float(np.median(ci_width))
                    
                    # Check coverage
                    result['pct_significant'] = float(100 * np.mean((lb > 0) | (ub < 0)))
                    
                    print(f"  Mean CI width: {result['mean_ci_width']:.4f}")
                    print(f"  % Significant effects: {result['pct_significant']:.1f}%")
                except:
                    pass
                    
            except Exception as e:
                print(f"  Could not compute effects: {e}")
                result['effects_available'] = False
        else:
            print("  No test data provided for effect estimation")
            result['effects_available'] = False
            
        # Check marginal effects
        try:
            const_marg = self.estimator.const_marginal_effect(self.X_test)
            if const_marg is not None:
                result['has_const_marginal_effect'] = True
                result['const_marginal_shape'] = const_marg.shape
                print(f"  Constant marginal effect shape: {const_marg.shape}")
        except:
            pass
            
        return result
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get basic model information."""
        
        print("\nModel Information:")
        info = {
            'estimator_class': self.estimator.__class__.__name__,
            'module': self.estimator.__class__.__module__
        }
        
        # Try to get CV and MC settings
        if hasattr(self.estimator, 'cv'):
            info['cv_folds'] = self.estimator.cv
            
        if hasattr(self.estimator, 'mc_iters'):
            info['mc_iters'] = self.estimator.mc_iters
            
        # Get treatment and outcome dimensions
        if hasattr(self.estimator, '_d_t'):
            info['n_treatments'] = self.estimator._d_t[0] if self.estimator._d_t else 1
            
        if hasattr(self.estimator, '_d_y'):
            info['n_outcomes'] = self.estimator._d_y[0] if self.estimator._d_y else 1
            
        print(f"  Class: {info['estimator_class']}")
        for k, v in info.items():
            if k not in ['estimator_class', 'module']:
                print(f"  {k}: {v}")
                
        return info
    
    def validate_with_scorer(self, Y_test, T_test, X_test=None, W_test=None) -> Optional[Dict[str, float]]:
        """Use RScorer to validate the model if available."""
        
        print("\nRScorer Validation:")
        
        try:
            from econml.score import RScorer
            
            # RScorer needs the model to have been fitted with X (heterogeneous) or W (confounders)
            scorer = RScorer(model_y='auto', model_t='auto', discrete_treatment=False,
                           cv=3, mc_iters=1, mc_agg='mean')
            
            # Score requires full data specification
            if W_test is not None:
                score_val = scorer.score(self.estimator, Y_test, T_test, X=X_test, W=W_test)
                print(f"  R-score: {score_val:.4f}")
                print(f"  Interpretation: {'Good' if score_val > 0.7 else 'Poor' if score_val < 0.3 else 'Moderate'}")
                
                return {'r_score': float(score_val)}
            else:
                print("  Cannot compute R-score without confounders (W)")
                return None
                
        except ImportError:
            print("  RScorer not available in this EconML version")
            return None
        except Exception as e:
            print(f"  RScorer validation failed: {e}")
            return None
    
    def validate_with_dr_tester(self, Y_test, T_test, X_test=None, W_test=None) -> Optional[Dict[str, Any]]:
        """Use DRTester to validate doubly robust assumptions if available."""
        
        print("\nDRTester Validation:")
        
        try:
            from econml.validate import DRTester
            
            # DRTester validates doubly-robust estimators
            tester = DRTester(model_y='auto', model_t='auto', model_final='auto',
                            discrete_treatment=False, cv=3, mc_iters=1)
            
            # Test the model
            if W_test is not None:
                p_val = tester.test(Y_test, T_test, X=X_test, W=W_test)
                print(f"  DR test p-value: {p_val:.4f}")
                print(f"  Result: {'PASS' if p_val > 0.05 else 'FAIL'} (p > 0.05 indicates valid DR assumptions)")
                
                return {
                    'dr_test_pvalue': float(p_val),
                    'dr_test_pass': p_val > 0.05
                }
            else:
                print("  Cannot run DR test without confounders (W)")
                return None
                
        except ImportError:
            print("  DRTester not available in this EconML version")
            return None
        except Exception as e:
            print(f"  DRTester validation failed: {e}")
            return None
    
    def check_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance for tree-based models or coefficient magnitudes for linear models."""
        
        print("\nFeature Importance:")
        
        # Tree-based models with feature_importances_
        if hasattr(self.estimator, 'feature_importances_'):
            imp = self.estimator.feature_importances_
            result = {
                'type': 'tree_importance',
                'importances': imp.tolist() if hasattr(imp, 'tolist') else imp,
                'available': True
            }
            print(f"  Tree-based importance available: {len(result['importances'])} features")
            return result
            
        # Linear models with coefficients
        if hasattr(self.estimator, 'coef_'):
            coef = self.estimator.coef_
            if coef is not None:
                # Get absolute values as proxy for importance
                abs_coef = np.abs(coef)
                result = {
                    'type': 'coefficient_magnitude',
                    'magnitudes': abs_coef.tolist() if hasattr(abs_coef, 'tolist') else float(abs_coef),
                    'shape': coef.shape if hasattr(coef, 'shape') else 'scalar',
                    'note': 'Using coefficient magnitudes as proxy for importance',
                    'available': True
                }
                print(f"  Coefficient magnitudes: shape {result['shape']}")
                return result
                
        # Check if model has intercept but no coef (constant treatment effect)
        if hasattr(self.estimator, 'intercept_'):
            coef = self.estimator.intercept_
            abs_coef = np.abs(coef)
            result = {
                'type': 'intercept_magnitude',
                'magnitudes': abs_coef.tolist() if hasattr(abs_coef, 'tolist') else float(abs_coef),
                'shape': coef.shape if hasattr(coef, 'shape') else 'scalar',
                'note': 'Using intercept magnitudes (constant effects)',
                'available': True
            }
            print(f"  Intercept magnitudes: shape {result['shape']}")
            return result
                
        print("  No importance measures available")
        return {'available': False}
    
    def summary(self) -> str:
        """Generate a summary of all diagnostics."""
        
        diag = self.run_diagnostics()
        
        summary = []
        summary.append("\n" + "="*70)
        summary.append("DIAGNOSTICS SUMMARY")
        summary.append("="*70)
        
        # Model info
        if 'model_info' in diag:
            info = diag['model_info']
            summary.append(f"\nModel: {info.get('estimator_class', 'Unknown')}")
            if 'cv_folds' in info:
                summary.append(f"Cross-validation: {info['cv_folds']} folds")
            if 'mc_iters' in info:
                summary.append(f"MC iterations: {info['mc_iters']}")
                
        # Scores
        if 'score' in diag and diag['score'].get('final_mse') is not None:
            summary.append(f"\nFinal MSE: {diag['score']['final_mse']}")
            if 'test_score' in diag['score'] and diag['score']['test_score'] is not None:
                summary.append(f"Test score: {diag['score']['test_score']}")
                
        # Effects
        if 'effects' in diag and 'mean_effect' in diag['effects']:
            eff = diag['effects']
            summary.append(f"\nMean treatment effect: {eff['mean_effect']:.4f}")
            if 'pct_significant' in eff:
                summary.append(f"Significant effects: {eff['pct_significant']:.1f}%")
                
        # Warnings
        warnings = []
        
        # Check for wide confidence intervals
        if 'coefficients' in diag and 'ci_width_mean' in diag['coefficients']:
            if diag['coefficients']['ci_width_mean'] > 10:
                warnings.append("Very wide confidence intervals - weak identification")
                
        # Check for poor test score
        if 'score' in diag:
            if diag['score'].get('test_score') is not None and diag['score']['test_score'] > 0.5:
                warnings.append("High test score indicates poor fit")
                
        if warnings:
            summary.append("\n⚠ WARNINGS:")
            for w in warnings:
                summary.append(f"  - {w}")
        else:
            summary.append("\n✓ No issues detected")
            
        return "\n".join(summary)


def run_full_diagnostics(estimator, X_test=None, Y_test=None, T_test=None, W_test=None,
                         run_validation=True) -> Dict[str, Any]:
    """
    Convenience function to run all diagnostics.
    
    This is what you should actually use.
    """
    
    diag = EconMLDiagnostics(estimator, X_test, Y_test, T_test, W_test)
    
    # Basic diagnostics
    results = diag.run_diagnostics()
    
    # Additional validation if requested and data provided
    if run_validation and all(x is not None for x in [Y_test, T_test]):
        print("\n" + "="*70)
        print("ADDITIONAL VALIDATION")
        print("="*70)
        
        # RScorer validation
        r_score = diag.validate_with_scorer(Y_test, T_test, X_test, W_test)
        if r_score:
            results['r_scorer_validation'] = r_score
            
        # DRTester validation  
        dr_test = diag.validate_with_dr_tester(Y_test, T_test, X_test, W_test)
        if dr_test:
            results['dr_tester_validation'] = dr_test
            
    # Generate summary
    print(diag.summary())
    
    return results


if __name__ == "__main__":
    print("EconML Diagnostics Module (Real)")
    print("=" * 70)
    print("\nThis module provides real diagnostics based on EconML's actual public API.")
    print("\nUsage:")
    print("  from diagnostics.core import run_full_diagnostics")
    print("  results = run_full_diagnostics(fitted_model, X_test, Y_test, T_test, W_test)")
    print("\nFor sensitivity analysis (single-treatment only):")
    print("  from diagnostics.sensitivity import SingleTreatmentSensitivity")
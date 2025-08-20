"""Simplified Wald Method Wrapper for Method 1.

Orchestrates traditional, SESOI, and TOST tests without duplication.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Use the Wald test
from ..hypothesis.wald_test import WaldTest

logger = logging.getLogger(__name__)


class WaldMethod:
    """Simplified wrapper for Wald tests with SESOI and TOST."""
    
    def __init__(self, 
                 sesoi_threshold: float = 0.1,
                 tost_bounds: Tuple[float, float] = (-0.05, 0.05)):
        """Initialize with test parameters.
        
        Args:
            sesoi_threshold: SESOI threshold (default: 0.1 = 10% change)
            tost_bounds: TOST equivalence bounds (default: ±5%)
        """
        self.sesoi_threshold = sesoi_threshold
        self.tost_bounds = tost_bounds
        logger.info(f"WaldMethod initialized: SESOI={sesoi_threshold}, TOST={tost_bounds}")
    
    def run_all_tests(self, 
                      coefficients: np.ndarray,
                      std_errors: np.ndarray,
                      coef_names: List[str],
                      vcov_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run all three Wald test variants on all predictors.
        
        Args:
            coefficients: Array of coefficient estimates
            std_errors: Array of standard errors
            coef_names: List of coefficient names
            vcov_matrix: Optional covariance matrix for joint test
        
        Returns:
            Dictionary with all test results
        """
        results = {'predictors': {}}
        
        # Track biodiversity predictors separately for joint test
        bio_indices = []
        bio_names = []
        
        # Test ALL predictors (except intercept)
        for i, name in enumerate(coef_names):
            # Skip intercept
            if name.lower() in ['(intercept)', 'intercept']:
                continue
                
            # Track biodiversity predictors for joint test
            if 'plants' in name.lower() or 'vertebrates' in name.lower():
                bio_indices.append(i)
                bio_names.append(name)
        
        logger.info(f"\nTesting {len(coef_names)-1} predictors (excluding intercept)")
        
        # Test each predictor with all three methods
        for idx, name in enumerate(coef_names):
            # Skip intercept
            if name.lower() in ['(intercept)', 'intercept']:
                continue
            coef = coefficients[idx]
            se = std_errors[idx]
            
            logger.info(f"\n{name}: β={coef:.4f} (SE={se:.4f})")
            
            # Run all three tests
            traditional = WaldTest.test(coef, se)
            sesoi = WaldTest.test_sesoi(coef, se, self.sesoi_threshold)
            tost = WaldTest.test_equivalence(coef, se, 
                                            self.tost_bounds[0], 
                                            self.tost_bounds[1])
            
            # Convert to response scale (assuming log link)
            response_scale = WaldTest.to_response_scale(coef, se, 'log')
            
            # Combined interpretation
            interpretation = self._interpret_results(traditional, sesoi, tost)
            
            results['predictors'][name] = {
                'coefficient': coef,
                'standard_error': se,
                'traditional': traditional,
                'sesoi': sesoi,
                'tost': tost,
                'response_scale': response_scale,
                'interpretation': interpretation
            }
            
            logger.info(f"  Traditional: p={traditional['p_value']:.4f}")
            logger.info(f"  SESOI: {sesoi['interpretation']}")
            logger.info(f"  TOST: {tost['conclusion']}")
            logger.info(f"  Overall: {interpretation}")
        
        # Joint test if multiple predictors and vcov available
        if len(bio_indices) >= 2 and vcov_matrix is not None:
            bio_coefs = coefficients[bio_indices]
            bio_vcov = vcov_matrix[np.ix_(bio_indices, bio_indices)]
            
            joint = WaldTest.test_multiple(bio_coefs, bio_vcov)
            results['joint_test'] = joint
            logger.info(f"\nJoint test: χ²={joint['chi2_statistic']:.2f}, "
                       f"p={joint['p_value']:.4f}")
        
        # Overall summary
        results['summary'] = self._summarize_results(results)
        
        return results
    
    def _interpret_results(self, traditional: Dict, sesoi: Dict, tost: Dict) -> str:
        """Combine results from all three tests into interpretation.
        
        Args:
            traditional: Traditional test results
            sesoi: SESOI test results
            tost: TOST test results
            
        Returns:
            Combined interpretation string
        """
        trad_sig = traditional['significant']
        exceeds_sesoi = sesoi['exceeds_threshold']
        is_equivalent = tost['is_equivalent']
        
        if exceeds_sesoi and trad_sig:
            return "Practically and statistically significant"
        elif is_equivalent:
            return "Practically equivalent to zero (negligible)"
        elif exceeds_sesoi and not trad_sig:
            return "Exceeds threshold but not statistically significant"
        elif trad_sig and not exceeds_sesoi:
            return "Statistically significant but below meaningful threshold"
        else:
            return "No evidence of meaningful effect"
    
    def _summarize_results(self, results: Dict) -> Dict[str, Any]:
        """Summarize results across all predictors.
        
        Args:
            results: Full results dictionary
            
        Returns:
            Summary statistics
        """
        summary = {
            'n_predictors': len(results.get('predictors', {})),
            'n_significant': 0,
            'n_meaningful': 0,
            'n_negligible': 0
        }
        
        for name, pred_results in results.get('predictors', {}).items():
            interp = pred_results.get('interpretation', '')
            
            if 'statistically significant' in interp.lower():
                summary['n_significant'] += 1
            if 'practically' in interp.lower() and 'significant' in interp.lower():
                summary['n_meaningful'] += 1
            if 'negligible' in interp.lower() or 'equivalent' in interp.lower():
                summary['n_negligible'] += 1
        
        # Overall conclusion
        if summary['n_meaningful'] > 0:
            summary['conclusion'] = "Evidence for meaningful biodiversity effects"
        elif summary['n_negligible'] == summary['n_predictors']:
            summary['conclusion'] = "All effects are negligible"
        elif summary['n_significant'] > 0:
            summary['conclusion'] = "Statistical but not practical significance"
        else:
            summary['conclusion'] = "No evidence of effects"
        
        return summary
    
    def create_forest_plot(self, 
                          results: Dict[str, Any],
                          output_dir: Optional[Path] = None,
                          filename: str = "wald_forest_plot.png"):
        """Create forest plot visualization.
        
        Delegates to visualization module.
        
        Args:
            results: Results from run_all_tests()
            output_dir: Optional directory to save plot
            filename: Plot filename
            
        Returns:
            Path to saved plot or figure object
        """
        try:
            # Convert our simplified results to format expected by visualization
            wald_results = {}
            for name, pred in results.get('predictors', {}).items():
                wald_results[name] = {
                    'coefficient': pred['coefficient'],
                    'standard_error': pred['standard_error'],
                    'sesoi_test': pred['sesoi'],
                    'tost_test': pred['tost'],
                    'summary': {
                        'interpretation': pred['interpretation'],
                        'percent_change': pred['response_scale']['percent_change']
                    }
                }
            
            # Create figure using new API
            from ..visualization.forest_plots import create_wald_visualization
            fig = create_wald_visualization(
                wald_results=wald_results,
                sesoi_threshold=self.sesoi_threshold,
                tost_bounds=self.tost_bounds,
                title='Wald Method: Combined Analysis'
            )
            
            # Handle file I/O if needed
            if output_dir and fig:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / filename
                
                import matplotlib.pyplot as plt
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Forest plot saved to {save_path}")
                return save_path
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create forest plot: {e}")
            return None
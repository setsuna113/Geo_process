"""Wald Method Wrapper for SESOI and TOST versions.

This module provides a unified interface for running Wald tests with:
1. SESOI (Smallest Effect Size of Interest) version
2. TOST (Two One-Sided Tests) wrapped version
Plus visualization (forest plot) generation.

Method 1 calls this wrapper to perform all Wald-related testing.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Import hypothesis test modules
from ..hypothesis.wald_test import WaldTest
from ..hypothesis.h0_definitions import H0Definition
from ..hypothesis.tost_wrapper import TOSTWrapper

logger = logging.getLogger(__name__)


class WaldMethod:
    """Unified Wald test wrapper for SESOI and TOST versions with visualization.
    
    This class:
    1. Runs Wald test with SESOI H0 definition
    2. Runs Wald test wrapped with TOST
    3. Generates forest plot visualization
    4. Returns comprehensive results
    """
    
    def __init__(self, sesoi_thresholds: Optional[Dict[str, float]] = None,
                 tost_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """Initialize Wald method wrapper.
        
        Args:
            sesoi_thresholds: Dictionary mapping predictor names to SESOI thresholds
                             Default: 0.1 for all predictors
            tost_bounds: Dictionary mapping predictor names to TOST bounds (lower, upper)
                        Default: (-0.05, 0.05) for all predictors
        """
        # Default SESOI thresholds (10% change is meaningful)
        self.sesoi_thresholds = sesoi_thresholds or {
            'plants_richness_log': 0.1,
            'vertebrates_richness_log': 0.1
        }
        
        # Default TOST bounds (±5% is negligible)
        self.tost_bounds = tost_bounds or {
            'plants_richness_log': (-0.05, 0.05),
            'vertebrates_richness_log': (-0.05, 0.05)
        }
        
        logger.info("Wald Method wrapper initialized")
    
    def run_wald_tests(self, 
                       coefficients: np.ndarray,
                       std_errors: np.ndarray,
                       coef_names: List[str],
                       vcov_matrix: Optional[np.ndarray] = None,
                       bio_predictors: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run Wald tests with both SESOI and TOST versions.
        
        Args:
            coefficients: Array of coefficient estimates
            std_errors: Array of standard errors
            coef_names: List of coefficient names
            vcov_matrix: Optional variance-covariance matrix for joint tests
            bio_predictors: Optional list of biodiversity predictor names to test
                           If None, detects automatically from coef_names
        
        Returns:
            Dictionary containing:
            - Individual test results (SESOI and TOST for each predictor)
            - Joint test results (if applicable)
            - Summary statistics
        """
        results = {}
        
        # Auto-detect biodiversity predictors if not provided
        if bio_predictors is None:
            bio_predictors = []
            bio_indices = []
            for i, name in enumerate(coef_names):
                if 'plants' in name.lower() or 'vertebrates' in name.lower():
                    bio_predictors.append(name)
                    bio_indices.append(i)
        else:
            bio_indices = [i for i, name in enumerate(coef_names) if name in bio_predictors]
        
        if not bio_predictors:
            logger.warning("No biodiversity predictors found for Wald testing")
            return results
        
        logger.info("\n" + "="*60)
        logger.info("WALD METHOD: SESOI & TOST TESTING")
        logger.info("="*60)
        
        # Test each biodiversity coefficient
        for idx, name in zip(bio_indices, bio_predictors):
            if idx < len(coefficients) and idx < len(std_errors):
                coef = coefficients[idx]
                se = std_errors[idx]
                
                logger.info(f"\nTesting: {name}")
                logger.info(f"  Coefficient: {coef:.4f} (SE: {se:.4f})")
                
                # Get thresholds for this predictor
                sesoi = self.sesoi_thresholds.get(name, 0.1)
                bounds = self.tost_bounds.get(name, (-0.05, 0.05))
                
                # 1. SESOI Test
                sesoi_result = self._run_sesoi_test(coef, se, sesoi, name)
                
                # 2. TOST Test
                tost_result = self._run_tost_test(coef, se, bounds, name)
                
                # 3. Combined interpretation
                interpretation = self._interpret_combined_results(sesoi_result, tost_result)
                
                # Store results
                results[name] = {
                    'coefficient': coef,
                    'standard_error': se,
                    'sesoi_test': sesoi_result,
                    'tost_test': tost_result,
                    'summary': {
                        'exceeds_sesoi': sesoi_result['reject_h0'],
                        'is_equivalent': tost_result['is_equivalent'],
                        'is_different_from_zero': tost_result['is_different_from_zero'],
                        'percent_change': sesoi_result['response_scale']['percent_change'],
                        'interpretation': interpretation
                    }
                }
        
        # Joint test for all biodiversity coefficients (if multiple and vcov provided)
        if len(bio_indices) >= 2 and vcov_matrix is not None:
            joint_result = self._run_joint_test(
                coefficients[bio_indices],
                vcov_matrix[np.ix_(bio_indices, bio_indices)],
                bio_predictors
            )
            results['joint_test'] = joint_result
        
        return results
    
    def _run_sesoi_test(self, coef: float, se: float, sesoi: float, name: str) -> Dict[str, Any]:
        """Run SESOI version of Wald test.
        
        Args:
            coef: Coefficient estimate
            se: Standard error
            sesoi: SESOI threshold
            name: Predictor name
        
        Returns:
            SESOI test results with response scale interpretation
        """
        # Create SESOI H0 definition
        h0_sesoi = H0Definition.sesoi(threshold=sesoi, direction='greater')
        
        # Run Wald test with SESOI H0
        sesoi_result = WaldTest.test_single_coefficient(
            coefficient=coef,
            standard_error=se,
            h0_definition=h0_sesoi
        )
        
        # Add response scale interpretation (for log-link models)
        pct_change = (np.exp(coef) - 1) * 100
        pct_ci_lower = (np.exp(coef - 1.96*se) - 1) * 100
        pct_ci_upper = (np.exp(coef + 1.96*se) - 1) * 100
        sesoi_pct = (np.exp(sesoi) - 1) * 100
        
        sesoi_result['response_scale'] = {
            'percent_change': pct_change,
            'ci_lower_percent': pct_ci_lower,
            'ci_upper_percent': pct_ci_upper,
            'sesoi_threshold_percent': sesoi_pct,
            'exceeds_threshold': abs(pct_change) > sesoi_pct
        }
        
        # Add reject_h0 flag for compatibility
        sesoi_result['reject_h0'] = sesoi_result.get('significant', False) or sesoi_result.get('exceeds_sesoi', False)
        
        # Add practical significance interpretation
        if sesoi_result['reject_h0']:
            sesoi_result['practical_significance'] = f"Effect exceeds meaningful threshold ({sesoi_pct:.1f}%)"
        else:
            sesoi_result['practical_significance'] = f"Effect below meaningful threshold ({sesoi_pct:.1f}%)"
        
        logger.info(f"  SESOI Test: p={sesoi_result['p_value']:.4f}, "
                   f"{'✅ Exceeds threshold' if sesoi_result['reject_h0'] else '❌ Below threshold'}")
        logger.info(f"  Response scale: {pct_change:.1f}% change [95% CI: {pct_ci_lower:.1f}%, {pct_ci_upper:.1f}%]")
        
        return sesoi_result
    
    def _run_tost_test(self, coef: float, se: float, bounds: Tuple[float, float], name: str) -> Dict[str, Any]:
        """Run TOST wrapped version of Wald test.
        
        Args:
            coef: Coefficient estimate
            se: Standard error
            bounds: TOST equivalence bounds (lower, upper)
            name: Predictor name
        
        Returns:
            TOST test results
        """
        # Run TOST using the wrapper
        tost_result = TOSTWrapper.perform_tost(
            test_function=WaldTest.test_single_coefficient,
            effect_estimate=coef,
            standard_error=se,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            alpha=0.05
        )
        
        logger.info(f"  TOST Test: p={tost_result['tost_p_value']:.4f}")
        logger.info(f"  Equivalence bounds: [{bounds[0]}, {bounds[1]}]")
        logger.info(f"  Conclusion: {tost_result['conclusion']}")
        
        return tost_result
    
    def _interpret_combined_results(self, sesoi: Dict[str, Any], tost: Dict[str, Any]) -> str:
        """Interpret combined SESOI and TOST results.
        
        Args:
            sesoi: SESOI test results
            tost: TOST equivalence test results
            
        Returns:
            Combined interpretation string
        """
        exceeds_sesoi = sesoi.get('reject_h0', False)
        is_equivalent = tost.get('is_equivalent', False)
        is_different = tost.get('is_different_from_zero', False)
        
        if exceeds_sesoi and is_different:
            return "Practically and statistically significant"
        elif is_equivalent:
            return "Practically equivalent to zero (negligible effect)"
        elif exceeds_sesoi and not is_different:
            return "Exceeds threshold but not statistically significant"
        else:
            return "Inconclusive (insufficient evidence)"
    
    def _run_joint_test(self, coefs: np.ndarray, vcov: np.ndarray, names: List[str]) -> Dict[str, Any]:
        """Run joint Wald test for multiple coefficients.
        
        Args:
            coefs: Array of coefficients
            vcov: Variance-covariance matrix subset
            names: List of coefficient names
        
        Returns:
            Joint test results
        """
        # Joint test statistic: β'V^{-1}β
        chi2_stat = np.dot(coefs, np.linalg.solve(vcov, coefs))
        df = len(coefs)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi2_stat, df)  # Chi-squared p-value
        
        joint_result = {
            'chi2_statistic': chi2_stat,
            'df': df,
            'p_value': p_value,
            'reject_h0': p_value < 0.05,
            'coefficients_tested': names,
            'interpretation': f'Joint test for practical significance of {len(names)} biodiversity predictors'
        }
        
        logger.info(f"\nJoint Test: χ²={chi2_stat:.2f}, df={df}, p={p_value:.4f}")
        
        return joint_result
    
    def create_forest_plot(self, 
                          wald_results: Dict[str, Any],
                          output_dir: Optional[Path] = None,
                          filename: str = "wald_forest_plot.png") -> Optional[Path]:
        """Create forest plot for Wald test results with SESOI and TOST regions.
        
        Args:
            wald_results: Results from run_wald_tests()
            output_dir: Directory to save plot (if None, returns figure)
            filename: Name of output file
            
        Returns:
            Path to saved plot or None
        """
        try:
            # Prepare data for plotting
            predictors = []
            effects = []
            ci_lowers = []
            ci_uppers = []
            colors = []
            interpretations = []
            
            for name, result in wald_results.items():
                if 'joint' not in name and 'summary' in result:
                    # Clean up name for display
                    display_name = name.replace('_log', '').replace('_', ' ').title()
                    predictors.append(display_name)
                    
                    # Get response scale effects from SESOI test
                    if 'response_scale' in result.get('sesoi_test', {}):
                        resp = result['sesoi_test']['response_scale']
                        effects.append(resp['percent_change'])
                        ci_lowers.append(resp['ci_lower_percent'])
                        ci_uppers.append(resp['ci_upper_percent'])
                    else:
                        # Fallback calculation
                        coef = result['coefficient']
                        se = result['standard_error']
                        pct = (np.exp(coef) - 1) * 100
                        effects.append(pct)
                        ci_lowers.append((np.exp(coef - 1.96*se) - 1) * 100)
                        ci_uppers.append((np.exp(coef + 1.96*se) - 1) * 100)
                    
                    # Determine color based on interpretation
                    interp = result['summary'].get('interpretation', '')
                    interpretations.append(interp)
                    
                    if 'Practically and statistically significant' in interp:
                        colors.append('darkgreen')
                    elif 'negligible' in interp or 'equivalent' in interp:
                        colors.append('gray')
                    elif 'Exceeds threshold' in interp:
                        colors.append('orange')
                    else:
                        colors.append('darkred')
            
            if not predictors:
                logger.warning("No predictors to plot in forest plot")
                return None
            
            # Create forest plot
            fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(14, 6), 
                                                      gridspec_kw={'width_ratios': [3, 1]})
            
            y_pos = np.arange(len(predictors))
            
            # Main plot
            # Add reference regions with proper SESOI and TOST bounds
            sesoi_pct = 10.5  # Default SESOI threshold in percent (exp(0.1) - 1) * 100
            tost_pct = 5.1    # Default TOST bound in percent (exp(0.05) - 1) * 100
            
            # TOST equivalence region (narrowest, gray)
            ax_main.axvspan(-tost_pct, tost_pct, alpha=0.15, color='gray', 
                            label=f'TOST Equivalence (±{tost_pct:.1f}%)', zorder=1)
            
            # SESOI threshold region (medium, light blue)
            ax_main.axvspan(-sesoi_pct, -tost_pct, alpha=0.1, color='blue', zorder=1)
            ax_main.axvspan(tost_pct, sesoi_pct, alpha=0.1, color='blue', 
                            label=f'Below SESOI (±{sesoi_pct:.1f}%)', zorder=1)
            
            # Meaningful effect regions (widest, light green/red)
            ax_main.axvspan(sesoi_pct, 100, alpha=0.1, color='green', 
                            label=f'Exceeds SESOI (>{sesoi_pct:.1f}%)', zorder=1)
            ax_main.axvspan(-100, -sesoi_pct, alpha=0.1, color='red', 
                            label=f'Exceeds SESOI (<-{sesoi_pct:.1f}%)', zorder=1)
            
            # Add reference lines
            ax_main.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax_main.axvline(x=-tost_pct, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax_main.axvline(x=tost_pct, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax_main.axvline(x=-sesoi_pct, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            ax_main.axvline(x=sesoi_pct, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            
            # Plot effects with CIs
            for i, (effect, ci_low, ci_high, color) in enumerate(zip(effects, ci_lowers, ci_uppers, colors)):
                # Error bars
                ax_main.errorbar(effect, i, xerr=[[effect-ci_low], [ci_high-effect]], 
                               fmt='o', color=color, markersize=10, 
                               capsize=6, capthick=2.5, linewidth=2.5, 
                               elinewidth=2, markeredgewidth=2,
                               markeredgecolor='white', zorder=3)
                
                # Add value label
                ax_main.text(effect, i+0.25, f'{effect:.1f}%', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Formatting
            ax_main.set_yticks(y_pos)
            ax_main.set_yticklabels(predictors, fontsize=11)
            ax_main.set_xlabel('Percent Change in Fungi Richness (%)', fontsize=12)
            ax_main.set_title('Wald Method: Forest Plot with SESOI & TOST Regions', 
                             fontsize=14, fontweight='bold')
            ax_main.grid(True, alpha=0.2, axis='x')
            ax_main.set_xlim([-50, 50])
            
            # Legend panel
            ax_legend.axis('off')
            
            # Create comprehensive legend
            legend_items = []
            legend_items.append("INTERPRETATION GUIDE\n" + "="*20 + "\n")
            legend_items.append("Regions:\n")
            legend_items.append(f"• Gray: TOST equivalence (±{tost_pct:.1f}%)\n")
            legend_items.append(f"• Light blue: Below SESOI\n")
            legend_items.append(f"• Light green/red: Exceeds SESOI\n\n")
            
            legend_items.append("Point Colors:\n")
            legend_items.append("• Green: Significant & meaningful\n")
            legend_items.append("• Gray: Equivalent to zero\n")
            legend_items.append("• Orange: Exceeds SESOI, not sig.\n")
            legend_items.append("• Red: Inconclusive\n\n")
            
            legend_items.append("Decision Rules:\n")
            legend_items.append("1. CI entirely > SESOI → Meaningful\n")
            legend_items.append("2. CI within TOST → Negligible\n")
            legend_items.append("3. CI crosses bounds → Check tests\n\n")
            
            # Add actual results
            legend_items.append("RESULTS\n" + "="*20 + "\n")
            for pred, interp in zip(predictors, interpretations):
                legend_items.append(f"{pred}:\n  {interp}\n\n")
            
            legend_text = ''.join(legend_items)
            ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle('Wald Method: Combined SESOI & TOST Analysis', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save or return
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Wald forest plot saved to {save_path}")
                return save_path
            else:
                return fig
            
        except Exception as e:
            logger.error(f"Failed to create Wald forest plot: {e}")
            return None
    
    def generate_wald_report(self, 
                            wald_results: Dict[str, Any],
                            output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive Wald method report.
        
        Args:
            wald_results: Results from run_wald_tests()
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with report content and file paths
        """
        report = {
            'method': 'Wald Method (SESOI & TOST)',
            'tests_performed': ['Wald-SESOI', 'Wald-TOST'],
            'results': wald_results,
            'summary': self._summarize_results(wald_results)
        }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate forest plot
            plot_path = self.create_forest_plot(wald_results, output_dir)
            if plot_path:
                report['visualization'] = str(plot_path)
            
            # Save JSON report
            import json
            json_path = output_dir / "wald_method_results.json"
            with open(json_path, 'w') as f:
                # Custom serializer for numpy types
                def default(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    return obj
                
                json.dump(report, f, indent=2, default=default)
            
            report['json_report'] = str(json_path)
            logger.info(f"Wald method report saved to {json_path}")
        
        return report
    
    def _summarize_results(self, wald_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize Wald test results.
        
        Args:
            wald_results: Results from run_wald_tests()
            
        Returns:
            Summary dictionary
        """
        summary = {
            'n_predictors_tested': 0,
            'n_exceeding_sesoi': 0,
            'n_equivalent_tost': 0,
            'n_significant_meaningful': 0,
            'predictors': []
        }
        
        for name, result in wald_results.items():
            if 'joint' not in name and 'summary' in result:
                summary['n_predictors_tested'] += 1
                
                if result['summary'].get('exceeds_sesoi', False):
                    summary['n_exceeding_sesoi'] += 1
                
                if result['summary'].get('is_equivalent', False):
                    summary['n_equivalent_tost'] += 1
                
                interp = result['summary'].get('interpretation', '')
                if 'Practically and statistically significant' in interp:
                    summary['n_significant_meaningful'] += 1
                
                summary['predictors'].append({
                    'name': name,
                    'percent_change': result['summary'].get('percent_change', 0),
                    'interpretation': interp
                })
        
        # Overall conclusion
        if summary['n_significant_meaningful'] > 0:
            summary['overall_conclusion'] = "Evidence for meaningful biodiversity effects"
        elif summary['n_equivalent_tost'] == summary['n_predictors_tested']:
            summary['overall_conclusion'] = "All effects are negligible"
        else:
            summary['overall_conclusion'] = "Mixed or inconclusive results"
        
        return summary
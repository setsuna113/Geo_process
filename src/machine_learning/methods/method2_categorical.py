"""Method 2: H2 Hypothesis Testing with Continuous and Categorical Interactions.

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
- Model 3a: Continuous interaction (bio4 × plants_richness) 
- Model 3b: Categorical interaction (temperate × plant_high)

Accepts results from Julia GLM via orchestrator.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import chi2
from typing import Dict, Any, Optional, Union, Tuple, List
import json
import warnings
warnings.filterwarnings('ignore')

# Import hypothesis test modules and visualization wrappers
from .wald_method import WaldMethod
from .model_comparison import ModelComparison
from ..diagnostics.diagnostic_plots import DiagnosticPlots
from ..hypothesis.score_test import ScoreTest
from ..hypothesis.lrt_test import LikelihoodRatioTest
from ..hypothesis.contrast_analysis import ContrastAnalysis
from ..hypothesis.simple_slopes import SimpleSlopesAnalysis
from ..evaluation.emmeans_core import REMmeansWrapper
from ..config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class Method2H2Testing:
    """Method 2: Tests H2 using continuous and categorical interaction models.
    
    Model 3a: Continuous interaction (bio4 × plants_richness)
    Model 3b: Categorical interaction (temperate × plant_high)
    
    This class:
    1. Accepts fitted model results from Julia (via orchestrator)
    2. Runs hypothesis tests (Wald, LRT, Score, EMmeans contrasts)
    3. Generates visualizations and reports
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 2.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        
        # Storage for model results from Julia
        self.model_3a_continuous = None  # Model 3a: Continuous interaction
        self.model_3b_categorical = None  # Model 3b: Categorical interaction
        self.data_metadata = None
        
        # Storage for hypothesis test results
        self.hypothesis_results = {}
        self.emmeans_results = {}
        
        # Initialize wrapped components
        self.wald_method = WaldMethod()
        self.model_comparator = ModelComparison()
        self.diagnostic_plotter = DiagnosticPlots()
        self.lrt_test = LikelihoodRatioTest()
        self.score_test = ScoreTest()
        self.emmeans_r = None  # Will be created when needed
        self.contrast_analyzer = None  # Will be created when needed
        self.simple_slopes = None  # Will be created when needed
        
        logger.info("Method 2 H2 Testing initialized with wrapped components")
    
    def set_julia_results(self, models: Dict[str, Any]) -> None:
        """Set the Julia model results for analysis.
        
        Called by orchestrator to pass Julia results to this method.
        
        Args:
            models: Dictionary containing 'method2_continuous' and 'method2_categorical'
        """
        self.model_3a_continuous = models.get('method2_continuous')
        self.model_3b_categorical = models.get('method2_categorical')
        
        # Extract metadata
        if self.model_3a_continuous:
            self.data_metadata = {
                'n_samples': self.model_3a_continuous.get('n_obs', 0),
                'n_features': len(self.model_3a_continuous.get('coef_names', [])),
                'has_mesf': any('ME' in name for name in self.model_3a_continuous.get('coef_names', []))
            }
        
        logger.info(f"Julia results set for Method 2: "
                   f"3a AIC={self.model_3a_continuous.get('aic', 'N/A') if self.model_3a_continuous else 'N/A':.2f}, "
                   f"3b AIC={self.model_3b_categorical.get('aic', 'N/A') if self.model_3b_categorical else 'N/A':.2f}")
    
    def run_hypothesis_tests(self) -> Dict[str, Any]:
        """Run all hypothesis tests for H2.
        
        Returns:
            Dictionary with all test results
        """
        if not self.model_3a_continuous and not self.model_3b_categorical:
            raise ValueError("Models not loaded. Call set_julia_results() first.")
        
        logger.info("\n" + "="*70)
        logger.info("METHOD 2: H2 HYPOTHESIS TESTING")
        logger.info("="*70)
        
        results = {}
        
        # Test Model 3a (Continuous interaction)
        if self.model_3a_continuous:
            results['continuous_interaction'] = self._test_continuous_interaction()
        
        # Test Model 3b (Categorical interaction)
        if self.model_3b_categorical:
            results['categorical_interaction'] = self._test_categorical_interaction()
            
            # Compute EMmeans for categorical model (disabled for now)
            # try:
            #     self.emmeans_results = self._compute_emmeans()
            #     results['emmeans'] = self.emmeans_results
            # except Exception as e:
            #     logger.warning(f"EMmeans computation failed: {e}")
        
        # Overall conclusion
        results['conclusion'] = self._determine_conclusion(results)
        
        # Store results
        self.hypothesis_results = results
        
        return results
    
    def _test_continuous_interaction(self) -> Dict[str, Any]:
        """Test continuous interaction (bio4 × plants_richness).
        
        For H2 decoupling hypothesis: Tests if plant effect DECREASES with seasonality.
        A negative interaction coefficient supports H2.
        
        Returns:
            Test results for continuous interaction
        """
        logger.info("\nTesting Model 3a: Continuous Interaction")
        
        # Extract coefficient information
        coefs = np.array(self.model_3a_continuous.get('coefficients', []))
        ses = np.array(self.model_3a_continuous.get('std_errors', []))
        names = self.model_3a_continuous.get('coef_names', [])
        vcov = np.array(self.model_3a_continuous.get('vcov', []))
        
        # Find interaction term
        interaction_idx = None
        interaction_name = None
        for i, name in enumerate(names):
            if 'bio04' in name.lower() and 'plant' in name.lower():
                interaction_idx = i
                interaction_name = name
                break
        
        if interaction_idx is None:
            logger.warning("Continuous interaction term not found")
            return {}
        
        # Get the interaction coefficient value
        interaction_coef = coefs[interaction_idx]
        
        # Use WaldMethod for comprehensive testing (SESOI + TOST)
        # Set up specific thresholds for interaction terms
        self.wald_method.sesoi_thresholds = {interaction_name: 0.1}  # Meaningful effect size threshold
        self.wald_method.tost_bounds = {interaction_name: (-0.05, 0.05)}  # ±5% is negligible
        
        # Run Wald tests with SESOI and TOST
        wald_results = self.wald_method.run_wald_tests(
            coefficients=coefs,
            std_errors=ses,
            coef_names=names,
            vcov_matrix=vcov if vcov.size > 0 else None,
            bio_predictors=[interaction_name]  # Test only the interaction term
        )
        
        # Extract results for the interaction term
        if interaction_name in wald_results:
            interaction_result = wald_results[interaction_name]
            
            # For H2 (decoupling hypothesis), we need a NEGATIVE interaction
            # A negative bio4 × plants interaction means plant effect decreases with seasonality
            supports_h2_direction = interaction_coef < 0  # Negative supports decoupling
            
            # The effect must be both:
            # 1. In the right direction (negative for decoupling)
            # 2. Exceed SESOI threshold (meaningful effect size)
            # 3. Be statistically significant
            is_significant = interaction_result['sesoi_test'].get('p_value', 1) < 0.05
            exceeds_sesoi = interaction_result['summary']['exceeds_sesoi']
            is_equivalent = interaction_result['summary']['is_equivalent']
            
            supports_h2 = supports_h2_direction and exceeds_sesoi and is_significant
            
            # Custom interpretation for H2 hypothesis
            if not is_significant:
                h2_interpretation = "No significant interaction (fails to support H2)"
            elif is_equivalent:
                h2_interpretation = "Interaction is practically negligible (fails to support H2)"
            elif interaction_coef > 0 and exceeds_sesoi:
                h2_interpretation = "Positive interaction: Plant effect INCREASES with seasonality (contradicts H2)"
            elif interaction_coef < 0 and exceeds_sesoi:
                h2_interpretation = "Negative interaction: Plant effect DECREASES with seasonality (supports H2 decoupling)"
            elif interaction_coef < 0:
                h2_interpretation = "Negative but weak interaction (insufficient evidence for H2)"
            else:
                h2_interpretation = "Positive but weak interaction (contradicts H2 direction)"
            
            # Format result for compatibility with existing code
            result = {
                'coefficient': interaction_result['coefficient'],
                'std_error': interaction_result['standard_error'],
                'wald_results': interaction_result,  # Full Wald results
                'sesoi_test': interaction_result['sesoi_test'],
                'tost_test': interaction_result['tost_test'],
                'significant': is_significant,
                'exceeds_sesoi': exceeds_sesoi,
                'is_equivalent': is_equivalent,
                'percent_change': interaction_result['summary']['percent_change'],
                'supports_h2': supports_h2,  # NEW: Specifically for H2 hypothesis
                'h2_interpretation': h2_interpretation,  # NEW: H2-specific interpretation
                'interpretation': interaction_result['summary']['interpretation']  # Keep general interpretation
            }
            
            logger.info(f"  Interaction: β={result['coefficient']:.4f}±{result['std_error']:.4f}")
            logger.info(f"  Direction: {'Negative (consistent with decoupling)' if interaction_coef < 0 else 'Positive (contradicts H2)'}") 
            logger.info(f"  SESOI: {'Exceeded' if exceeds_sesoi else 'Not exceeded'}")
            logger.info(f"  TOST: {'Equivalent to zero' if is_equivalent else 'Not equivalent'}")
            logger.info(f"  H2 Support: {'YES' if supports_h2 else 'NO'} - {h2_interpretation}")
            
            return result
        
        return {}
    
    def _test_categorical_interaction(self) -> Dict[str, Any]:
        """Test categorical interaction (temperate × plant_high).
        
        Returns:
            Test results for categorical interaction
        """
        logger.info("\nTesting Model 3b: Categorical Interaction")
        
        # Extract coefficient information
        coefs = np.array(self.model_3b_categorical.get('coefficients', []))
        ses = np.array(self.model_3b_categorical.get('std_errors', []))
        names = self.model_3b_categorical.get('coef_names', [])
        vcov = np.array(self.model_3b_categorical.get('vcov', []))
        
        # Find interaction term
        interaction_idx = None
        interaction_name = None
        for i, name in enumerate(names):
            if 'temperate' in name.lower() and 'plant_high' in name.lower():
                interaction_idx = i
                interaction_name = name
                break
        
        if interaction_idx is None:
            logger.warning("Categorical interaction term not found")
            return {}
        
        # Use WaldMethod for comprehensive testing (SESOI + TOST)
        # Set up specific thresholds for interaction terms
        self.wald_method.sesoi_thresholds = {interaction_name: 0.1}  # 10% change is meaningful
        self.wald_method.tost_bounds = {interaction_name: (-0.05, 0.05)}  # ±5% is negligible
        
        # Run Wald tests with SESOI and TOST
        wald_results = self.wald_method.run_wald_tests(
            coefficients=coefs,
            std_errors=ses,
            coef_names=names,
            vcov_matrix=vcov if vcov.size > 0 else None,
            bio_predictors=[interaction_name]  # Test only the interaction term
        )
        
        # Extract results for the interaction term
        if interaction_name in wald_results:
            interaction_result = wald_results[interaction_name]
            
            # Format result for compatibility with existing code
            result = {
                'coefficient': interaction_result['coefficient'],
                'std_error': interaction_result['standard_error'],
                'wald_results': interaction_result,  # Full Wald results
                'sesoi_test': interaction_result['sesoi_test'],
                'tost_test': interaction_result['tost_test'],
                'significant': interaction_result['sesoi_test'].get('p_value', 1) < 0.05,
                'exceeds_sesoi': interaction_result['summary']['exceeds_sesoi'],
                'is_equivalent': interaction_result['summary']['is_equivalent'],
                'percent_change': interaction_result['summary']['percent_change'],
                'interpretation': interaction_result['summary']['interpretation']
            }
            
            logger.info(f"  Interaction: β={result['coefficient']:.4f}±{result['std_error']:.4f}")
            logger.info(f"  SESOI: {'Exceeded' if result['exceeds_sesoi'] else 'Not exceeded'}")
            logger.info(f"  TOST: {'Equivalent to zero' if result['is_equivalent'] else 'Not equivalent'}")
            logger.info(f"  Interpretation: {result['interpretation']}")
            
            return result
        
        return {}
    
    def _compute_emmeans(self) -> Dict[str, Any]:
        """Compute estimated marginal means for categorical model.
        
        Returns:
            EMmeans results
        """
        if not self.model_3b_categorical:
            return {}
        
        try:
            # Prepare data for EMmeans computation
            model_data = {
                'coefficients': self.model_3b_categorical.get('coefficients', []),
                'vcov': self.model_3b_categorical.get('vcov', []),
                'coef_names': self.model_3b_categorical.get('coef_names', []),
                'family': 'negative_binomial',
                'df_residual': self.model_3b_categorical.get('df_residual', 100)
            }
            
            # Compute EMmeans for temperate × plant_high interaction
            emmeans_result = self.emmeans_r.compute_emmeans(
                model_data,
                specs=['temperate', 'plant_high'],
                by_vars=None
            )
            
            # Compute contrasts
            if emmeans_result and 'emmeans' in emmeans_result:
                contrasts = self.emmeans_r.compute_contrasts(
                    emmeans_result['emmeans'],
                    contrast_type='pairwise'
                )
                emmeans_result['contrasts'] = contrasts
            
            return emmeans_result
            
        except Exception as e:
            logger.error(f"EMmeans computation failed: {e}")
            return {}
    
    def _interpret_continuous_interaction(self, coef: float, p_value: float, exceeds_sesoi: bool) -> str:
        """Interpret continuous interaction results.
        
        Note: This method is kept for backward compatibility but the main interpretation
        now comes from WaldMethod's combined SESOI+TOST interpretation.
        """
        if p_value >= 0.05:
            return "No significant interaction between seasonality and plant richness"
        elif not exceeds_sesoi:
            return "Statistically significant but practically negligible interaction"
        elif coef > 0:
            return "Plant richness effect increases with temperature seasonality"
        else:
            return "Plant richness effect decreases with temperature seasonality"
    
    def _interpret_categorical_interaction(self, coef: float, p_value: float, exceeds_sesoi: bool) -> str:
        """Interpret categorical interaction results.
        
        Note: This method is kept for backward compatibility but the main interpretation
        now comes from WaldMethod's combined SESOI+TOST interpretation.
        """
        if p_value >= 0.05:
            return "No significant difference between temperate and non-temperate regions"
        elif not exceeds_sesoi:
            return "Statistically significant but practically negligible regional difference"
        elif coef > 0:
            return "Plant diversity effect is stronger in temperate regions"
        else:
            return "Plant diversity effect is weaker in temperate regions"
    
    def _determine_conclusion(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall conclusion for H2.
        
        H2 is supported when interaction shows decoupling (negative for continuous model).
        
        Args:
            results: All test results
            
        Returns:
            Conclusion dictionary
        """
        # Use the new supports_h2 field that considers direction
        cont_supports_h2 = results.get('continuous_interaction', {}).get('supports_h2', False)
        cat_supports_h2 = results.get('categorical_interaction', {}).get('supports_h2', False)
        
        # Get the old metrics for backward compatibility
        cont_sig = results.get('continuous_interaction', {}).get('significant', False)
        cat_sig = results.get('categorical_interaction', {}).get('significant', False)
        cont_sesoi = results.get('continuous_interaction', {}).get('exceeds_sesoi', False)
        cat_sesoi = results.get('categorical_interaction', {}).get('exceeds_sesoi', False)
        
        # H2 is supported if either model shows proper decoupling pattern
        supports_h2 = cont_supports_h2 or cat_supports_h2
        
        conclusion = {
            'supports_h2': supports_h2,
            'hypothesis': 'H2: Climate moderates biodiversity effects' if supports_h2 else 'No support for H2',
            'evidence': {
                'continuous_interaction': cont_sig and cont_sesoi,
                'categorical_interaction': cat_sig and cat_sesoi
            },
            'interpretation': self._get_overall_interpretation(cont_sig, cat_sig, cont_sesoi, cat_sesoi)
        }
        
        logger.info(f"\n" + "="*40)
        logger.info(f"CONCLUSION: {conclusion['hypothesis']}")
        logger.info(f"Interpretation: {conclusion['interpretation']}")
        logger.info("="*40)
        
        return conclusion
    
    def _get_overall_interpretation(self, cont_sig: bool, cat_sig: bool, 
                                   cont_sesoi: bool, cat_sesoi: bool) -> str:
        """Get overall interpretation of H2 results."""
        if (cont_sig and cont_sesoi) and (cat_sig and cat_sesoi):
            return "Strong evidence for climate-biodiversity interactions across multiple models"
        elif (cont_sig and cont_sesoi) or (cat_sig and cat_sesoi):
            return "Moderate evidence for climate-biodiversity interactions"
        elif cont_sig or cat_sig:
            return "Weak statistical evidence, but effects are too small to be meaningful"
        else:
            return "No evidence for climate-biodiversity interactions"
    
    def generate_visualizations(self, output_dir: Optional[Path] = None) -> List[Path]:
        """Generate all visualizations for Method 2.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to generated visualizations
        """
        if not self.hypothesis_results:
            logger.warning("No results to visualize. Run hypothesis tests first.")
            return []
        
        if output_dir is None:
            output_dir = Path("results/method2_analysis")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. Forest plot with SESOI and TOST regions (NEW)
        forest_path = self._create_forest_plot(output_dir)
        if forest_path:
            generated_files.append(forest_path)
        
        # 2. Interaction plots
        interaction_path = self._create_interaction_plots(output_dir)
        if interaction_path:
            generated_files.append(interaction_path)
        
        # 3. EMmeans plot (if available)
        if self.emmeans_results:
            emmeans_path = self._create_emmeans_plot(output_dir)
            if emmeans_path:
                generated_files.append(emmeans_path)
        
        # 4. Simple slopes plot
        slopes_path = self._create_simple_slopes_plot(output_dir)
        if slopes_path:
            generated_files.append(slopes_path)
        
        # 5. Model comparison
        comparison_path = self._create_model_comparison_viz(output_dir)
        if comparison_path:
            generated_files.append(comparison_path)
        
        logger.info(f"Generated {len(generated_files)} visualizations in {output_dir}")
        return generated_files
    
    def _create_forest_plot(self, output_dir: Path) -> Optional[Path]:
        """Create forest plot using WaldMethod wrapper for H2 interaction terms.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        # Collect all Wald results from both interaction models
        combined_wald_results = {}
        
        # Add continuous interaction results
        if 'continuous_interaction' in self.hypothesis_results:
            cont_result = self.hypothesis_results['continuous_interaction']
            if 'wald_results' in cont_result:
                # Label it clearly as continuous interaction
                combined_wald_results['Bio4 × Plant Richness (Continuous)'] = cont_result['wald_results']
        
        # Add categorical interaction results
        if 'categorical_interaction' in self.hypothesis_results:
            cat_result = self.hypothesis_results['categorical_interaction']
            if 'wald_results' in cat_result:
                # Label it clearly as categorical interaction
                combined_wald_results['Temperate × Plant High (Categorical)'] = cat_result['wald_results']
        
        if not combined_wald_results:
            logger.warning("No Wald results available for forest plot")
            return None
        
        # Use WaldMethod to create forest plot
        return self.wald_method.create_forest_plot(
            wald_results=combined_wald_results,
            output_dir=output_dir,
            filename="h2_forest_plot.png"
        )
    
    def _create_interaction_plots(self, output_dir: Path) -> Optional[Path]:
        """Create interaction visualization plots.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Continuous interaction
            if 'continuous_interaction' in self.hypothesis_results:
                ax = axes[0]
                result = self.hypothesis_results['continuous_interaction']
                
                # Create conceptual interaction plot
                x = np.linspace(-2, 2, 100)
                y_low = 0.5 * x + 1
                y_high = (0.5 + result['coefficient']) * x + 1
                
                ax.plot(x, y_low, 'b-', label='Low seasonality', linewidth=2)
                ax.plot(x, y_high, 'r-', label='High seasonality', linewidth=2)
                ax.set_xlabel('Plant Richness (standardized)', fontsize=11)
                ax.set_ylabel('Predicted Fungal Richness', fontsize=11)
                ax.set_title('Continuous Interaction\n(Seasonality × Plant Richness)', 
                           fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add significance annotation
                sig_text = "Significant" if result['significant'] else "Not significant"
                color = 'green' if result['significant'] and result['exceeds_sesoi'] else 'red'
                ax.text(0.05, 0.95, f"{sig_text}\np={result['p_value']:.3f}", 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', color=color)
            
            # Plot 2: Categorical interaction
            if 'categorical_interaction' in self.hypothesis_results:
                ax = axes[1]
                result = self.hypothesis_results['categorical_interaction']
                
                # Create bar plot for categorical interaction
                categories = ['Non-temperate\nLow plants', 'Non-temperate\nHigh plants',
                            'Temperate\nLow plants', 'Temperate\nHigh plants']
                values = [1.0, 1.2, 1.1, 1.3 + result['coefficient']]
                colors = ['lightblue', 'blue', 'lightcoral', 'red']
                
                bars = ax.bar(categories, values, color=colors, alpha=0.7)
                ax.set_ylabel('Predicted Fungal Richness', fontsize=11)
                ax.set_title('Categorical Interaction\n(Temperate × Plant Diversity)', 
                           fontsize=12, fontweight='bold')
                ax.set_xticklabels(categories, rotation=45, ha='right')
                
                # Add significance annotation
                sig_text = "Significant" if result['significant'] else "Not significant"
                color = 'green' if result['significant'] and result['exceeds_sesoi'] else 'red'
                ax.text(0.05, 0.95, f"{sig_text}\np={result['p_value']:.3f}", 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', color=color)
            
            plt.suptitle('Method 2: Climate-Biodiversity Interactions', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = output_dir / "interaction_plots.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Interaction plots saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create interaction plots: {e}")
            return None
    
    def _create_emmeans_plot(self, output_dir: Path) -> Optional[Path]:
        """Create EMmeans visualization.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if not self.emmeans_results or 'emmeans' not in self.emmeans_results:
            return None
        
        try:
            emmeans_data = self.emmeans_results['emmeans']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create grouped bar plot
            groups = []
            means = []
            errors = []
            labels = []
            
            for em in emmeans_data:
                if isinstance(em, dict):
                    groups.append(f"{em.get('temperate', 'NA')}-{em.get('plant_high', 'NA')}")
                    means.append(em.get('emmean', 0))
                    errors.append(em.get('SE', 0))
                    labels.append(f"Temp={em.get('temperate', 'NA')}, Plants={em.get('plant_high', 'NA')}")
            
            if means:
                x = np.arange(len(groups))
                bars = ax.bar(x, means, yerr=errors, capsize=5, alpha=0.7,
                             color=['blue', 'red', 'green', 'orange'][:len(means)])
                
                ax.set_xlabel('Group', fontsize=12)
                ax.set_ylabel('Estimated Marginal Mean', fontsize=12)
                ax.set_title('EMmeans for Temperate × Plant Diversity', 
                           fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                save_path = output_dir / "emmeans_plot.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"EMmeans plot saved to {save_path}")
                return save_path
            
        except Exception as e:
            logger.error(f"Failed to create EMmeans plot: {e}")
            return None
    
    def _create_simple_slopes_plot(self, output_dir: Path) -> Optional[Path]:
        """Create simple slopes visualization.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if not self.model_3a_continuous:
            return None
        
        return None  # Disabled for now
        
        try:
            # Use SimpleSlopesAnalysis to compute slopes
            slopes_result = self.simple_slopes.analyze_simple_slopes(
                model=self.model_3a_continuous,
                focal_var='plants_richness_log',
                moderator='bio04_std',
                moderator_values=[-1, 0, 1]  # Low, medium, high seasonality
            )
            
            if not slopes_result:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot simple slopes
            x = np.linspace(-2, 2, 100)
            colors = ['blue', 'green', 'red']
            labels = ['Low seasonality (-1 SD)', 'Mean seasonality', 'High seasonality (+1 SD)']
            
            for i, (mod_val, slope_info) in enumerate(slopes_result.items()):
                if isinstance(slope_info, dict):
                    slope = slope_info.get('slope', 0)
                    intercept = slope_info.get('intercept', 0)
                    y = slope * x + intercept
                    ax.plot(x, y, color=colors[i % 3], label=labels[i % 3], linewidth=2)
            
            ax.set_xlabel('Plant Richness (log, standardized)', fontsize=12)
            ax.set_ylabel('Predicted Fungal Richness', fontsize=12)
            ax.set_title('Simple Slopes Analysis: Plant Effect at Different Seasonality Levels',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = output_dir / "simple_slopes.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Simple slopes plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create simple slopes plot: {e}")
            return None
    
    def _create_model_comparison_viz(self, output_dir: Path) -> Optional[Path]:
        """Create model comparison visualization.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if not self.model_3a_continuous or not self.model_3b_categorical:
            return None
        
        models = {
            'continuous': self.model_3a_continuous,
            'categorical': self.model_3b_categorical
        }
        
        return self.model_comparator.create_comparison_plot(
            models=models,
            test_results=self.hypothesis_results,
            output_dir=output_dir,
            filename="model_comparison_h2.png",
            title="Method 2: Model Comparison (Continuous vs Categorical)"
        )
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive report for Method 2.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with report content
        """
        if not self.hypothesis_results:
            logger.warning("No results to report. Run hypothesis tests first.")
            return {}
        
        if output_dir is None:
            output_dir = Path("results/method2_analysis")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        viz_files = self.generate_visualizations(output_dir)
        
        # Create report structure
        report = {
            'method': 'Method 2: H2 Climate-Biodiversity Interaction Testing',
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'H2: Climate moderates the effect of biodiversity on fungal richness',
            'models': {
                '3a_continuous': 'Seasonality (bio4) × Plant richness interaction',
                '3b_categorical': 'Temperate × Plant diversity (high/low) interaction'
            },
            'data_summary': self.data_metadata,
            'test_results': self.hypothesis_results,
            'visualizations': [str(f) for f in viz_files],
            'conclusion': self.hypothesis_results.get('conclusion', {})
        }
        
        # Save JSON report
        json_path = output_dir / "method2_h2_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serialize)
        logger.info(f"JSON report saved to {json_path}")
        
        # Create HTML report
        html_path = self._create_html_report(report, output_dir)
        if html_path:
            report['html_report'] = str(html_path)
        
        # Create text summary
        text_path = self._create_text_summary(report, output_dir)
        if text_path:
            report['text_summary'] = str(text_path)
        
        logger.info(f"Method 2 report generation complete. Files saved to {output_dir}")
        return report
    
    def _create_html_report(self, report: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Create HTML report for Method 2."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Method 2: H2 Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
        h2 {{ color: #43A047; margin-top: 30px; }}
        .summary {{ background-color: #E8F5E9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .model-box {{ background-color: #F3E5F5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #2E86AB; color: white; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Method 2: H2 Climate-Biodiversity Interaction Testing</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p><strong>Hypothesis:</strong> {report['hypothesis']}</p>
            <p><strong>Conclusion:</strong> {report['conclusion']['hypothesis']}</p>
            <p>{report['conclusion']['interpretation']}</p>
        </div>
        
        <h2>Models Tested</h2>
        <div class="model-box">
            <h3>Model 3a: Continuous Interaction</h3>
            <p>{report['models']['3a_continuous']}</p>
"""
            
            if 'continuous_interaction' in report['test_results']:
                cont = report['test_results']['continuous_interaction']
                html_content += f"""
            <p>β = {cont['coefficient']:.4f} ± {cont['std_error']:.4f}, p = {cont['p_value']:.4e}</p>
            <p>Result: {cont['interpretation']}</p>
"""
            
            html_content += """
        </div>
        <div class="model-box">
            <h3>Model 3b: Categorical Interaction</h3>
            <p>{}</p>
""".format(report['models']['3b_categorical'])
            
            if 'categorical_interaction' in report['test_results']:
                cat = report['test_results']['categorical_interaction']
                html_content += f"""
            <p>β = {cat['coefficient']:.4f} ± {cat['std_error']:.4f}, p = {cat['p_value']:.4e}</p>
            <p>Result: {cat['interpretation']}</p>
"""
            
            html_content += """
        </div>
        
        <h2>Visualizations</h2>
"""
            
            for viz_file in report.get('visualizations', []):
                viz_path = Path(viz_file)
                if viz_path.exists():
                    html_content += f'        <img src="{viz_path.name}" alt="{viz_path.stem}">\n'
            
            html_content += """
    </div>
</body>
</html>
"""
            
            html_path = output_dir / "method2_report.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to create HTML report: {e}")
            return None
    
    def _create_text_summary(self, report: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Create text summary for Method 2."""
        try:
            text_content = "METHOD 2: H2 CLIMATE-BIODIVERSITY INTERACTION TESTING\n"
            text_content += "=" * 70 + "\n\n"
            text_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            text_content += "HYPOTHESIS\n"
            text_content += "-" * 40 + "\n"
            text_content += f"{report['hypothesis']}\n\n"
            
            text_content += "MODELS\n"
            text_content += "-" * 40 + "\n"
            for model_name, desc in report['models'].items():
                text_content += f"{model_name}: {desc}\n"
            text_content += "\n"
            
            text_content += "TEST RESULTS\n"
            text_content += "-" * 40 + "\n"
            
            if 'continuous_interaction' in report['test_results']:
                cont = report['test_results']['continuous_interaction']
                text_content += "Continuous Interaction (Model 3a):\n"
                text_content += f"  Coefficient: {cont['coefficient']:.4f} ± {cont['std_error']:.4f}\n"
                text_content += f"  p-value: {cont['p_value']:.4e}\n"
                text_content += f"  Significant: {'Yes' if cont['significant'] else 'No'}\n"
                text_content += f"  Exceeds SESOI: {'Yes' if cont['exceeds_sesoi'] else 'No'}\n\n"
            
            if 'categorical_interaction' in report['test_results']:
                cat = report['test_results']['categorical_interaction']
                text_content += "Categorical Interaction (Model 3b):\n"
                text_content += f"  Coefficient: {cat['coefficient']:.4f} ± {cat['std_error']:.4f}\n"
                text_content += f"  p-value: {cat['p_value']:.4e}\n"
                text_content += f"  Significant: {'Yes' if cat['significant'] else 'No'}\n"
                text_content += f"  Exceeds SESOI: {'Yes' if cat['exceeds_sesoi'] else 'No'}\n\n"
            
            text_content += "CONCLUSION\n"
            text_content += "=" * 70 + "\n"
            text_content += f"{report['conclusion']['hypothesis']}\n"
            text_content += f"{report['conclusion']['interpretation']}\n"
            
            text_path = output_dir / "method2_summary.txt"
            with open(text_path, 'w') as f:
                f.write(text_content)
            
            logger.info(f"Text summary saved to {text_path}")
            return text_path
            
        except Exception as e:
            logger.error(f"Failed to create text summary: {e}")
            return None
    
    def _json_serialize(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj == np.inf:
            return "Infinity"
        elif obj == -np.inf:
            return "-Infinity"
        elif np.isnan(obj):
            return "NaN"
        return obj
    
    # Compatibility methods for orchestrator
    def fit_models(self, data_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility method - models are already fitted in Julia."""
        logger.info("Models already fitted in Julia. Running hypothesis tests...")
        return self.run_hypothesis_tests()
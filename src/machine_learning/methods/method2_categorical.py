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
from typing import Dict, Any, Optional, Union, Tuple, List
import json
import warnings
warnings.filterwarnings('ignore')

# Import hypothesis test modules and visualization wrappers
from .wald_method import WaldMethod
from ..evaluation.model_comparison import ModelComparison
from ..diagnostics.diagnostic_plots import ResidualDiagnostics
# Score test not used in Method2 - removed
from ..interpretation.simple_slopes import analyze_simple_slopes
from ..interpretation.marginal_means import analyze_marginal_means
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
        self.diagnostic_plotter = ResidualDiagnostics()
        # LRT is now done in Julia - removed Python LikelihoodRatioTest
        # Score test not used in Method2 - removed
        
        # Storage for interpretation results
        self.simple_slopes_results = None
        self.marginal_means_results = None
        
        logger.info("Method 2 H2 Testing initialized with wrapped components")
    
    def set_julia_results(self, models: Dict[str, Any], lrt_results: Optional[Dict[str, Any]] = None) -> None:
        """Set the Julia model results for analysis.
        
        Called by orchestrator to pass Julia results to this method.
        
        Args:
            models: Dictionary containing 'method2_continuous' and 'method2_categorical'
            lrt_results: Optional LRT results comparing Method1 vs Method2 models
        """
        self.model_3a_continuous = models.get('method2_continuous')
        self.model_3b_categorical = models.get('method2_categorical')
        
        # Store LRT results from Julia
        self.lrt_continuous = lrt_results.get('lrt_method1_vs_method2a') if lrt_results else None
        self.lrt_categorical = lrt_results.get('lrt_method1_vs_method2b') if lrt_results else None
        
        # Extract metadata
        if self.model_3a_continuous:
            self.data_metadata = {
                'n_samples': self.model_3a_continuous.get('n_obs', 0),
                'n_features': len(self.model_3a_continuous.get('coef_names', [])),
                'has_mesf': any('ME' in name for name in self.model_3a_continuous.get('coef_names', []))
            }
        
        aic_3a = self.model_3a_continuous.get('aic', 0) if self.model_3a_continuous else 0
        aic_3b = self.model_3b_categorical.get('aic', 0) if self.model_3b_categorical else 0
        logger.info(f"Julia results set for Method 2: 3a AIC={aic_3a:.2f}, 3b AIC={aic_3b:.2f}")
        
        if self.lrt_continuous:
            logger.info(f"LRT 3a: χ²={self.lrt_continuous.get('lr_statistic', 0):.2f}, p={self.lrt_continuous.get('p_value', 1):.4e}")
        if self.lrt_categorical:
            logger.info(f"LRT 3b: χ²={self.lrt_categorical.get('lr_statistic', 0):.2f}, p={self.lrt_categorical.get('p_value', 1):.4e}")
    
    def run_hypothesis_tests(self) -> Dict[str, Any]:
        """Run all hypothesis tests for H2.
        
        Uses Julia LRT results and specialized analysis classes.
        
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
        
        # Run Simple Slopes Interpretation for continuous interaction
        if self.model_3a_continuous:
            logger.info("\nRunning Simple Slopes Interpretation for Continuous Interaction...")
            
            try:
                # Get feature names for the model
                feature_names = self.model_3a_continuous.get('coef_names', [])
                
                # Run simple slopes analysis
                self.simple_slopes_results = analyze_simple_slopes(
                    julia_model=self.model_3a_continuous,
                    focal_var='plants_richness_log',
                    moderator_var='bio04_std',
                    interaction_term='bio04_std_x_plants_log',
                    feature_names=feature_names,
                    save_path=None  # Will be set later in visualization
                )
                
                if self.simple_slopes_results:
                    results['simple_slopes'] = self.simple_slopes_results
                    logger.info("Simple slopes interpretation completed")
                    
                    # Log interpretation summary
                    interpretation = self.simple_slopes_results.get('interpretation', {})
                    if interpretation.get('summary'):
                        logger.info(f"Pattern: {interpretation.get('pattern', 'unknown')}")
                        logger.info(f"Summary: {interpretation.get('summary', '')}")
                else:
                    logger.warning("Simple slopes interpretation returned None")
                    
            except Exception as e:
                import traceback
                logger.error(f"Simple slopes interpretation failed: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                self.simple_slopes_results = None
        
        # Run Marginal Means Interpretation for categorical interaction (Model 3b)
        if self.model_3b_categorical:
            logger.info("\nRunning Marginal Means Interpretation for Categorical Interaction...")
            
            try:
                # Get feature names for the model
                feature_names = self.model_3b_categorical.get('coef_names', [])
                
                # Run marginal means analysis
                self.marginal_means_results = analyze_marginal_means(
                    julia_model=self.model_3b_categorical,
                    factor1='temperate',
                    factor2='plant_high',
                    feature_names=feature_names,
                    factor1_levels=[0, 1],
                    factor2_levels=[0, 1],
                    factor1_labels=['Non-temperate', 'Temperate'],
                    factor2_labels=['Low plant diversity', 'High plant diversity'],
                    save_path=None  # Will be set later in visualization
                )
                
                if self.marginal_means_results:
                    results['marginal_means'] = self.marginal_means_results
                    logger.info("Marginal means interpretation completed")
                    
                    # Log interpretation summary
                    interpretation = self.marginal_means_results.get('interpretation', {})
                    if interpretation.get('summary'):
                        logger.info(f"Pattern: {interpretation.get('pattern', 'unknown')}")
                        logger.info(f"Summary: {interpretation.get('summary', '')}")
                        
                    # Store for backward compatibility
                    self.emmeans_results = self.marginal_means_results
                else:
                    logger.warning("Marginal means interpretation returned None")
                    
            except Exception as e:
                import traceback
                logger.error(f"Marginal means interpretation failed: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                self.marginal_means_results = None
                self.emmeans_results = {}
        
        # For Method2, the interaction terms themselves test the H2 hypothesis
        
        # Overall conclusion
        results['conclusion'] = self._determine_conclusion(results)
        
        # Store results
        self.hypothesis_results = results
        
        return results
    
    def _test_continuous_interaction(self) -> Dict[str, Any]:
        """Test continuous interaction using Julia LRT and WaldMethod wrapper.
        
        For H2: Tests if bio4 × plants_richness interaction is significant.
        Uses LRT from Julia comparing Method1 vs Method2a.
        
        Returns:
            Test results for continuous interaction
        """
        logger.info("\nTesting Model 3a: Continuous Interaction")
        
        result = {}
        
        # 1. Use Julia LRT result (primary test)
        if self.lrt_continuous:
            result['lrt'] = self.lrt_continuous
            logger.info(f"  LRT: χ²={self.lrt_continuous['lr_statistic']:.2f}, p={self.lrt_continuous['p_value']:.4e}")
            result['lrt_significant'] = self.lrt_continuous['p_value'] < 0.05
        
        # 2. Use WaldMethod wrapper for professional Wald testing
        # Handle both dict and array formats for coefficients
        coef_data = self.model_3a_continuous.get('coefficients', {})
        se_data = self.model_3a_continuous.get('std_errors', {})
        names = self.model_3a_continuous.get('coef_names', [])
        
        # Convert to arrays matching coef_names order
        if isinstance(coef_data, dict):
            coefs = np.array([coef_data.get(name, 0.0) for name in names])
            ses = np.array([se_data.get(name, 0.0) for name in names])
        else:
            coefs = np.array(coef_data)
            ses = np.array(se_data)
        
        vcov = np.array(self.model_3a_continuous.get('vcov', []))
        
        # Run Wald tests using the wrapper (like Method1)
        wald_results = self.wald_method.run_all_tests(
            coefficients=coefs,
            std_errors=ses,
            coef_names=names,
            vcov_matrix=vcov if vcov.size > 0 else None
        )
        result['wald_results'] = wald_results
        
        # Find and extract interaction term results
        # Looking for: bio04_std_x_plants_log or similar patterns
        interaction_result = None
        interaction_coef = None
        interaction_se = None
        
        # Check if wald_results has 'predictors' key (new structure)
        if 'predictors' in wald_results:
            search_dict = wald_results['predictors']
        else:
            # Fallback to old structure
            search_dict = wald_results
            
        for coef_name, coef_result in search_dict.items():
            # Check for the exact interaction term names from Julia
            if ('bio04_std_x_plants_log' in coef_name or 
                ('bio04' in coef_name.lower() and 'plant' in coef_name.lower()) or 
                ('bio4' in coef_name.lower() and 'plant' in coef_name.lower()) or
                'x_plants' in coef_name.lower() or
                'interaction' in coef_name.lower()):
                interaction_result = coef_result
                interaction_coef = coef_result.get('coefficient', 0)
                interaction_se = coef_result.get('standard_error', 0)
                logger.info(f"Found continuous interaction term: {coef_name}")
                break
        
        if interaction_result:
            # Extract Wald test p-value (prefer SESOI if available, fallback to basic)
            p_value = (interaction_result.get('sesoi_test', {}).get('p_value') or 
                      interaction_result.get('basic_test', {}).get('p_value') or 1.0)
            
            result['interaction_coef'] = interaction_coef
            result['interaction_se'] = interaction_se
            result['wald_p'] = p_value
            result['significant'] = p_value < 0.05
            
            # H2 interpretation based on coefficient sign
            if p_value < 0.05:
                if interaction_coef < 0:
                    result['interpretation'] = "Significant negative interaction: Plant effect decreases with seasonality (supports H2 decoupling)"
                    result['supports_h2'] = True
                else:
                    result['interpretation'] = "Significant positive interaction: Plant effect increases with seasonality (contradicts H2)"
                    result['supports_h2'] = False
            else:
                result['interpretation'] = "No significant interaction effect"
                result['supports_h2'] = False
            
            logger.info(f"  Interaction: β={interaction_coef:.4f}±{interaction_se:.4f}, p={p_value:.4f}")
            logger.info(f"  {result['interpretation']}")
        else:
            logger.warning("No interaction term found in Wald results")
            result['supports_h2'] = False
            result['interpretation'] = "No interaction term detected"
        
        # Store in hypothesis_results for reporting
        self.hypothesis_results['continuous_interaction'] = result
        
        return result
    
    def _test_categorical_interaction(self) -> Dict[str, Any]:
        """Test categorical interaction using Julia LRT and WaldMethod wrapper.
        
        For H2: Tests temperate × plant_high interaction.
        Uses LRT from Julia comparing Method1 vs Method2b.
        
        Returns:
            Test results for categorical interaction
        """
        logger.info("\nTesting Model 3b: Categorical Interaction")
        
        result = {}
        
        # 1. Use Julia LRT result (primary test)
        if self.lrt_categorical:
            result['lrt'] = self.lrt_categorical
            logger.info(f"  LRT: χ²={self.lrt_categorical['lr_statistic']:.2f}, p={self.lrt_categorical['p_value']:.4e}")
            result['lrt_significant'] = self.lrt_categorical['p_value'] < 0.05
        
        # 2. Use WaldMethod wrapper for professional Wald testing
        # Handle both dict and array formats for coefficients
        coef_data = self.model_3b_categorical.get('coefficients', {})
        se_data = self.model_3b_categorical.get('std_errors', {})
        names = self.model_3b_categorical.get('coef_names', [])
        
        # Convert to arrays matching coef_names order
        if isinstance(coef_data, dict):
            coefs = np.array([coef_data.get(name, 0.0) for name in names])
            ses = np.array([se_data.get(name, 0.0) for name in names])
        else:
            coefs = np.array(coef_data)
            ses = np.array(se_data)
        
        vcov = np.array(self.model_3b_categorical.get('vcov', []))
        
        # Run Wald tests using the wrapper (like Method1)
        wald_results = self.wald_method.run_all_tests(
            coefficients=coefs,
            std_errors=ses,
            coef_names=names,
            vcov_matrix=vcov if vcov.size > 0 else None
        )
        result['wald_results'] = wald_results
        
        # Find and extract interaction term results
        # Looking for: temperate_x_plant_high or similar patterns
        interaction_result = None
        interaction_coef = None
        interaction_se = None
        
        # Check if wald_results has 'predictors' key (new structure)
        if 'predictors' in wald_results:
            search_dict = wald_results['predictors']
        else:
            # Fallback to old structure
            search_dict = wald_results
            
        for coef_name, coef_result in search_dict.items():
            # Check for the exact interaction term names from Julia
            if ('temperate_x_plant_high' in coef_name or
                ('temperate' in coef_name.lower() and 'plant' in coef_name.lower()) or 
                'x_plant' in coef_name.lower() or
                'interaction' in coef_name.lower()):
                interaction_result = coef_result
                interaction_coef = coef_result.get('coefficient', 0)
                interaction_se = coef_result.get('standard_error', 0)
                logger.info(f"Found categorical interaction term: {coef_name}")
                break
        
        if interaction_result:
            # Extract Wald test p-value (prefer SESOI if available, fallback to basic)
            p_value = (interaction_result.get('sesoi_test', {}).get('p_value') or 
                      interaction_result.get('basic_test', {}).get('p_value') or 1.0)
            
            result['interaction_coef'] = interaction_coef
            result['interaction_se'] = interaction_se
            result['wald_p'] = p_value
            result['significant'] = p_value < 0.05
            
            # H2 interpretation
            if p_value < 0.05:
                if interaction_coef < 0:
                    result['interpretation'] = "Significant negative interaction: High-plant temperate regions have fewer fungi (supports H2)"
                    result['supports_h2'] = True
                else:
                    result['interpretation'] = "Significant positive interaction: High-plant temperate regions have more fungi (contradicts H2)"
                    result['supports_h2'] = False
            else:
                result['interpretation'] = "No significant interaction effect"
                result['supports_h2'] = False
            
            logger.info(f"  Interaction: β={interaction_coef:.4f}±{interaction_se:.4f}, p={p_value:.4f}")
            logger.info(f"  {result['interpretation']}")
        else:
            logger.warning("No interaction term found in Wald results")
            result['supports_h2'] = False
            result['interpretation'] = "No interaction term detected"
        
        # Store in hypothesis_results for reporting
        self.hypothesis_results['categorical_interaction'] = result
        
        return result
    
    
    
    def _determine_conclusion(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall conclusion for H2.
        
        H2 is supported when interaction shows significant climate moderation.
        
        Args:
            results: All test results
            
        Returns:
            Conclusion dictionary
        """
        # Check support from different tests
        cont_supports_h2 = results.get('continuous_interaction', {}).get('supports_h2', False)
        cat_supports_h2 = results.get('categorical_interaction', {}).get('supports_h2', False)
        
        # Check LRT significance (primary test)
        cont_lrt_sig = results.get('continuous_interaction', {}).get('lrt_significant', False)
        cat_lrt_sig = results.get('categorical_interaction', {}).get('lrt_significant', False)
        
        # Check Wald significance
        cont_wald_sig = results.get('continuous_interaction', {}).get('significant', False)
        cat_wald_sig = results.get('categorical_interaction', {}).get('significant', False)
        
        # H2 is supported if either model shows significant interaction
        supports_h2 = cont_supports_h2 or cat_supports_h2
        
        # Determine strength of evidence
        if cont_lrt_sig and cat_lrt_sig and (cont_supports_h2 or cat_supports_h2):
            interpretation = "Strong evidence for climate-biodiversity interactions (both models significant)"
        elif (cont_lrt_sig and cont_supports_h2) or (cat_lrt_sig and cat_supports_h2):
            interpretation = "Moderate evidence for climate-biodiversity interactions"
        elif cont_lrt_sig or cat_lrt_sig:
            interpretation = "Weak evidence - interaction detected but direction inconsistent with H2"
        else:
            interpretation = "No evidence for climate-biodiversity interactions"
        
        conclusion = {
            'supports_h2': supports_h2,
            'hypothesis': 'H2: Climate moderates biodiversity effects' if supports_h2 else 'No support for H2',
            'evidence': {
                'continuous_lrt': cont_lrt_sig,
                'continuous_wald': cont_wald_sig,
                'categorical_lrt': cat_lrt_sig,
                'categorical_wald': cat_wald_sig
            },
            'interpretation': interpretation
        }
        
        logger.info(f"\n" + "="*40)
        logger.info(f"CONCLUSION: {conclusion['hypothesis']}")
        logger.info(f"Interpretation: {conclusion['interpretation']}")
        logger.info("="*40)
        
        return conclusion
    
    
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
        
        # 1. LRT plots (Method 1 vs Method 2a and Method 1 vs Method 2b)
        lrt_paths = self._create_lrt_plots(output_dir)
        generated_files.extend(lrt_paths)
        
        # 2. Forest plot with SESOI and TOST regions
        forest_path = self._create_forest_plot(output_dir)
        if forest_path:
            generated_files.append(forest_path)
        
        # 3. Interaction plots (uses visualization module)
        interaction_path = self._create_interaction_plots(output_dir)
        if interaction_path:
            generated_files.append(interaction_path)
        
        # 4. Simple slopes plot (if results available)
        slopes_path = self._create_simple_slopes_plot(output_dir)
        if slopes_path:
            generated_files.append(slopes_path)
        
        # 5. Model comparison (uses visualization module)
        comparison_path = self._create_model_comparison_viz(output_dir)
        if comparison_path:
            generated_files.append(comparison_path)
        
        logger.info(f"Generated {len(generated_files)} visualizations in {output_dir}")
        return generated_files
    
    def _create_forest_plot(self, output_dir: Path) -> Optional[Path]:
        """Create specialized Method 2 forest plot showing both models side by side.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        # Prepare results in the format expected by WaldMethod
        method2_results = {}
        
        # Get continuous interaction results
        if 'continuous_interaction' in self.hypothesis_results:
            cont_result = self.hypothesis_results['continuous_interaction']
            if 'wald_results' in cont_result:
                method2_results['continuous'] = cont_result['wald_results']
        
        # Get categorical interaction results
        if 'categorical_interaction' in self.hypothesis_results:
            cat_result = self.hypothesis_results['categorical_interaction']
            if 'wald_results' in cat_result:
                method2_results['categorical'] = cat_result['wald_results']
        
        if not method2_results:
            logger.warning("No Wald results available for forest plot")
            return None
        
        # Use WaldMethod's create_forest_plot with method='method2'
        return self.wald_method.create_forest_plot(
            results=method2_results,
            output_dir=output_dir,
            filename="method2_h2_forest.png",
            method='method2'
        )
    
    def _create_interaction_plots(self, output_dir: Path) -> Optional[Path]:
        """Create interaction plots using new interpretation modules.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            saved_paths = []
            
            # Save marginal means plot if available
            if hasattr(self, 'marginal_means_results') and self.marginal_means_results:
                mm_fig = self.marginal_means_results.get('figure')
                if mm_fig:
                    mm_path = output_dir / "marginal_means_h2.png"
                    mm_fig.savefig(mm_path, dpi=300, bbox_inches='tight')
                    plt.close(mm_fig)
                    saved_paths.append(mm_path)
                    logger.info(f"Marginal means plot saved to {mm_path}")
            
            # Return the first saved path for backward compatibility
            return saved_paths[0] if saved_paths else None
            
        except Exception as e:
            logger.error(f"Failed to create interaction plots: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_lrt_plots(self, output_dir: Path) -> List[Path]:
        """Create LRT visualization plots.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            List of paths to saved plots
        """
        from ..visualization.lrt_plots import visualize_lrt_results
        
        saved_plots = []
        
        # Plot LRT 3a (continuous interaction)
        if self.lrt_continuous:
            try:
                save_path = output_dir / "lrt_method2a_continuous.png"
                fig = visualize_lrt_results(
                    lrt_result=self.lrt_continuous,
                    model_names=['Method 1', 'Method 2a (Continuous)'],
                    save_path=str(save_path)
                )
                if save_path:
                    saved_plots.append(save_path)
                    logger.info(f"LRT 3a plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to create LRT 3a plot: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Plot LRT 3b (categorical interaction)
        if self.lrt_categorical:
            try:
                save_path = output_dir / "lrt_method2b_categorical.png"
                fig = visualize_lrt_results(
                    lrt_result=self.lrt_categorical,
                    model_names=['Method 1', 'Method 2b (Categorical)'],
                    save_path=str(save_path)
                )
                if save_path:
                    saved_plots.append(save_path)
                    logger.info(f"LRT 3b plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to create LRT 3b plot: {e}")
        
        return saved_plots
    
    def _create_simple_slopes_plot(self, output_dir: Path) -> Optional[Path]:
        """Create simple slopes visualization using new interpretation module.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if hasattr(self, 'simple_slopes_results') and self.simple_slopes_results:
            try:
                # The new simple slopes already creates visualizations
                # Just save it to the right location
                save_path = output_dir / "simple_slopes_h2.png"
                
                # Get the figure from results
                fig = self.simple_slopes_results.get('figure')
                if fig:
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Simple slopes plot saved to {save_path}")
                    return save_path
                    
            except Exception as e:
                logger.error(f"Failed to save simple slopes plot: {e}")
        
        return None
    
    def _create_model_comparison_viz(self, output_dir: Path) -> Optional[Path]:
        """Create model comparison visualization using visualization module.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        from ..visualization.model_comparison_plots import plot_model_comparison
        
        if not self.model_3a_continuous or not self.model_3b_categorical:
            return None
        
        # Prepare comparison results with all required fields
        aic_3a = self.model_3a_continuous.get('aic', 0)
        aic_3b = self.model_3b_categorical.get('aic', 0)
        
        # Calculate AIC weights (simplified)
        delta_aic = abs(aic_3a - aic_3b)
        if aic_3a < aic_3b:
            weight_3a = 1 / (1 + np.exp(delta_aic/2))
            weight_3b = 1 - weight_3a
            best = 'Method2a_Continuous'
        else:
            weight_3b = 1 / (1 + np.exp(delta_aic/2))
            weight_3a = 1 - weight_3b
            best = 'Method2b_Categorical'
        
        comparison_results = {
            'models': {
                'Method2a_Continuous': {
                    'aic': aic_3a,
                    'bic': self.model_3a_continuous.get('bic', 0),
                    'deviance': self.model_3a_continuous.get('deviance', 0),
                    'df': self.model_3a_continuous.get('df_residual', 0),
                    'delta_aic': 0 if best == 'Method2a_Continuous' else delta_aic,
                    'aic_weight': weight_3a
                },
                'Method2b_Categorical': {
                    'aic': aic_3b,
                    'bic': self.model_3b_categorical.get('bic', 0),
                    'deviance': self.model_3b_categorical.get('deviance', 0),
                    'df': self.model_3b_categorical.get('df_residual', 0),
                    'delta_aic': 0 if best == 'Method2b_Categorical' else delta_aic,
                    'aic_weight': weight_3b
                }
            },
            'weights': {
                'Method2a_Continuous': {'aic_weight': weight_3a},
                'Method2b_Categorical': {'aic_weight': weight_3b}
            },
            'best_model': best,
            'summary': {
                'Method2a_Continuous': 'Continuous interaction model (bio4 × plants)',
                'Method2b_Categorical': 'Categorical interaction model (temperate × plant_high)'
            }
        }
        
        try:
            save_path = output_dir / "model_comparison_h2.png"
            fig = plot_model_comparison(
                comparison_results=comparison_results,
                title="Method 2: Model Comparison",
                save_path=save_path
            )
            if save_path:
                logger.info(f"Model comparison plot saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to create model comparison plot: {e}")
            return None
    
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
        
        # Create report structure - avoid circular references
        report = {
            'method': 'Method 2: H2 Climate-Biodiversity Interaction Testing',
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'H2: Climate moderates the effect of biodiversity on fungal richness',
            'models': {
                '3a_continuous': 'Seasonality (bio4) × Plant richness interaction',
                '3b_categorical': 'Temperate × Plant diversity (high/low) interaction'
            },
            'data_summary': self.data_metadata,
            'test_results': self._serialize_results(self.hypothesis_results),
            'visualizations': [str(f) for f in viz_files],
            'conclusion': self.hypothesis_results.get('conclusion', {})
        }
        
        # Save JSON report
        json_path = output_dir / "method2_h2_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serialize)
        logger.info(f"JSON report saved to {json_path}")
        
        # Create professional text report using reporting module
        text_path = self._create_text_summary(report, output_dir)
        if text_path:
            report['text_summary'] = str(text_path)
        
        logger.info(f"Method 2 report generation complete. Files saved to {output_dir}")
        return report
    
    
    def _create_text_summary(self, report: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Create professional text report using reporting module.
        
        Similar to Method1, uses the professional reporting functions.
        """
        from ..reporting import (
            generate_wald_publication_report,
            generate_contrast_publication_report,
            generate_simple_slopes_publication_report
        )
        
        try:
            reports = []
            latex_tables = []
            
            # Header
            header = "METHOD 2: H2 CLIMATE-BIODIVERSITY INTERACTION TESTING\n"
            header += "=" * 70 + "\n\n"
            header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            reports.append(header)
            
            # 1. Continuous Interaction Report
            if 'continuous_interaction' in report['test_results']:
                cont = report['test_results']['continuous_interaction']
                if 'wald_results' in cont:
                    wald_report = generate_wald_publication_report(
                        test_results=cont['wald_results'],
                        test_name="Bio4 × Plant Richness (Continuous)",
                        hypothesis="H2: Climate seasonality moderates plant-fungi relationship",
                        sesoi_threshold=0.1,
                        tost_bounds=(-0.05, 0.05)
                    )
                    reports.append(wald_report.get('main_text', ''))
                    
                    if 'latex_table' in wald_report:
                        latex_tables.append(("Continuous Interaction", wald_report['latex_table']))
                    
                    # Add H2-specific interpretation
                    if cont.get('supports_h2'):
                        reports.append(f"\n** H2 INTERPRETATION: {cont.get('h2_interpretation', '')} **\n")
            
            # 2. Categorical Interaction Report  
            if 'categorical_interaction' in report['test_results']:
                cat = report['test_results']['categorical_interaction']
                if 'wald_results' in cat:
                    wald_report = generate_wald_publication_report(
                        test_results=cat['wald_results'],
                        test_name="Temperate × Plant High (Categorical)",
                        hypothesis="H2: Temperate regions show different biodiversity patterns",
                        sesoi_threshold=0.1,
                        tost_bounds=(-0.05, 0.05)
                    )
                    reports.append(wald_report.get('main_text', ''))
                    
                    if 'latex_table' in wald_report:
                        latex_tables.append(("Categorical Interaction", wald_report['latex_table']))
            
            # 3. EMmeans/Contrast Report (if available)
            if self.emmeans_results:
                try:
                    contrast_report = generate_contrast_publication_report(
                        contrast_results=self.emmeans_results,
                        emmeans_results=self.emmeans_results,
                        hypothesis_name="H2 (Temperate Mismatch)"
                    )
                    reports.append(contrast_report.get('main_text', ''))
                    
                    if 'latex_table' in contrast_report:
                        latex_tables.append(("Contrast Analysis", contrast_report['latex_table']))
                except Exception as e:
                    logger.warning(f"Could not generate contrast report: {e}")
            
            # 4. Simple Slopes Report (if available)
            if self.simple_slopes_results:
                try:
                    slopes_report = generate_simple_slopes_publication_report(
                        simple_slopes_results=self.simple_slopes_results,
                        hypothesis_name="H2: Plant effect varies with climate seasonality"
                    )
                    reports.append(slopes_report.get('main_text', ''))
                    
                    if 'latex_table' in slopes_report:
                        latex_tables.append(("Simple Slopes", slopes_report['latex_table']))
                except Exception as e:
                    logger.warning(f"Could not generate simple slopes report: {e}")
            
            # 4. Overall Conclusion
            conclusion_text = "\n\nOVERALL CONCLUSION\n" + "=" * 70 + "\n"
            if 'conclusion' in report:
                conclusion = report['conclusion']
                conclusion_text += f"Decision: {conclusion.get('hypothesis', 'N/A')}\n"
                conclusion_text += f"Interpretation: {conclusion.get('interpretation', 'N/A')}\n"
                
                if 'evidence' in conclusion:
                    conclusion_text += "\nEvidence Summary:\n"
                    for test_type, supported in conclusion['evidence'].items():
                        conclusion_text += f"  {test_type}: {'Supports H2' if supported else 'Does not support H2'}\n"
            reports.append(conclusion_text)
            
            # Combine all reports
            full_report = "\n".join(reports)
            
            # Save main report
            text_path = output_dir / "method2_professional_report.txt"
            with open(text_path, 'w') as f:
                f.write(full_report)
            logger.info(f"Professional report saved to {text_path}")
            
            # Save LaTeX tables separately if any
            if latex_tables:
                latex_path = output_dir / "method2_latex_tables.tex"
                with open(latex_path, 'w') as f:
                    f.write("% LaTeX Tables for Method 2 H2 Results\n\n")
                    for name, table in latex_tables:
                        f.write(f"% {name}\n")
                        f.write(table)
                        f.write("\n\n")
                logger.info(f"LaTeX tables saved to {latex_path}")
            
            return text_path
            
        except Exception as e:
            logger.error(f"Failed to create professional report: {e}")
            return None
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Serialize results to avoid circular references."""
        serialized = {}
        for key, value in results.items():
            if isinstance(value, dict):
                # Recursively serialize nested dicts
                serialized[key] = self._serialize_results(value)
            elif isinstance(value, pd.DataFrame):
                # Convert DataFrames to dict
                serialized[key] = value.to_dict('records')
            elif isinstance(value, np.ndarray):
                # Convert arrays to lists
                serialized[key] = value.tolist()
            elif hasattr(value, '__dict__'):
                # Skip complex objects that might have circular refs
                serialized[key] = str(type(value).__name__)
            else:
                serialized[key] = value
        return serialized
    
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
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, float):
            if np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            elif np.isnan(obj):
                return "NaN"
            return obj
        return obj
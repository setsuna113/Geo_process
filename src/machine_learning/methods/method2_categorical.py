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
        self.diagnostic_plotter = ResidualDiagnostics()
        # LRT is now done in Julia - removed Python LikelihoodRatioTest
        # Score test not used in Method2 - removed
        self.emmeans_r = None  # Will be created when needed
        self.contrast_analyzer = None  # Will be created when needed
        self.simple_slopes = None  # Will be created when needed
        
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
        
        logger.info(f"Julia results set for Method 2: "
                   f"3a AIC={self.model_3a_continuous.get('aic', 'N/A') if self.model_3a_continuous else 'N/A':.2f}, "
                   f"3b AIC={self.model_3b_categorical.get('aic', 'N/A') if self.model_3b_categorical else 'N/A':.2f}")
        
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
            
            # Run ContrastAnalysis if we have the capability
            if self.contrast_analyzer is None and self.emmeans_results:
                try:
                    from ..hypothesis.contrast_analysis import ContrastAnalysis
                    self.contrast_analyzer = ContrastAnalysis(
                        emmeans_wrapper=None,  # We'll use pre-computed emmeans
                        vcov_matrix=np.array(self.model_3b_categorical.get('vcov', []))
                    )
                    contrast_result = self.contrast_analyzer.test_h2_contrast()
                    results['contrast_analysis'] = contrast_result
                    logger.info(f"Contrast analysis: {contrast_result.get('interpretation', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Contrast analysis failed: {e}")
            
            # Run SimpleSlopesAnalysis if we have continuous model
            if self.model_3a_continuous and self.simple_slopes is None:
                try:
                    from ..hypothesis.simple_slopes import SimpleSlopesAnalysis
                    self.simple_slopes = SimpleSlopesAnalysis()
                    slopes_result = self.simple_slopes.analyze_simple_slopes(
                        model=self.model_3a_continuous,
                        focal_var='plants_richness_log',
                        moderator='bio04_std',
                        moderator_values=[-1, 0, 1]  # Low, medium, high seasonality
                    )
                    results['simple_slopes'] = slopes_result
                    # Store for visualization
                    self.simple_slopes_results = slopes_result
                    logger.info("Simple slopes analysis completed")
                except Exception as e:
                    logger.warning(f"Simple slopes analysis failed: {e}")
        
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
        coefs = np.array(self.model_3a_continuous.get('coefficients', []))
        ses = np.array(self.model_3a_continuous.get('std_errors', []))
        names = self.model_3a_continuous.get('coef_names', [])
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
        interaction_result = None
        interaction_coef = None
        interaction_se = None
        
        for coef_name, coef_result in wald_results.items():
            if ('bio04' in coef_name.lower() and 'plant' in coef_name.lower()) or 'interaction' in coef_name.lower():
                interaction_result = coef_result
                interaction_coef = coef_result.get('coefficient', 0)
                interaction_se = coef_result.get('standard_error', 0)
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
        coefs = np.array(self.model_3b_categorical.get('coefficients', []))
        ses = np.array(self.model_3b_categorical.get('std_errors', []))
        names = self.model_3b_categorical.get('coef_names', [])
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
        interaction_result = None
        interaction_coef = None
        interaction_se = None
        
        for coef_name, coef_result in wald_results.items():
            if (('temperate' in coef_name.lower() and 'plant' in coef_name.lower()) or 
                'interaction' in coef_name.lower()):
                interaction_result = coef_result
                interaction_coef = coef_result.get('coefficient', 0)
                interaction_se = coef_result.get('standard_error', 0)
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
                combined_wald_results.update(cont_result['wald_results'])
        
        # Add categorical interaction results
        if 'categorical_interaction' in self.hypothesis_results:
            cat_result = self.hypothesis_results['categorical_interaction']
            if 'wald_results' in cat_result:
                combined_wald_results.update(cat_result['wald_results'])
        
        if not combined_wald_results:
            logger.warning("No Wald results available for forest plot")
            return None
        
        # Use WaldMethod to create forest plot (like Method1)
        return self.wald_method.create_forest_plot(
            results=combined_wald_results,
            output_dir=output_dir,
            filename="h2_forest_plot.png"
        )
    
    def _create_interaction_plots(self, output_dir: Path) -> Optional[Path]:
        """Create interaction visualization plots using visualization module.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            from ..visualization.interaction_plots import create_h2_publication_figure
            
            # Prepare combined results for visualization
            combined_results = {
                'continuous': self.hypothesis_results.get('continuous_interaction'),
                'categorical': self.hypothesis_results.get('categorical_interaction'),
                'emmeans': self.emmeans_results
            }
            
            # Use the professional visualization function
            save_path = create_h2_publication_figure(
                contrast_results=combined_results,
                output_dir=output_dir,
                filename="h2_interaction_plots.png"
            )
            
            if save_path:
                logger.info(f"Interaction plots saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create interaction plots: {e}")
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
                save_path = visualize_lrt_results(
                    lrt_result=self.lrt_continuous,
                    model_names=['Method 1', 'Method 2a (Continuous)'],
                    output_dir=output_dir,
                    filename="lrt_method2a_continuous.png",
                    title="LRT: Method 1 vs Method 2a (Bio4 × Plant Richness)"
                )
                if save_path:
                    saved_plots.append(save_path)
                    logger.info(f"LRT 3a plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to create LRT 3a plot: {e}")
        
        # Plot LRT 3b (categorical interaction)
        if self.lrt_categorical:
            try:
                save_path = visualize_lrt_results(
                    lrt_result=self.lrt_categorical,
                    model_names=['Method 1', 'Method 2b (Categorical)'],
                    output_dir=output_dir,
                    filename="lrt_method2b_categorical.png",
                    title="LRT: Method 1 vs Method 2b (Temperate × Plant Diversity)"
                )
                if save_path:
                    saved_plots.append(save_path)
                    logger.info(f"LRT 3b plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to create LRT 3b plot: {e}")
        
        return saved_plots
    
    def _create_simple_slopes_plot(self, output_dir: Path) -> Optional[Path]:
        """Create simple slopes visualization using visualization module.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        from ..visualization.simple_slopes_plots import visualize_simple_slopes
        
        # Use results from SimpleSlopesAnalysis (already computed in run_hypothesis_tests)
        if hasattr(self, 'simple_slopes_results') and self.simple_slopes_results:
            try:
                save_path = visualize_simple_slopes(
                    slopes_results=self.simple_slopes_results,
                    focal_var='Plant Richness',
                    moderator='Climate (Temperate)',
                    output_dir=output_dir,
                    filename="simple_slopes_h2.png"
                )
                if save_path:
                    logger.info(f"Simple slopes plot saved to {save_path}")
                return save_path
            except Exception as e:
                logger.error(f"Failed to create simple slopes plot: {e}")
        
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
        
        # Prepare comparison results
        comparison_results = {
            'models': {
                'Method2a_Continuous': {
                    'aic': self.model_3a_continuous.get('aic', 0),
                    'bic': self.model_3a_continuous.get('bic', 0),
                    'deviance': self.model_3a_continuous.get('deviance', 0),
                    'df': self.model_3a_continuous.get('df_residual', 0)
                },
                'Method2b_Categorical': {
                    'aic': self.model_3b_categorical.get('aic', 0),
                    'bic': self.model_3b_categorical.get('bic', 0),
                    'deviance': self.model_3b_categorical.get('deviance', 0),
                    'df': self.model_3b_categorical.get('df_residual', 0)
                }
            },
            'best_model': 'Method2a_Continuous' if self.model_3a_continuous.get('aic', float('inf')) < self.model_3b_categorical.get('aic', float('inf')) else 'Method2b_Categorical'
        }
        
        try:
            save_path = plot_model_comparison(
                comparison_results=comparison_results,
                output_dir=output_dir,
                filename="model_comparison_h2.png"
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
            if self.emmeans_results and self.contrast_analyzer:
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
            if self.simple_slopes and hasattr(self.simple_slopes, 'results'):
                try:
                    slopes_report = generate_simple_slopes_publication_report(
                        simple_slopes_results=self.simple_slopes.results,
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
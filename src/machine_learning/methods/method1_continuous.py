"""Method 1: H0-H1 Testing with GLM (Julia-based).

Tests:
- H0: Fungi richness is independent of plant/vertebrate richness
- H1: Biodiversity begets biodiversity (positive correlation exists)

Compares Model 1 (climate only) vs Model 2 (climate + biodiversity).
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
from ..config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class Method1H0H1GLM:
    """Method 1: Tests H0 vs H1 using nested GLM models.
    
    Model 1 (H0): Climate + spatial only
    Model 2 (H1): Climate + spatial + biodiversity
    
    This class:
    1. Accepts fitted model results from Julia (via orchestrator)
    2. Runs hypothesis tests (Wald-SESOI, Wald-TOST, Score)
    3. Generates visualizations and reports
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 1.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        
        # Storage for model results from Julia
        self.model_null = None  # Model 1: Climate only
        self.model_full = None  # Model 2: Climate + biodiversity
        self.lr_test_result = None  # LRT result from Julia
        self.data_metadata = None  # Metadata about the data
        
        # Storage for hypothesis test results
        self.hypothesis_results = {}
        
        # Initialize wrapped components
        self.wald_method = WaldMethod()  # Uses default SESOI and TOST thresholds
        self.model_comparator = ModelComparison()
        self.diagnostic_plotter = DiagnosticPlots()
        self.lrt_test = LikelihoodRatioTest()
        self.score_test = ScoreTest()
        
        logger.info("Method 1 H0-H1 GLM initialized with wrapped components")
    
    def set_julia_results(self, models: Dict[str, Any], lr_test: Optional[Dict] = None) -> None:
        """Set the Julia model results for analysis.
        
        Called by orchestrator to pass Julia results to this method.
        
        Args:
            models: Dictionary containing 'method1_null' and 'method1_full' model results
            lr_test: Optional pre-computed LRT result from Julia
        """
        self.model_null = models.get('method1_null')
        self.model_full = models.get('method1_full')
        self.lr_test_result = lr_test
        
        # Extract metadata
        if self.model_full:
            self.data_metadata = {
                'n_samples': self.model_full.get('n_obs', 0),
                'n_features': len(self.model_full.get('coef_names', [])),
                'has_mesf': any('ME' in name for name in self.model_full.get('coef_names', []))
            }
        
        logger.info(f"Julia results set: null model AIC={self.model_null.get('aic', 'N/A'):.2f}, "
                   f"full model AIC={self.model_full.get('aic', 'N/A'):.2f}")
    
    def run_hypothesis_tests(self) -> Dict[str, Any]:
        """Run all hypothesis tests on the loaded models.
        
        Returns:
            Dictionary with all test results
        """
        if not self.model_null or not self.model_full:
            raise ValueError("Models not loaded. Call set_julia_results() first.")
        
        logger.info("\n" + "="*70)
        logger.info("METHOD 1: HYPOTHESIS TESTING")
        logger.info("="*70)
        
        results = {}
        
        # 1. Likelihood Ratio Test (if not pre-computed)
        if self.lr_test_result:
            results['likelihood_ratio_test'] = self.lr_test_result
            logger.info(f"LRT (from Julia): χ²={self.lr_test_result.get('lr_statistic', 0):.2f}, "
                       f"p={self.lr_test_result.get('p_value', 0):.4e}")
        else:
            # Compute LRT from log-likelihoods
            null_loglik = self.model_null.get('loglik', 0)
            full_loglik = self.model_full.get('loglik', 0)
            lr_statistic = 2 * (full_loglik - null_loglik)
            df_diff = self.model_null.get('df_residual', 0) - self.model_full.get('df_residual', 0)
            p_value = 1 - chi2.cdf(lr_statistic, df_diff) if df_diff > 0 else 1.0
            
            results['likelihood_ratio_test'] = {
                'lr_statistic': lr_statistic,
                'df': df_diff,
                'p_value': p_value,
                'reject_h0': p_value < 0.05
            }
            logger.info(f"LRT (computed): χ²={lr_statistic:.2f}, p={p_value:.4e}")
        
        # 2. Score Test (approximated from deviance)
        null_deviance = self.model_null.get('deviance', 0)
        full_deviance = self.model_full.get('deviance', 0)
        deviance_diff = null_deviance - full_deviance
        df_diff = self.model_null.get('df_residual', 0) - self.model_full.get('df_residual', 0)
        
        if df_diff > 0 and deviance_diff > 0:
            score_p_value = 1 - chi2.cdf(deviance_diff, df_diff)
            results['score_test'] = {
                'statistic': deviance_diff,
                'df': df_diff,
                'p_value': score_p_value,
                'reject_h0': score_p_value < 0.05,
                'interpretation': 'Testing if biodiversity improves over climate-only model'
            }
            logger.info(f"Score Test: χ²={deviance_diff:.2f}, p={score_p_value:.4e}")
        
        # 3. Wald Tests with SESOI and TOST
        wald_results = self._run_wald_tests()
        results['wald_tests'] = wald_results
        
        # 4. Model Comparison
        comparison = self._compare_models()
        results['model_comparison'] = comparison
        
        # 5. Overall Conclusion
        results['conclusion'] = self._determine_conclusion(results)
        
        # Store results for reporting
        self.hypothesis_results = results
        
        return results
    
    def _run_wald_tests(self) -> Dict[str, Any]:
        """Run Wald tests with SESOI and TOST for biodiversity coefficients.
        
        Returns:
            Dictionary with Wald test results
        """
        # Extract coefficient information
        coefs = np.array(self.model_full.get('coefficients', []))
        ses = np.array(self.model_full.get('std_errors', []))
        names = self.model_full.get('coef_names', [])
        vcov = np.array(self.model_full.get('vcov', []))
        
        # Use WaldMethod wrapper to run tests
        wald_results = self.wald_method.run_wald_tests(
            coefficients=coefs,
            std_errors=ses,
            coef_names=names,
            vcov_matrix=vcov if vcov.size > 0 else None
        )
        
        return wald_results
    
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare null and full models using information criteria.
        
        Returns:
            Model comparison results
        """
        null_aic = self.model_null.get('aic', 0)
        full_aic = self.model_full.get('aic', 0)
        null_bic = self.model_null.get('bic', 0)
        full_bic = self.model_full.get('bic', 0)
        
        delta_aic = null_aic - full_aic
        delta_bic = null_bic - full_bic
        
        # Calculate AIC weights
        aic_values = np.array([full_aic, null_aic])
        delta_aic_weights = aic_values - np.min(aic_values)
        relative_likelihood = np.exp(-0.5 * delta_aic_weights)
        aic_weights = relative_likelihood / np.sum(relative_likelihood)
        
        evidence_ratio = aic_weights[0] / aic_weights[1] if aic_weights[1] > 0 else np.inf
        
        comparison = {
            'null_model': {
                'aic': null_aic,
                'bic': null_bic,
                'n_parameters': self.model_null.get('n_parameters', 0)
            },
            'full_model': {
                'aic': full_aic,
                'bic': full_bic,
                'n_parameters': self.model_full.get('n_parameters', 0)
            },
            'delta_aic': delta_aic,
            'delta_bic': delta_bic,
            'aic_weight_full': aic_weights[0],
            'aic_weight_null': aic_weights[1],
            'evidence_ratio': evidence_ratio,
            'best_model_aic': 'full' if delta_aic > 0 else 'null',
            'best_model_bic': 'full' if delta_bic > 0 else 'null'
        }
        
        logger.info(f"\nModel Comparison:")
        logger.info(f"  ΔAIC: {delta_aic:.2f} ({'full' if delta_aic > 0 else 'null'} model preferred)")
        logger.info(f"  ΔBIC: {delta_bic:.2f} ({'full' if delta_bic > 0 else 'null'} model preferred)")
        logger.info(f"  Evidence ratio: {evidence_ratio:.2e}:1 in favor of full model")
        
        return comparison
    
    def _determine_conclusion(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall conclusion based on all tests.
        
        Args:
            results: All test results
            
        Returns:
            Conclusion dictionary
        """
        # Check LRT
        lrt_supports_h1 = results.get('likelihood_ratio_test', {}).get('reject_h0', False)
        
        # Check Score test
        score_supports_h1 = results.get('score_test', {}).get('reject_h0', False)
        
        # Check Wald tests
        wald_supports_h1 = False
        if 'wald_tests' in results:
            for name, test in results['wald_tests'].items():
                if 'summary' in test and test['summary'].get('exceeds_sesoi', False):
                    wald_supports_h1 = True
                    break
        
        # Check model comparison
        comparison = results.get('model_comparison', {})
        aic_supports_h1 = comparison.get('best_model_aic') == 'full'
        bic_supports_h1 = comparison.get('best_model_bic') == 'full'
        
        # Overall decision (majority vote with emphasis on LRT and Wald)
        evidence_count = sum([lrt_supports_h1, score_supports_h1, wald_supports_h1, aic_supports_h1])
        supports_h1 = evidence_count >= 2  # At least 2 out of 4 support H1
        
        conclusion = {
            'supports_h1': supports_h1,
            'hypothesis': 'H1: Biodiversity begets biodiversity' if supports_h1 else 'H0: No biodiversity effect',
            'evidence': {
                'lrt': lrt_supports_h1,
                'score': score_supports_h1,
                'wald_sesoi': wald_supports_h1,
                'aic': aic_supports_h1,
                'bic': bic_supports_h1
            },
            'interpretation': self._get_interpretation(supports_h1, evidence_count)
        }
        
        logger.info(f"\n" + "="*40)
        logger.info(f"CONCLUSION: {conclusion['hypothesis']}")
        logger.info(f"Interpretation: {conclusion['interpretation']}")
        logger.info("="*40)
        
        return conclusion
    
    def _get_interpretation(self, supports_h1: bool, evidence_count: int) -> str:
        """Get detailed interpretation of results.
        
        Args:
            supports_h1: Whether H1 is supported
            evidence_count: Number of tests supporting H1
            
        Returns:
            Interpretation string
        """
        if supports_h1:
            if evidence_count >= 4:
                return "Strong evidence that biodiversity positively influences fungal richness"
            elif evidence_count == 3:
                return "Good evidence that biodiversity influences fungal richness"
            else:
                return "Moderate evidence for biodiversity effects on fungal richness"
        else:
            if evidence_count == 0:
                return "No evidence for biodiversity effects on fungal richness"
            else:
                return "Insufficient evidence to conclude biodiversity affects fungal richness"
    
    def generate_visualizations(self, output_dir: Optional[Path] = None) -> List[Path]:
        """Generate all visualizations for Method 1.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to generated visualizations
        """
        if not self.hypothesis_results:
            logger.warning("No results to visualize. Run hypothesis tests first.")
            return []
        
        if output_dir is None:
            output_dir = Path("results/method1_analysis")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. Forest Plot
        forest_path = self._create_forest_plot(output_dir)
        if forest_path:
            generated_files.append(forest_path)
        
        # 2. Hypothesis Tests Visualization
        tests_path = self._create_hypothesis_tests_viz(output_dir)
        if tests_path:
            generated_files.append(tests_path)
        
        # 3. Model Comparison Visualization
        comparison_path = self._create_model_comparison_viz(output_dir)
        if comparison_path:
            generated_files.append(comparison_path)
        
        # 4. Diagnostic Plots
        diagnostic_path = self._create_diagnostic_plots(output_dir)
        if diagnostic_path:
            generated_files.append(diagnostic_path)
        
        # 5. LRT Visualization  
        lrt_path = self._create_lrt_viz(output_dir)
        if lrt_path:
            generated_files.append(lrt_path)
        
        # 6. Score Test Visualization
        score_path = self._create_score_test_viz(output_dir)
        if score_path:
            generated_files.append(score_path)
        
        # 7. Feature Importance Plot
        feature_path = self._create_feature_importance_viz(output_dir)
        if feature_path:
            generated_files.append(feature_path)
        
        logger.info(f"Generated {len(generated_files)} visualizations in {output_dir}")
        return generated_files
    
    def _create_forest_plot(self, output_dir: Path) -> Optional[Path]:
        """Create forest plot using WaldMethod wrapper.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if 'wald_tests' not in self.hypothesis_results:
            return None
        
        return self.wald_method.create_forest_plot(
            wald_results=self.hypothesis_results['wald_tests'],
            output_dir=output_dir,
            filename="forest_plot_improved.png"
        )
    
    def _create_hypothesis_tests_viz(self, output_dir: Path) -> Optional[Path]:
        """Create visualization for hypothesis test results.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Test Statistics Summary
            ax = axes[0, 0]
            tests = []
            statistics = []
            p_values = []
            
            if 'likelihood_ratio_test' in self.hypothesis_results:
                tests.append('LRT')
                statistics.append(self.hypothesis_results['likelihood_ratio_test'].get('lr_statistic', 0))
                p_values.append(self.hypothesis_results['likelihood_ratio_test'].get('p_value', 1))
            
            if 'score_test' in self.hypothesis_results:
                tests.append('Score')
                statistics.append(self.hypothesis_results['score_test'].get('statistic', 0))
                p_values.append(self.hypothesis_results['score_test'].get('p_value', 1))
            
            if tests:
                colors = ['green' if p < 0.05 else 'red' for p in p_values]
                bars = ax.bar(tests, statistics, color=colors, alpha=0.7)
                ax.set_ylabel('Test Statistic', fontsize=11)
                ax.set_title('Model Comparison Tests', fontsize=12, fontweight='bold')
                
                # Add p-values as text
                for bar, p in zip(bars, p_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'p={p:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 2. SESOI Results
            ax = axes[0, 1]
            wald_tests = self.hypothesis_results.get('wald_tests', {})
            if wald_tests:
                predictors = []
                exceeds = []
                for name, result in wald_tests.items():
                    if 'joint' not in name and 'summary' in result:
                        predictors.append(name.replace('_log', '').replace('_', ' ').title())
                        exceeds.append(1 if result['summary'].get('exceeds_sesoi', False) else 0)
                
                if predictors:
                    colors = ['green' if e else 'gray' for e in exceeds]
                    ax.bar(predictors, exceeds, color=colors, alpha=0.7)
                    ax.set_ylim([0, 1.2])
                    ax.set_ylabel('Exceeds SESOI', fontsize=11)
                    ax.set_title('SESOI Test Results', fontsize=12, fontweight='bold')
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(['No', 'Yes'])
            
            # 3. TOST Results
            ax = axes[1, 0]
            if wald_tests:
                predictors = []
                conclusions = []
                for name, result in wald_tests.items():
                    if 'joint' not in name and 'tost_test' in result:
                        predictors.append(name.replace('_log', '').replace('_', ' ').title())
                        tost = result['tost_test']
                        if tost.get('is_equivalent'):
                            conclusions.append('Equivalent')
                        elif tost.get('is_different_from_zero'):
                            conclusions.append('Different')
                        else:
                            conclusions.append('Inconclusive')
                
                if predictors:
                    # Create stacked bar chart
                    equiv_counts = [1 if c == 'Equivalent' else 0 for c in conclusions]
                    diff_counts = [1 if c == 'Different' else 0 for c in conclusions]
                    inconcl_counts = [1 if c == 'Inconclusive' else 0 for c in conclusions]
                    
                    x = np.arange(len(predictors))
                    ax.bar(x, equiv_counts, label='Equivalent', color='gray', alpha=0.7)
                    ax.bar(x, diff_counts, bottom=equiv_counts, label='Different', color='green', alpha=0.7)
                    ax.bar(x, inconcl_counts, bottom=np.array(equiv_counts)+np.array(diff_counts), 
                          label='Inconclusive', color='orange', alpha=0.7)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(predictors, rotation=45, ha='right')
                    ax.set_ylabel('TOST Conclusion', fontsize=11)
                    ax.set_title('TOST Equivalence Test Results', fontsize=12, fontweight='bold')
                    ax.legend(loc='upper right')
            
            # 4. Evidence Summary
            ax = axes[1, 1]
            conclusion = self.hypothesis_results.get('conclusion', {})
            evidence = conclusion.get('evidence', {})
            
            if evidence:
                labels = []
                values = []
                colors_pie = []
                for test, supports in evidence.items():
                    labels.append(test.upper())
                    values.append(1 if supports else 0)
                    colors_pie.append('green' if supports else 'lightgray')
                
                if sum(values) > 0:
                    ax.pie(values, labels=labels, colors=colors_pie, autopct='',
                          startangle=90, counterclock=False)
                    ax.set_title('Evidence for H1', fontsize=12, fontweight='bold')
                    
                    # Add text summary
                    support_count = sum(values)
                    total_count = len(values)
                    ax.text(0, -1.5, f'{support_count}/{total_count} tests support H1',
                           ha='center', va='top', fontsize=11, fontweight='bold')
            
            plt.suptitle('Method 1: Hypothesis Test Results', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = output_dir / "hypothesis_tests.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Hypothesis tests visualization saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create hypothesis tests visualization: {e}")
            return None
    
    def _create_model_comparison_viz(self, output_dir: Path) -> Optional[Path]:
        """Create model comparison visualization using ModelComparison wrapper.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if not self.model_null or not self.model_full:
            return None
        
        models = {
            'null': self.model_null,
            'full': self.model_full
        }
        
        # Prepare test results for summary
        test_results = {}
        if 'likelihood_ratio_test' in self.hypothesis_results:
            test_results['lrt'] = self.hypothesis_results['likelihood_ratio_test']
        if 'score_test' in self.hypothesis_results:
            test_results['score'] = self.hypothesis_results['score_test']
        
        return self.model_comparator.create_comparison_plot(
            models=models,
            test_results=test_results,
            output_dir=output_dir,
            filename="model_comparison_improved.png",
            title="Method 1: Model Comparison (H0 vs H1)"
        )
    
    def _create_diagnostic_plots(self, output_dir: Path) -> Optional[Path]:
        """Create diagnostic plots using DiagnosticPlots wrapper.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if not self.model_null or not self.model_full:
            return None
        
        models = {
            'null': self.model_null,
            'full': self.model_full
        }
        
        return self.diagnostic_plotter.create_diagnostic_plots(
            models=models,
            output_dir=output_dir,
            filename="diagnostics_improved.png",
            title="Method 1: Model Diagnostics"
        )
    
    def _create_lrt_viz(self, output_dir: Path) -> Optional[Path]:
        """Create LRT visualization.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if 'likelihood_ratio_test' not in self.hypothesis_results:
            return None
        
        try:
            # Use the LRT visualization method
            fig = self.lrt_test.visualize_lrt(
                test_results=self.hypothesis_results['likelihood_ratio_test'],
                null_model_name="Climate Only (H0)",
                full_model_name="Climate + Biodiversity (H1)"
            )
            
            save_path = output_dir / "lrt_test.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"LRT visualization saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to create LRT visualization: {e}")
            return None
    
    def _create_score_test_viz(self, output_dir: Path) -> Optional[Path]:
        """Create Score test visualization.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if 'score_test' not in self.hypothesis_results:
            return None
        
        try:
            # Use the Score test visualization method
            fig = self.score_test.visualize_score_test(
                test_results=self.hypothesis_results['score_test'],
                model_name="Method 1: Biodiversity Effect"
            )
            
            save_path = output_dir / "score_test.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Score test visualization saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to create Score test visualization: {e}")
            return None
    
    def _create_feature_importance_viz(self, output_dir: Path) -> Optional[Path]:
        """Create feature importance visualization.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if not self.model_full:
            return None
        
        try:
            # Extract coefficient information
            coefs = np.array(self.model_full.get('coefficients', []))
            ses = np.array(self.model_full.get('std_errors', []))
            names = self.model_full.get('coef_names', [])
            
            # Filter out intercept and MESF terms for clarity
            valid_indices = []
            filtered_names = []
            for i, name in enumerate(names):
                if name != '(Intercept)' and not name.startswith('ME'):
                    valid_indices.append(i)
                    # Clean up names
                    clean_name = name.replace('_log', ' (log)').replace('_', ' ').title()
                    filtered_names.append(clean_name)
            
            if not valid_indices:
                return None
            
            filtered_coefs = coefs[valid_indices]
            filtered_ses = ses[valid_indices]
            
            # Calculate importance as absolute standardized coefficients
            importance = np.abs(filtered_coefs / filtered_ses)
            
            # Sort by importance
            sorted_indices = np.argsort(importance)[::-1]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Take top 15 features
            top_n = min(15, len(sorted_indices))
            top_indices = sorted_indices[:top_n]
            
            y_pos = np.arange(top_n)
            importance_values = importance[top_indices]
            feature_names = [filtered_names[i] for i in top_indices]
            
            # Color based on coefficient sign
            colors = ['green' if filtered_coefs[i] > 0 else 'red' for i in top_indices]
            
            bars = ax.barh(y_pos, importance_values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Importance (|z-score|)', fontsize=12)
            ax.set_title('Feature Importance in Full Model', fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, importance_values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}', ha='left', va='center', fontsize=9)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Positive effect'),
                Patch(facecolor='red', alpha=0.7, label='Negative effect')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            save_path = output_dir / "feature_importance.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Feature importance visualization saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create feature importance visualization: {e}")
            return None
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive report for Method 1.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with report content
        """
        if not self.hypothesis_results:
            logger.warning("No results to report. Run hypothesis tests first.")
            return {}
        
        if output_dir is None:
            output_dir = Path("results/method1_analysis")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        viz_files = self.generate_visualizations(output_dir)
        
        # Create report structure
        report = {
            'method': 'Method 1: H0-H1 Hypothesis Testing',
            'timestamp': datetime.now().isoformat(),
            'hypotheses': {
                'H0': 'Fungi richness is independent of plant/vertebrate richness',
                'H1': 'Biodiversity begets biodiversity (positive correlation)'
            },
            'data_summary': {
                'n_observations': self.model_full.get('n_obs', 0),
                'n_features_null': self.model_null.get('n_parameters', 0),
                'n_features_full': self.model_full.get('n_parameters', 0),
                'has_spatial_filtering': self.data_metadata.get('has_mesf', False)
            },
            'test_results': self.hypothesis_results,
            'visualizations': [str(f) for f in viz_files],
            'conclusion': self.hypothesis_results.get('conclusion', {})
        }
        
        # Save JSON report
        json_path = output_dir / "method1_report.json"
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
        
        logger.info(f"Method 1 report generation complete. Files saved to {output_dir}")
        return report
    
    def _create_html_report(self, report: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Create HTML report.
        
        Args:
            report: Report data
            output_dir: Directory to save report
            
        Returns:
            Path to HTML file or None
        """
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Method 1: H0-H1 Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
        h2 {{ color: #43A047; margin-top: 30px; }}
        .summary {{ background-color: #E8F5E9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .conclusion {{ background-color: #FFF3E0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .result {{ font-size: 18px; font-weight: bold; color: #2E7D32; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #2E86AB; color: white; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .evidence {{ display: inline-block; padding: 5px 10px; border-radius: 4px; margin: 2px; }}
        .evidence.support {{ background-color: #C8E6C9; color: #1B5E20; }}
        .evidence.nosupport {{ background-color: #FFCDD2; color: #B71C1C; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Method 1: H0-H1 Hypothesis Testing Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p class="result">{report['conclusion']['hypothesis']}</p>
            <p>{report['conclusion']['interpretation']}</p>
            <p><strong>Sample size:</strong> {report['data_summary']['n_observations']:,} observations</p>
        </div>
        
        <h2>Hypotheses</h2>
        <ul>
            <li><strong>H0 (Null):</strong> {report['hypotheses']['H0']}</li>
            <li><strong>H1 (Alternative):</strong> {report['hypotheses']['H1']}</li>
        </ul>
        
        <h2>Evidence Summary</h2>
        <p>Tests supporting H1:</p>
        <div>
"""
            
            # Add evidence badges
            for test, supports in report['conclusion']['evidence'].items():
                class_name = 'support' if supports else 'nosupport'
                symbol = '✓' if supports else '✗'
                html_content += f'            <span class="evidence {class_name}">{test.upper()} {symbol}</span>\n'
            
            html_content += """
        </div>
        
        <h2>Visualizations</h2>
"""
            
            # Add images
            for viz_file in report.get('visualizations', []):
                viz_path = Path(viz_file)
                if viz_path.exists():
                    html_content += f'        <img src="{viz_path.name}" alt="{viz_path.stem}">\n'
            
            html_content += """
        <div class="conclusion">
            <h2>Conclusion</h2>
            <p>{}</p>
        </div>
        
        <h2>Statistical Details</h2>
        <p>For detailed statistical results, see the accompanying JSON file: method1_report.json</p>
    </div>
</body>
</html>
""".format(report['conclusion']['interpretation'])
            
            # Save HTML
            html_path = output_dir / "method1_report.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to create HTML report: {e}")
            return None
    
    def _create_text_summary(self, report: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Create text summary of results.
        
        Args:
            report: Report data
            output_dir: Directory to save summary
            
        Returns:
            Path to text file or None
        """
        try:
            text_content = "METHOD 1: H0-H1 HYPOTHESIS TESTING RESULTS\n"
            text_content += "=" * 70 + "\n\n"
            text_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            text_content += "HYPOTHESES\n"
            text_content += "-" * 40 + "\n"
            text_content += f"H0: {report['hypotheses']['H0']}\n"
            text_content += f"H1: {report['hypotheses']['H1']}\n\n"
            
            text_content += "DATA SUMMARY\n"
            text_content += "-" * 40 + "\n"
            text_content += f"Sample size: {report['data_summary']['n_observations']:,}\n"
            text_content += f"Null model parameters: {report['data_summary']['n_features_null']}\n"
            text_content += f"Full model parameters: {report['data_summary']['n_features_full']}\n"
            text_content += f"Spatial filtering: {'Yes' if report['data_summary']['has_spatial_filtering'] else 'No'}\n\n"
            
            text_content += "TEST RESULTS\n"
            text_content += "-" * 40 + "\n"
            
            # LRT
            if 'likelihood_ratio_test' in report['test_results']:
                lrt = report['test_results']['likelihood_ratio_test']
                text_content += f"Likelihood Ratio Test:\n"
                text_content += f"  χ² = {lrt.get('lr_statistic', 0):.2f}\n"
                text_content += f"  df = {lrt.get('df', 0)}\n"
                text_content += f"  p-value = {lrt.get('p_value', 1):.4e}\n"
                text_content += f"  Result: {'REJECT H0' if lrt.get('reject_h0', False) else 'FAIL TO REJECT H0'}\n\n"
            
            # Score Test
            if 'score_test' in report['test_results']:
                score = report['test_results']['score_test']
                text_content += f"Score Test:\n"
                text_content += f"  χ² = {score.get('statistic', 0):.2f}\n"
                text_content += f"  df = {score.get('df', 0)}\n"
                text_content += f"  p-value = {score.get('p_value', 1):.4e}\n"
                text_content += f"  Result: {'REJECT H0' if score.get('reject_h0', False) else 'FAIL TO REJECT H0'}\n\n"
            
            # Wald Tests
            if 'wald_tests' in report['test_results']:
                text_content += "Wald Tests (SESOI & TOST):\n"
                for name, result in report['test_results']['wald_tests'].items():
                    if 'summary' in result:
                        text_content += f"  {name}:\n"
                        text_content += f"    Exceeds SESOI: {'Yes' if result['summary'].get('exceeds_sesoi', False) else 'No'}\n"
                        text_content += f"    Equivalent to zero: {'Yes' if result['summary'].get('is_equivalent', False) else 'No'}\n"
                        text_content += f"    Interpretation: {result['summary'].get('interpretation', '')}\n"
            
            text_content += "\nMODEL COMPARISON\n"
            text_content += "-" * 40 + "\n"
            if 'model_comparison' in report['test_results']:
                comp = report['test_results']['model_comparison']
                text_content += f"ΔAIC = {comp.get('delta_aic', 0):.2f} ({'Full' if comp.get('delta_aic', 0) > 0 else 'Null'} model preferred)\n"
                text_content += f"ΔBIC = {comp.get('delta_bic', 0):.2f} ({'Full' if comp.get('delta_bic', 0) > 0 else 'Null'} model preferred)\n"
                text_content += f"Evidence ratio: {comp.get('evidence_ratio', 1):.2e}:1 in favor of full model\n\n"
            
            text_content += "CONCLUSION\n"
            text_content += "=" * 70 + "\n"
            text_content += f"{report['conclusion']['hypothesis']}\n"
            text_content += f"{report['conclusion']['interpretation']}\n\n"
            
            text_content += "Evidence supporting H1:\n"
            for test, supports in report['conclusion']['evidence'].items():
                text_content += f"  {test.upper()}: {'Yes' if supports else 'No'}\n"
            
            # Save text file
            text_path = output_dir / "method1_summary.txt"
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
    
    def get_biodiversity_coefficients(self) -> Dict[str, Any]:
        """Get biodiversity coefficients for reporting.
        
        Returns:
            Dictionary with coefficient information
        """
        if not self.hypothesis_results or 'wald_tests' not in self.hypothesis_results:
            return {}
        
        coefficients = {}
        for name, result in self.hypothesis_results['wald_tests'].items():
            if 'joint' not in name and 'coefficient' in result:
                coefficients[name] = {
                    'coefficient': result['coefficient'],
                    'std_error': result['standard_error'],
                    'p_value': result.get('sesoi_test', {}).get('p_value', np.nan)
                }
        
        return coefficients
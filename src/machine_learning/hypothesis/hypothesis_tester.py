"""Hypothesis testing orchestrator for biodiversity regression analysis.

Based on regression_analysis_decisions.md lines 69-178.
Tests H0, H1, and H2 hypotheses using all three methods.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..research.methods import (
    Method1ContinuousGLM,
    Method2CategoricalGLM,
    Method3DML
)
from ..config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)


class HypothesisTester:
    """Orchestrates hypothesis testing across all methods.
    
    Tests three hypotheses as specified in regression_analysis_decisions.md:
    - H0: Independence (lines 69-85)
    - H1: Biodiversity begets biodiversity (lines 88-119)
    - H2: Temperate mismatch (lines 122-178)
    """
    
    def __init__(self, config: Optional[MLRegressionConfig] = None):
        """Initialize hypothesis tester.
        
        Args:
            config: ML regression configuration
        """
        self.config = config or MLRegressionConfig()
        
        # Initialize all three methods
        self.method1 = Method1ContinuousGLM(config)
        self.method2 = Method2CategoricalGLM(config)
        self.method3 = Method3DML(config)
        
        # Storage for results
        self.data_prepared = None
        self.results = {
            'method1': {},
            'method2': {},
            'method3': {},
            'summary': {}
        }
        
        logger.info("Initialized HypothesisTester with all three methods")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for all methods.
        
        Args:
            data: Raw biodiversity data
            
        Returns:
            Prepared DataFrame
        """
        logger.info("Preparing data for hypothesis testing")
        
        # Use Method 1's preparation (common for all)
        self.data_prepared = self.method1.prepare_data(data)
        
        # Method 2 uses Method 1's prepared data
        self.method2.data_processed = self.data_prepared.copy()
        
        # Method 3 needs its own preparation (creates different columns)
        self.method3.prepare_data(data)
        
        logger.info(f"Data prepared: {len(self.data_prepared)} samples")
        
        return self.data_prepared
    
    def test_method1(self) -> Dict[str, Any]:
        """Run Method 1: Continuous interaction GLM.
        
        Returns:
            Method 1 results
        """
        logger.info("\n" + "="*60)
        logger.info("METHOD 1: CONTINUOUS INTERACTION GLM")
        logger.info("="*60)
        
        # Fit models
        model_results = self.method1.fit_models()
        
        # Get interaction coefficient for H2
        h2_results = self.method1.get_interaction_coefficient()
        
        # Run diagnostics
        diagnostics = self.method1.run_diagnostics()
        
        # Get summary
        summary = self.method1.summarize_results()
        
        self.results['method1'] = {
            'models': model_results,
            'h2_test': h2_results,
            'diagnostics': diagnostics,
            'summary': summary
        }
        
        # Log key findings
        logger.info("\nMethod 1 Results:")
        logger.info(f"  H2 supported: {h2_results['supports_h2']}")
        logger.info(f"  Interaction coef: {h2_results['coefficient']:.4f} "
                   f"(p={h2_results['p_value']:.4f})")
        logger.info(f"  Model adequate: {diagnostics['model_adequate']}")
        
        return self.results['method1']
    
    def test_method2(self) -> Dict[str, Any]:
        """Run Method 2: Categorical GLM with emmeans.
        
        Returns:
            Method 2 results
        """
        logger.info("\n" + "="*60)
        logger.info("METHOD 2: CATEGORICAL GLM WITH EMMEANS")
        logger.info("="*60)
        
        # Prepare categorical data
        self.method2.prepare_data(self.data_prepared)
        
        # Fit model
        model_results = self.method2.fit_model()
        
        # Compute emmeans
        emmeans_results = self.method2.compute_emmeans()
        
        # Test H2 contrast
        h2_results = self.method2.test_h2_contrast()
        
        # Run diagnostics
        diagnostics = self.method2.run_diagnostics()
        
        # Get summary
        summary = self.method2.summarize_results()
        
        self.results['method2'] = {
            'model': model_results,
            'emmeans': emmeans_results,
            'h2_test': h2_results,
            'diagnostics': diagnostics,
            'summary': summary
        }
        
        # Log key findings
        logger.info("\nMethod 2 Results:")
        logger.info(f"  H2 supported: {h2_results['h2_supported']}")
        logger.info(f"  Contrast estimate: {h2_results['contrast_estimate']:.4f} "
                   f"(p={h2_results['p_value']:.4f})")
        logger.info(f"  Model adequate: {diagnostics['model_adequate']}")
        
        return self.results['method2']
    
    def test_method3(self) -> Dict[str, Any]:
        """Run Method 3: Double Machine Learning.
        
        Returns:
            Method 3 results
        """
        logger.info("\n" + "="*60)
        logger.info("METHOD 3: DOUBLE MACHINE LEARNING (DML)")
        logger.info("="*60)
        
        # Test all hypotheses
        hypothesis_results = self.method3.test_all_hypotheses()
        
        # Run robustness checks
        robustness = self.method3.run_robustness_checks()
        
        # Get summary
        summary = self.method3.summarize_results()
        
        self.results['method3'] = {
            'hypotheses': hypothesis_results,
            'robustness': robustness,
            'summary': summary
        }
        
        # Log key findings
        logger.info("\nMethod 3 Results:")
        for hyp, res in hypothesis_results.items():
            logger.info(f"  {hyp.upper()}: effect={res['causal_effect']:.4f} "
                       f"(p={res['p_value']:.4f})")
            logger.info(f"    {res['interpretation']}")
        
        return self.results['method3']
    
    def run_all_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all hypothesis tests.
        
        Args:
            data: Raw biodiversity data
            
        Returns:
            Complete results from all methods
        """
        logger.info("\n" + "="*70)
        logger.info("BIODIVERSITY HYPOTHESIS TESTING - COMPLETE ANALYSIS")
        logger.info("="*70)
        
        # Prepare data
        self.prepare_data(data)
        
        # Run all methods
        self.test_method1()
        self.test_method2()
        self.test_method3()
        
        # Synthesize results
        self.synthesize_results()
        
        # Create visualizations
        self.create_visualizations()
        
        return self.results
    
    def synthesize_results(self) -> Dict[str, Any]:
        """Synthesize results across all methods.
        
        Returns:
            Synthesis of findings
        """
        logger.info("\n" + "="*60)
        logger.info("SYNTHESIS OF RESULTS")
        logger.info("="*60)
        
        synthesis = {}
        
        # H0: Independence test (mainly from Method 3)
        if 'h0' in self.results['method3']['hypotheses']:
            h0_dml = self.results['method3']['hypotheses']['h0']
            synthesis['h0'] = {
                'supported': not h0_dml['significant'],
                'dml_effect': h0_dml['causal_effect'],
                'dml_pvalue': h0_dml['p_value'],
                'interpretation': h0_dml['interpretation']
            }
        
        # H1: Biodiversity begets biodiversity (from Method 3)
        if 'h1' in self.results['method3']['hypotheses']:
            h1_dml = self.results['method3']['hypotheses']['h1']
            synthesis['h1'] = {
                'supported': h1_dml['significant'] and h1_dml['causal_effect'] > 0,
                'dml_effect': h1_dml['causal_effect'],
                'dml_pvalue': h1_dml['p_value'],
                'interpretation': h1_dml['interpretation']
            }
        
        # H2: Temperate mismatch (combine all methods)
        h2_method1 = self.results['method1']['h2_test']['supports_h2']
        h2_method2 = self.results['method2']['h2_test']['h2_supported']
        h2_method3 = False
        if 'h2' in self.results['method3']['hypotheses']:
            h2_dml = self.results['method3']['hypotheses']['h2']
            h2_method3 = h2_dml['significant'] and h2_dml['causal_effect'] < 0
        
        synthesis['h2'] = {
            'method1_supported': h2_method1,
            'method2_supported': h2_method2,
            'method3_supported': h2_method3,
            'consensus': sum([h2_method1, h2_method2, h2_method3]) >= 2,
            'interpretation': self._interpret_h2_consensus(h2_method1, h2_method2, h2_method3)
        }
        
        # Model adequacy
        synthesis['model_adequacy'] = {
            'method1_adequate': self.results['method1']['diagnostics']['model_adequate'],
            'method2_adequate': self.results['method2']['diagnostics']['model_adequate']
        }
        
        # Overall conclusions
        synthesis['conclusions'] = self._draw_conclusions(synthesis)
        
        self.results['summary'] = synthesis
        
        # Log synthesis
        logger.info("\nHYPOTHESIS TESTING SYNTHESIS:")
        logger.info(f"\nH0 (Independence):")
        logger.info(f"  Supported: {synthesis.get('h0', {}).get('supported', 'N/A')}")
        
        logger.info(f"\nH1 (Biodiversity begets biodiversity):")
        logger.info(f"  Supported: {synthesis.get('h1', {}).get('supported', 'N/A')}")
        
        logger.info(f"\nH2 (Temperate mismatch):")
        logger.info(f"  Method 1: {synthesis['h2']['method1_supported']}")
        logger.info(f"  Method 2: {synthesis['h2']['method2_supported']}")
        logger.info(f"  Method 3: {synthesis['h2']['method3_supported']}")
        logger.info(f"  Consensus: {synthesis['h2']['consensus']}")
        logger.info(f"  {synthesis['h2']['interpretation']}")
        
        return synthesis
    
    def create_visualizations(self, output_dir: str = 'outputs/regression_plots') -> Dict[str, str]:
        """Create all visualizations for the regression analysis.
        
        Args:
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary with paths to saved figures
        """
        import os
        from ..visualization import create_regression_synthesis_plot
        
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        # Method 1 visualization
        if hasattr(self, 'method1') and 'method1' in self.results:
            try:
                path = os.path.join(output_dir, 'method1_interaction.png')
                self.method1.create_visualization(save_path=path)
                saved_paths['method1'] = path
                logger.info(f"Saved Method 1 visualization to {path}")
            except Exception as e:
                logger.error(f"Failed to create Method 1 visualization: {e}")
        
        # Method 2 visualization (most comprehensive for H2)
        if hasattr(self, 'method2') and 'method2' in self.results:
            try:
                path = os.path.join(output_dir, 'method2_h2_mismatch.png')
                self.method2.create_visualization(save_path=path)
                saved_paths['method2'] = path
                logger.info(f"Saved Method 2 visualization to {path}")
            except Exception as e:
                logger.error(f"Failed to create Method 2 visualization: {e}")
        
        # Method 3 visualization
        if hasattr(self, 'method3') and 'method3' in self.results:
            try:
                path = os.path.join(output_dir, 'method3_dml_effects.png')
                self.method3.create_visualization(save_path=path)
                saved_paths['method3'] = path
                logger.info(f"Saved Method 3 visualization to {path}")
            except Exception as e:
                logger.error(f"Failed to create Method 3 visualization: {e}")
        
        # Synthesis visualization comparing all methods
        try:
            all_results = {
                'method1': self.results.get('method1', {}),
                'method2': self.results.get('method2', {}),
                'method3': self.results.get('method3', {})
            }
            
            path = os.path.join(output_dir, 'synthesis_comparison.png')
            fig = create_regression_synthesis_plot(all_results, save_path=path)
            saved_paths['synthesis'] = path
            logger.info(f"Saved synthesis visualization to {path}")
        except Exception as e:
            logger.error(f"Failed to create synthesis visualization: {e}")
        
        return saved_paths
    
    def _interpret_h2_consensus(self, m1: bool, m2: bool, m3: bool) -> str:
        """Interpret H2 consensus across methods.
        
        Args:
            m1: Method 1 supports H2
            m2: Method 2 supports H2
            m3: Method 3 supports H2
            
        Returns:
            String interpretation
        """
        support_count = sum([m1, m2, m3])
        
        if support_count == 3:
            return "STRONG EVIDENCE: All methods support temperate mismatch hypothesis"
        elif support_count == 2:
            return "MODERATE EVIDENCE: Majority of methods support temperate mismatch"
        elif support_count == 1:
            return "WEAK EVIDENCE: Only one method supports temperate mismatch"
        else:
            return "NO EVIDENCE: No methods support temperate mismatch hypothesis"
    
    def _draw_conclusions(self, synthesis: Dict[str, Any]) -> List[str]:
        """Draw overall conclusions from synthesis.
        
        Args:
            synthesis: Synthesized results
            
        Returns:
            List of conclusion statements
        """
        conclusions = []
        
        # H0 conclusion
        if 'h0' in synthesis and synthesis['h0']['supported']:
            conclusions.append("Fungi richness appears independent of plant richness after controlling for confounders")
        elif 'h0' in synthesis:
            conclusions.append("Fungi richness is causally influenced by plant richness")
        
        # H1 conclusion
        if 'h1' in synthesis and synthesis['h1']['supported']:
            conclusions.append("Evidence supports 'biodiversity begets biodiversity' hypothesis")
        elif 'h1' in synthesis:
            conclusions.append("No support for positive biodiversity spillovers")
        
        # H2 conclusion
        if synthesis['h2']['consensus']:
            conclusions.append("Temperate regions show decoupling between plant and fungi diversity")
        else:
            conclusions.append("No clear evidence for temperate mismatch in biodiversity relationships")
        
        # Model adequacy
        model_issues = []
        if not synthesis['model_adequacy']['method1_adequate']:
            model_issues.append("Method 1 GLM")
        if not synthesis['model_adequacy']['method2_adequate']:
            model_issues.append("Method 2 GLM")
        
        if model_issues:
            conclusions.append(f"Caution: Model assumption violations in {', '.join(model_issues)}")
        
        return conclusions
    
    def create_visualizations(self, save_dir: Optional[str] = None) -> None:
        """Create visualization of hypothesis testing results.
        
        Args:
            save_dir: Directory to save plots
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Method comparison for H2
        ax = axes[0, 0]
        methods = ['Method 1\n(Continuous)', 'Method 2\n(Categorical)', 'Method 3\n(DML)']
        h2_support = [
            self.results['method1']['h2_test']['supports_h2'],
            self.results['method2']['h2_test']['h2_supported'],
            self.results['method3']['hypotheses'].get('h2', {}).get('significant', False)
        ]
        colors = ['green' if s else 'red' for s in h2_support]
        ax.bar(methods, [1 if s else 0 for s in h2_support], color=colors, alpha=0.6)
        ax.set_ylabel('H2 Support')
        ax.set_title('H2 Hypothesis Support by Method')
        ax.set_ylim([0, 1.2])
        
        # 2. Effect sizes comparison
        ax = axes[0, 1]
        if 'h2' in self.results['method3']['hypotheses']:
            effects = {
                'Method 1': self.results['method1']['h2_test']['coefficient'],
                'Method 3': self.results['method3']['hypotheses']['h2']['causal_effect']
            }
            ax.bar(effects.keys(), effects.values(), color='steelblue', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Effect Size')
            ax.set_title('H2 Effect Sizes')
        
        # 3. P-values comparison
        ax = axes[0, 2]
        pvalues = {
            'Method 1': self.results['method1']['h2_test']['p_value'],
            'Method 2': self.results['method2']['h2_test']['p_value']
        }
        if 'h2' in self.results['method3']['hypotheses']:
            pvalues['Method 3'] = self.results['method3']['hypotheses']['h2']['p_value']
        
        ax.bar(pvalues.keys(), pvalues.values(), color='coral', alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        ax.set_ylabel('P-value')
        ax.set_title('Statistical Significance')
        ax.legend()
        
        # 4. DML causal effects for all hypotheses
        ax = axes[1, 0]
        if self.results['method3']['hypotheses']:
            hyp_names = []
            effects = []
            errors = []
            
            for hyp in ['h0', 'h1', 'h2']:
                if hyp in self.results['method3']['hypotheses']:
                    hyp_names.append(hyp.upper())
                    effects.append(self.results['method3']['hypotheses'][hyp]['causal_effect'])
                    errors.append(self.results['method3']['hypotheses'][hyp]['standard_error'])
            
            x_pos = np.arange(len(hyp_names))
            ax.bar(x_pos, effects, yerr=errors, capsize=5, color='purple', alpha=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(hyp_names)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Causal Effect')
            ax.set_title('DML Causal Effects')
        
        # 5. Model diagnostics summary
        ax = axes[1, 1]
        diagnostic_labels = ['Method 1', 'Method 2']
        adequate = [
            self.results['method1']['diagnostics']['model_adequate'],
            self.results['method2']['diagnostics']['model_adequate']
        ]
        colors = ['green' if a else 'orange' for a in adequate]
        ax.bar(diagnostic_labels, [1 if a else 0.5 for a in adequate], color=colors, alpha=0.6)
        ax.set_ylabel('Model Adequacy')
        ax.set_title('Diagnostic Check Results')
        ax.set_ylim([0, 1.2])
        
        # 6. Overall synthesis
        ax = axes[1, 2]
        if 'summary' in self.results:
            synthesis = self.results['summary']
            hypotheses = []
            support = []
            
            if 'h0' in synthesis:
                hypotheses.append('H0')
                support.append(1 if synthesis['h0']['supported'] else 0)
            if 'h1' in synthesis:
                hypotheses.append('H1')
                support.append(1 if synthesis['h1']['supported'] else 0)
            if 'h2' in synthesis:
                hypotheses.append('H2')
                support.append(1 if synthesis['h2']['consensus'] else 0)
            
            colors = ['green' if s else 'red' for s in support]
            ax.bar(hypotheses, support, color=colors, alpha=0.6)
            ax.set_ylabel('Hypothesis Support')
            ax.set_title('Overall Hypothesis Testing Results')
            ax.set_ylim([0, 1.2])
        
        plt.suptitle('Biodiversity Hypothesis Testing Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'hypothesis_testing_results.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, output_dir: str) -> None:
        """Generate comprehensive report with all outputs.
        
        Args:
            output_dir: Directory to save report files
        """
        import os
        from ..reporting import RegressionReportGenerator
        
        # First create visualizations
        viz_paths = self.create_visualizations(os.path.join(output_dir, 'visualizations'))
        
        # Get all results
        all_results = {
            'method1': self.results.get('method1', {}),
            'method2': self.results.get('method2', {}),
            'method3': self.results.get('method3', {}),
            'synthesis': self.synthesize_results()
        }
        
        # Generate comprehensive report
        report_gen = RegressionReportGenerator(all_results, viz_paths)
        report_gen.generate_full_report(output_dir)
        
        logger.info(f"Full report generated in {output_dir}")
    
    def save_results(self, output_path: str) -> None:
        """Save results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
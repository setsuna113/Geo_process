"""Method 3: Double Machine Learning (Julia-based).

Clean pipeline that uses Julia DML results and calls analysis modules.
Following the same pattern as Method1 and Method2.
"""

import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import matplotlib.pyplot as plt

from ..config_loader import MLRegressionConfig
from ..interpretation.dml_interpreter import DMLEffectInterpreter
from ..hypothesis.dml_magnitude import DMLMagnitude
from ..hypothesis.dml_bayesian import BayesianDML
from ..visualization.dml_magnitude_plots import visualize_dml_magnitude
from ..visualization.bayesian_dml_plots import visualize_bayesian_dml

logger = logging.getLogger(__name__)


class Method3DML:
    """Method 3 - DML pipeline using Julia results.
    
    This class:
    1. Accepts fitted DML results from Julia
    2. Runs hypothesis tests and magnitude analysis
    3. Generates interpretations and visualizations
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 3.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        
        # Storage for Julia results
        self.dml_results = None
        self.data_metadata = None
        
        # Storage for analysis results
        self.hypothesis_results = {}
        self.magnitude_results = None
        self.bayesian_results = None
        
        # Initialize interpreter
        self.interpreter = None
        
        logger.info("Method 3 DML initialized")
    
    def set_julia_results(self, models: Dict[str, Any]) -> None:
        """Set the Julia DML results for analysis.
        
        Called by orchestrator to pass Julia results to this method.
        
        Args:
            models: Dictionary containing 'method3_dml' results
        """
        self.dml_results = models.get('method3_dml')
        
        if not self.dml_results:
            raise ValueError("No method3_dml results found in models")
        
        # Extract metadata
        self.data_metadata = {
            'n_samples': self.dml_results.get('n_obs', 0),
            'n_treatments': len(self.dml_results.get('coef_names', [])),
            'has_bdml': 'bdml_stats' in self.dml_results
        }
        
        # Initialize interpreter with Julia results
        self.interpreter = DMLEffectInterpreter(julia_results=self.dml_results)
        
        logger.info(f"Julia DML results set: {self.data_metadata['n_samples']:,} observations, "
                   f"{self.data_metadata['n_treatments']} treatments")
    
    def run_hypothesis_tests(self) -> Dict[str, Any]:
        """Run hypothesis tests on DML results.
        
        Returns:
            Dictionary with test results
        """
        if not self.dml_results:
            raise ValueError("Julia results not set. Call set_julia_results() first.")
        
        logger.info("\n" + "="*70)
        logger.info("METHOD 3: DML HYPOTHESIS TESTING")
        logger.info("="*70)
        
        results = {}
        
        # 1. Extract pre-computed statistics from Julia
        if 'treatments' in self.dml_results:
            effects = {}
            for name, stats in self.dml_results['treatments'].items():
                effects[name] = {
                    'coefficient': stats['coefficient'],
                    'std_error': stats['std_error'],
                    'z_statistic': stats['z_statistic'],  # Use Julia value
                    'p_value': stats['p_value'],  # Use Julia value
                    'ci_95': stats['ci_95'],  # Use Julia value
                    'significant': stats['statistically_significant']  # Use Julia value
                }
            results['individual_effects'] = effects
            
            # Log key results
            for name, stats in effects.items():
                logger.info(f"{name}: θ={stats['coefficient']:.6f}, Z={stats['z_statistic']:.2f}, "
                          f"p={stats['p_value']:.4f}")
        
        # 2. Joint test (from Julia)
        if 'model_stats' in self.dml_results:
            joint_test = self.dml_results['model_stats'].get('joint_test')
            if joint_test:
                results['joint_test'] = joint_test
                logger.info(f"Joint Wald Test: χ²={joint_test['chi2_statistic']:.2f}, "
                          f"p={joint_test['p_value']:.6f}")
            
            # Condition number warning
            cn = self.dml_results['model_stats'].get('condition_number')
            if cn:
                results['condition_number'] = cn
                if cn > 1e10:
                    logger.warning(f"Condition number {cn:.2e}: severe multicollinearity")
        
        # 3. H0 vs H1 determination
        results['h0_h1_test'] = self._test_h0_h1()
        
        self.hypothesis_results = results
        return results
    
    def run_magnitude_analysis(self) -> Dict[str, Any]:
        """Run magnitude analysis on DML effects.
        
        Returns:
            Magnitude analysis results
        """
        if not self.dml_results:
            raise ValueError("Julia results not set. Call set_julia_results() first.")
        
        logger.info("\nRunning magnitude analysis...")
        
        # Use the wrapped function
        self.magnitude_results = DMLMagnitude.from_julia_results(self.dml_results)
        
        # Log summary
        if self.magnitude_results.get('biodiversity_evaluation'):
            eval = self.magnitude_results['biodiversity_evaluation']
            logger.info(f"Biodiversity assessment: {eval['overall_assessment']}")
        
        return self.magnitude_results
    
    def run_bayesian_analysis(self) -> Dict[str, Any]:
        """Run Bayesian DML analysis if BDML stats available.
        
        Returns:
            Bayesian analysis results or None
        """
        if not self.dml_results:
            raise ValueError("Julia results not set. Call set_julia_results() first.")
        
        if not self.data_metadata.get('has_bdml'):
            logger.info("No BDML statistics available, skipping Bayesian analysis")
            return None
        
        logger.info("\nRunning Bayesian DML analysis...")
        
        # Use the wrapped function
        self.bayesian_results = BayesianDML.from_julia_results(self.dml_results)
        
        # Log summary
        if 'biodiversity_analysis' in self.bayesian_results:
            bio = self.bayesian_results['biodiversity_analysis']
            logger.info(f"P(both effects > 0) = {bio['both_positive']:.1%}")
            logger.info(f"H1 {'supported' if bio['h1_supported'] else 'not supported'}")
        
        return self.bayesian_results
    
    def generate_visualizations(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate all DML visualizations.
        
        Args:
            save_dir: Optional directory to save figures
            
        Returns:
            Dictionary with figure objects
        """
        if not self.dml_results:
            raise ValueError("Julia results not set. Call set_julia_results() first.")
        
        figures = {}
        
        # Magnitude visualization
        if self.magnitude_results:
            logger.info("Generating magnitude visualization...")
            save_path = Path(save_dir) / "dml_magnitude.png" if save_dir else None
            fig_mag = visualize_dml_magnitude(
                self.magnitude_results,
                save_path=str(save_path) if save_path else None
            )
            figures['magnitude'] = fig_mag
            if save_path:
                logger.info(f"Saved magnitude plot to {save_path}")
        
        # Bayesian visualization
        if self.bayesian_results:
            logger.info("Generating Bayesian visualization...")
            save_path = Path(save_dir) / "dml_bayesian.png" if save_dir else None
            fig_bayes = visualize_bayesian_dml(
                self.bayesian_results,
                save_path=str(save_path) if save_path else None
            )
            figures['bayesian'] = fig_bayes
            if save_path:
                logger.info(f"Saved Bayesian plot to {save_path}")
        
        return figures
    
    def _test_h0_h1(self) -> Dict[str, Any]:
        """Test H0 vs H1 based on DML results.
        
        Returns:
            Test outcome and interpretation
        """
        # Extract key effects
        effects = self.hypothesis_results.get('individual_effects', {})
        plants = effects.get('plants_richness', {})
        vertebrates = effects.get('vertebrates_richness', {})
        
        # Determine outcome
        plants_positive = plants.get('coefficient', 0) > 0 and plants.get('significant', False)
        vertebrates_positive = vertebrates.get('coefficient', 0) > 0 and vertebrates.get('significant', False)
        
        if plants_positive and vertebrates_positive:
            h1_supported = True
            interpretation = "Strong support for H1: Both biodiversity components have positive causal effects"
        elif plants_positive or vertebrates_positive:
            h1_supported = True
            interpretation = "Partial support for H1: At least one biodiversity component has positive effect"
        else:
            h1_supported = False
            interpretation = "H0 cannot be rejected: No clear positive biodiversity effects"
        
        return {
            'h1_supported': h1_supported,
            'plants_positive': plants_positive,
            'vertebrates_positive': vertebrates_positive,
            'interpretation': interpretation
        }
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Union[str, List[str]]:
        """Generate comprehensive report of DML results.
        
        Args:
            output_dir: Directory to save report files (optional)
            
        Returns:
            If output_dir provided: List of generated file paths
            If no output_dir: Formatted report string
        """
        if not self.dml_results:
            return "No results available. Run analysis first."
        
        lines = []
        lines.append("=" * 70)
        lines.append("METHOD 3: DOUBLE MACHINE LEARNING RESULTS")
        lines.append("=" * 70)
        lines.append(f"N observations: {self.data_metadata['n_samples']:,}")
        
        # Hypothesis test results
        if self.hypothesis_results:
            lines.append("\nHYPOTHESIS TESTS:")
            
            # Individual effects
            if 'individual_effects' in self.hypothesis_results:
                for name, stats in self.hypothesis_results['individual_effects'].items():
                    lines.append(f"\n{name}:")
                    lines.append(f"  θ = {stats['coefficient']:.6f} (SE = {stats['std_error']:.6f})")
                    lines.append(f"  Z = {stats['z_statistic']:.2f}, p = {stats['p_value']:.4f}")
                    lines.append(f"  95% CI: [{stats['ci_95'][0]:.6f}, {stats['ci_95'][1]:.6f}]")
                    sig_marker = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
                    lines.append(f"  Significant: {stats['significant']} {sig_marker}")
            
            # H0 vs H1
            if 'h0_h1_test' in self.hypothesis_results:
                test = self.hypothesis_results['h0_h1_test']
                lines.append(f"\nH0 vs H1 TEST:")
                lines.append(f"  {test['interpretation']}")
        
        # Magnitude analysis
        if self.magnitude_results:
            lines.append("\nMAGNITUDE ANALYSIS:")
            if 'biodiversity_evaluation' in self.magnitude_results:
                eval = self.magnitude_results['biodiversity_evaluation']
                lines.append(f"  {eval['overall_assessment']}")
        
        # Bayesian analysis
        if self.bayesian_results:
            lines.append("\nBAYESIAN ANALYSIS:")
            if 'biodiversity_analysis' in self.bayesian_results:
                bio = self.bayesian_results['biodiversity_analysis']
                lines.append(f"  {bio['interpretation']}")
        
        lines.append("=" * 70)
        
        report_text = "\n".join(lines)
        
        # If no output directory, return string
        if output_dir is None:
            return report_text
        
        # Save report to files
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        # Save text report
        report_file = output_dir / "method3_dml_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        generated_files.append(str(report_file))
        
        # Save JSON results
        json_file = output_dir / "method3_dml_results.json"
        results_dict = {
            'dml_results': self.dml_results,
            'hypothesis_results': self.hypothesis_results,
            'magnitude_results': self.magnitude_results,
            'bayesian_results': self.bayesian_results,
            'data_metadata': self.data_metadata
        }
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        generated_files.append(str(json_file))
        
        # Generate visualizations
        try:
            from src.machine_learning.visualization.dml_magnitude_plots import visualize_dml_magnitude
            from src.machine_learning.visualization.bayesian_dml_plots import visualize_bayesian_dml
            
            # Run magnitude analysis if not already done
            if not self.magnitude_results and self.dml_results:
                self.magnitude_results = DMLMagnitude.from_julia_results(self.dml_results)
            
            # Run Bayesian analysis if not already done  
            if not self.bayesian_results and self.dml_results:
                self.bayesian_results = BayesianDML.from_julia_results(self.dml_results)
            
            # Magnitude visualization - pass the magnitude results, not raw DML results
            if self.magnitude_results:
                mag_plot_path = output_dir / "dml_magnitude_analysis.png"
                mag_plot = visualize_dml_magnitude(
                    self.magnitude_results,
                    save_path=str(mag_plot_path)
                )
                if mag_plot:
                    generated_files.append(str(mag_plot_path))
                    plt.close(mag_plot)  # Clean up figure
            
            # Bayesian visualization - pass the Bayesian results
            if self.bayesian_results:
                bayes_plot_path = output_dir / "dml_bayesian_analysis.png"
                bayes_plot = visualize_bayesian_dml(
                    self.bayesian_results,
                    save_path=str(bayes_plot_path)
                )
                if bayes_plot:
                    generated_files.append(str(bayes_plot_path))
                    plt.close(bayes_plot)  # Clean up figure
                    
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return generated_files
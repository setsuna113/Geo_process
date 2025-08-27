#!/usr/bin/env python3
"""
Simplified ML Orchestrator - Loads Julia Results from Disk

This is the simplified version that:
1. Loads Julia model results from disk
2. Performs hypothesis testing
3. Runs diagnostics
4. Generates reports

All preprocessing and model fitting is done in Julia.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np

# Configuration and utilities
from src.machine_learning.config_loader import MLRegressionConfig
from src.machine_learning.utils.security_utils import PathValidator

# Import Method classes that have all reporters and visualizers
from src.machine_learning.methods import (
    Method1H0H1GLM,
    Method2H2Testing,
    Method3DML
)

# Note: Hypothesis testing is now handled by Method classes directly

# Diagnostics (will work with Julia residuals if available)
from src.machine_learning.diagnostics import DHARMaDiagnostics

logger = logging.getLogger(__name__)


class MLOrchestrator:
    """Simplified orchestrator that loads Julia results from disk.
    
    All data processing and model fitting is done in Julia.
    This orchestrator only handles analysis and reporting.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize simplified orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            self.config = MLRegressionConfig(config_path)
        else:
            self.config = MLRegressionConfig()
        
        # Initialize components
        self.path_validator = PathValidator()
        
        # Initialize Method classes
        self.method1 = Method1H0H1GLM(self.config)
        self.method2 = Method2H2Testing(self.config)
        self.method3 = Method3DML(self.config)
        
        # Storage for loaded results
        self.julia_results = None
        self.models = {}
        self.hypothesis_results = {}
        self.diagnostic_results = {}
        
        # Results directory - get from config or use default
        julia_dir = self.config.get('regression_analysis', {}).get('output', {}).get('julia_results_dir', '/scratch/yl998/julia')
        self.results_dir = Path(julia_dir)
        
        # Storage for results
        self.results = {
            'method1': {},
            'method2': {},
            'method3': {},
            'synthesis': {}
        }
        
        logger.info("Simplified ML Orchestrator initialized")
    
    def load_julia_results(self, results_file: Optional[str] = None) -> Dict[str, Any]:
        """Load results from Julia pipeline.
        
        Args:
            results_file: Path to Julia results JSON file.
                         If None, loads the latest results.
        
        Returns:
            Dictionary with Julia model results
        """
        logger.info("="*70)
        logger.info("LOADING JULIA RESULTS FROM DISK")
        logger.info("="*70)
        
        if results_file is None:
            # Load latest results
            latest_file = self.results_dir / "julia_results_latest.json"
            if not latest_file.exists():
                # Try looking for any results file
                results_files = list(self.results_dir.glob("julia_results_*.json"))
                if results_files:
                    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
                    logger.info(f"Using most recent results file: {latest_file.name}")
                else:
                    raise FileNotFoundError(
                        f"No Julia results found in {self.results_dir}.\n"
                        f"Run Julia pipeline first:\n"
                        f"  julia julia_complete_pipeline.jl run_pipeline <data.arrow> <step> <mesf> <n_mesf> '' {self.results_dir}"
                    )
            results_file = latest_file
        else:
            results_file = Path(results_file)
        
        logger.info(f"Loading results from: {results_file}")
        
        with open(results_file, 'r') as f:
            self.julia_results = json.load(f)
        
        # Extract models and metadata
        self.models = self.julia_results.get('models', {})
        self.metadata = self.julia_results.get('metadata', {})
        
        logger.info(f"Loaded results with {len(self.models)} models")
        logger.info(f"Dataset: {self.metadata.get('n_rows')} rows, {self.metadata.get('n_cols')} columns")
        
        # Log model summaries
        for model_name, model_data in self.models.items():
            if 'aic' in model_data:
                logger.info(f"  {model_name}: AIC={model_data['aic']:.2f}")
            elif 'r2' in model_data:
                logger.info(f"  {model_name}: R²={model_data['r2']:.4f}")
        
        return self.julia_results
    
    def run_hypothesis_tests(self) -> Dict[str, Any]:
        """Run hypothesis tests on Julia model results.
        
        Returns:
            Dictionary with hypothesis test results
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_julia_results() first.")
        
        logger.info("\n" + "="*70)
        logger.info("HYPOTHESIS TESTING")
        logger.info("="*70)
        
        # Create data metadata for Method classes
        data_metadata = {
            'n_samples': self.models.get('method1_full', {}).get('n_obs', 0),
            'columns': self.models.get('method1_full', {}).get('coef_names', []),
            'has_mesf': any('ME' in name for name in self.models.get('method1_full', {}).get('coef_names', [])),
            'julia_models_loaded': True
        }
        
        # Method 1: H0-H1 Testing
        if 'method1_full' in self.models and 'method1_null' in self.models:
            logger.info("\nMethod 1: H0-H1 Testing (Biodiversity Begets Biodiversity)")
            
            # Pass Julia results to Method1 using the new architecture
            lr_test = self.models.get('method1_lr_test')
            self.method1.set_julia_results(self.models, lr_test)
            
            # Run hypothesis tests with proper SESOI and TOST
            method1_results = self.method1.run_hypothesis_tests()
            self.results['method1'] = method1_results
            
            # Log summary results
            conclusion = method1_results.get('conclusion', {})
            logger.info(f"  Conclusion: {conclusion.get('hypothesis', 'Unknown')}")
            logger.info(f"  Interpretation: {conclusion.get('interpretation', '')}")
            
            # Log test results
            if 'likelihood_ratio_test' in method1_results:
                lrt = method1_results['likelihood_ratio_test']
                logger.info(f"  LRT: χ²={lrt.get('lr_statistic', 0):.2f}, p={lrt.get('p_value', 1):.4e}")
            
            # Log Wald test summaries
            if 'wald_tests' in method1_results:
                for name, result in method1_results['wald_tests'].items():
                    if 'summary' in result:
                        logger.info(f"  {name}: {result['summary'].get('interpretation', '')}")
            
            # Get biodiversity coefficients for additional reporting
            biodiv_coefs = self.method1.get_biodiversity_coefficients()
            if biodiv_coefs:
                for var, info in biodiv_coefs.items():
                    if isinstance(info, dict) and 'coefficient' in info:
                        logger.info(f"  {var}: β={info['coefficient']:.4f}±{info.get('std_error', 0):.4f}")
        
        # Method 2: H2 Testing (Interactions)
        if 'method2_continuous' in self.models or 'method2_categorical' in self.models:
            logger.info("\nMethod 2: H2 Testing (Climate-Diversity Interactions)")
            
            # Prepare LRT results for Method2
            lrt_results = {}
            if 'lrt_method1_vs_method2a' in self.models:
                lrt_results['lrt_method1_vs_method2a'] = self.models['lrt_method1_vs_method2a']
            if 'lrt_method1_vs_method2b' in self.models:
                lrt_results['lrt_method1_vs_method2b'] = self.models['lrt_method1_vs_method2b']
            
            # Pass Julia results to Method2 using set_julia_results
            method2_models = {}
            if 'method2_continuous' in self.models:
                method2_models['method2_continuous'] = self.models['method2_continuous']
            if 'method2_categorical' in self.models:
                method2_models['method2_categorical'] = self.models['method2_categorical']
            # Also pass Method 1 full model for comparison
            if 'method1_full' in self.models:
                method2_models['method1_full'] = self.models['method1_full']
            
            self.method2.set_julia_results(method2_models, lrt_results)
            self.method2.data_metadata = data_metadata
            
            # Run Method 2's comprehensive testing
            # Call run_hypothesis_tests() to get full results including simple slopes and marginal means
            method2_results = self.method2.run_hypothesis_tests()
            
            self.results['method2'] = method2_results
            
            # Log summary
            if 'categorical_interaction' in method2_results:
                cat_test = method2_results['categorical_interaction']
                if cat_test:
                    logger.info(f"  Categorical interaction: β={cat_test.get('estimate', 0):.4f}, "
                              f"p={cat_test.get('wald_test', {}).get('p_value', 1):.4e}")
        
        # Method 3: DML
        if 'method3_dml' in self.models:
            logger.info("\nMethod 3: Causal Effects (DML)")
            
            # Set the model directly on Method3 instance
            self.method3.model = self.models['method3_dml']
            
            # Set Julia results for analysis
            self.method3.set_julia_results(self.models)
            
            # Run hypothesis tests
            method3_results = self.method3.run_hypothesis_tests()
            self.results['method3'] = method3_results
            
            # Log summary results  
            if 'individual_effects' in method3_results:
                for name, stats in method3_results['individual_effects'].items():
                    logger.info(f"  {name}: θ={stats['coefficient']:.4f}±{stats['std_error']:.4f}, p={stats['p_value']:.4f}")
                    logger.info(f"    Significant: {stats['significant']}")
            
            if 'joint_test' in method3_results:
                joint = method3_results['joint_test']
                test_stat = joint.get('chi2_statistic', joint.get('f_statistic', 0))
                logger.info(f"  Joint test: χ²={test_stat:.2f}, p={joint['p_value']:.4e}")
                if joint['p_value'] < 0.05:
                    logger.info(f"    Significant joint effect detected")
                else:
                    logger.info(f"    No significant joint effect")
            
            if 'h0_h1_test' in method3_results:
                h0h1 = method3_results['h0_h1_test']
                logger.info(f"  H0 vs H1: {h0h1.get('interpretation', 'No interpretation available')}")
        
        self.hypothesis_results = self.results
        return self.results
    
    def visualize_results(self, output_dir: Optional[Path] = None) -> None:
        """Generate visualizations for all methods.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = Path("results/visualizations")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nGenerating visualizations in {output_dir}")
        
        # Method 1 visualizations
        if self.method1.model_full and self.method1.model_null:
            try:
                method1_report = self.method1.generate_report(output_dir)
                logger.info(f"Method 1 visualizations generated")
            except Exception as e:
                logger.warning(f"Method 1 visualization failed: {e}")
        
        # Method 2 visualizations  
        if self.method2.model_3b_categorical or self.method2.model_3a_continuous:
            try:
                method2_report = self.method2.generate_report(output_dir)
                logger.info(f"Method 2 visualizations generated")
            except Exception as e:
                logger.warning(f"Method 2 visualization failed: {e}")
        
        # Method 3 visualizations
        if hasattr(self, 'method3') and hasattr(self.method3, 'generate_report'):
            try:
                method3_report = self.method3.generate_report(output_dir)
                logger.info(f"Method 3 visualizations generated")
            except Exception as e:
                logger.warning(f"Method 3 visualization failed: {e}")
    
    def generate_reports(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive reports for all methods.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with all report data
        """
        if output_dir is None:
            output_dir = Path("results/reports")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nGenerating reports in {output_dir}")
        
        all_reports = {}
        
        # Method 1 report
        if self.method1.model_full and self.method1.model_null:
            try:
                all_reports['method1'] = self.method1.generate_report(output_dir)
                logger.info("Method 1 report generated")
            except Exception as e:
                logger.error(f"Method 1 report failed: {e}")
        
        # Method 2 report
        if self.method2.model_3b_categorical or self.method2.model_3a_continuous:
            try:
                all_reports['method2'] = self.method2.generate_report(output_dir)
                logger.info("Method 2 report generated")
            except Exception as e:
                logger.error(f"Method 2 report failed: {e}")
        
        # Method 3 report
        if 'method3_dml' in self.models:
            # Ensure Method 3 is properly initialized with Julia results
            if not hasattr(self.method3, 'model') or not self.method3.model:
                self.method3.model = self.models['method3_dml']
                self.method3.set_julia_results(self.models)
            
            try:
                all_reports['method3'] = self.method3.generate_report(output_dir)
                logger.info("Method 3 report generated")
            except Exception as e:
                logger.error(f"Method 3 report failed: {e}")
        
        # Save combined report
        summary_file = output_dir / "all_methods_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self._make_serializable(all_reports), f, indent=2)
        logger.info(f"Combined report saved to {summary_file}")
        
        return all_reports
    
    def _make_serializable(self, obj):
        """Convert numpy/pandas objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd and isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd and isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def synthesize_results(self) -> Dict[str, Any]:
        """Synthesize results across all methods.
        
        Returns:
            Synthesis of key findings
        """
        logger.info("\n" + "="*70)
        logger.info("RESULTS SYNTHESIS")
        logger.info("="*70)
        
        synthesis = {
            'h0_h1_consensus': None,
            'h2_conclusion': None,
            'causal_effects': None,
            'key_findings': []
        }
        
        # H0-H1 consensus
        if 'method1' in self.results and self.results['method1']:
            if 'h0_h1_conclusion' in self.results['method1']:
                synthesis['h0_h1_consensus'] = self.results['method1']['h0_h1_conclusion']
                if 'Reject H0' in synthesis['h0_h1_consensus']:
                    synthesis['key_findings'].append("Evidence supports biodiversity begets biodiversity hypothesis")
        
        # H2 conclusion
        if 'method2' in self.results and self.results['method2']:
            cont_sig = self.results['method2'].get('continuous_interaction', {}).get('significant', False)
            cat_sig = self.results['method2'].get('categorical_interaction', {}).get('significant', False)
            
            if cont_sig or cat_sig:
                synthesis['h2_conclusion'] = "Climate-diversity interactions detected"
                synthesis['key_findings'].append("Significant climate-biodiversity interactions found")
            else:
                synthesis['h2_conclusion'] = "No significant climate-diversity interactions"
        
        # Causal effects
        if 'method3' in self.results and 'causal_effects' in self.results['method3']:
            effects = self.results['method3']['causal_effects']
            if effects:
                meaningful_effects = [
                    k for k, v in effects.items() 
                    if v.get('sesoi', {}).get('exceeds_sesoi', False)
                ]
                if meaningful_effects:
                    synthesis['causal_effects'] = f"Meaningful causal effects: {', '.join(meaningful_effects)}"
                    synthesis['key_findings'].append(f"Causal analysis identifies {', '.join(meaningful_effects)} as drivers")
        
        # Log synthesis
        logger.info("\nKey Findings:")
        for finding in synthesis['key_findings']:
            logger.info(f"  • {finding}")
        
        if synthesis['h0_h1_consensus']:
            logger.info(f"\nH0-H1 Conclusion: {synthesis['h0_h1_consensus']}")
        if synthesis['h2_conclusion']:
            logger.info(f"H2 Conclusion: {synthesis['h2_conclusion']}")
        if synthesis['causal_effects']:
            logger.info(f"Causal Effects: {synthesis['causal_effects']}")
        
        self.results['synthesis'] = synthesis
        return synthesis
    
    def run_complete_analysis(self,
                             julia_results_file: Optional[str] = None) -> Dict[str, Any]:
        """Run complete analysis pipeline on Julia results.
        
        Args:
            julia_results_file: Path to Julia results (None for latest)
            
        Returns:
            Complete analysis results
        """
        logger.info("="*70)
        logger.info("SIMPLIFIED ML ANALYSIS PIPELINE")
        logger.info("="*70)
        logger.info("All preprocessing and model fitting done in Julia")
        logger.info("Python handles hypothesis testing and reporting only")
        logger.info("="*70)
        
        # Load Julia results
        self.load_julia_results(julia_results_file)
        
        # Run hypothesis tests
        self.run_hypothesis_tests()
        
        # Synthesize results
        self.synthesize_results()
        
        # Generate visualizations - THIS WAS MISSING!
        logger.info("\n" + "="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)
        self.visualize_results(Path("results"))
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("\n" + "="*70)
        logger.info("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        return self.results
    
    def _generate_summary_report(self):
        """Generate a text summary report."""
        report_path = self.results_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BIODIVERSITY ANALYSIS SUMMARY\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Rows: {self.metadata.get('n_rows', 'N/A')}\n")
            f.write(f"Columns: {self.metadata.get('n_cols', 'N/A')}\n")
            f.write(f"MESF eigenvectors: {self.metadata.get('mesf_count', 0)}\n\n")
            
            # Method 1
            if 'method1' in self.results and self.results['method1']:
                f.write("METHOD 1: H0-H1 TESTING\n")
                f.write("-"*40 + "\n")
                if 'lr_test' in self.results['method1']:
                    lr = self.results['method1']['lr_test']
                    f.write(f"LR Test p-value: {lr['p_value']:.4e}\n")
                f.write(f"Conclusion: {self.results['method1'].get('h0_h1_conclusion', 'N/A')}\n\n")
            
            # Method 2
            if 'method2' in self.results and self.results['method2']:
                f.write("METHOD 2: INTERACTION TESTING\n")
                f.write("-"*40 + "\n")
                if 'continuous_interaction' in self.results['method2']:
                    cont = self.results['method2']['continuous_interaction']
                    f.write(f"Continuous interaction: β={cont['estimate']:.4f}, "
                           f"p={cont['wald_test']['p_value']:.4e}\n")
                if 'categorical_interaction' in self.results['method2']:
                    cat = self.results['method2']['categorical_interaction']
                    f.write(f"Categorical interaction: β={cat['estimate']:.4f}, "
                           f"p={cat['wald_test']['p_value']:.4e}\n")
                f.write("\n")
            
            # Method 3
            if 'method3' in self.results and 'causal_effects' in self.results['method3']:
                f.write("METHOD 3: CAUSAL EFFECTS\n")
                f.write("-"*40 + "\n")
                for name, effect in self.results['method3']['causal_effects'].items():
                    f.write(f"{name.capitalize()}: {effect['estimate']:.4f}±{effect['std_error']:.4f}\n")
                f.write("\n")
            
            # Synthesis
            if 'synthesis' in self.results:
                f.write("SYNTHESIS\n")
                f.write("-"*40 + "\n")
                for finding in self.results['synthesis'].get('key_findings', []):
                    f.write(f"• {finding}\n")
        
        logger.info(f"Summary report saved to: {report_path}")
#!/usr/bin/env python3
"""
ML Regression Analysis Runner - Hypothesis testing for biodiversity analysis.

Based on regression_analysis_decisions.md, this script implements:
- H0: No biodiversity interaction beyond climate
- H1: Biodiversity begets biodiversity (continuous interaction)
- H2: Temperate mismatch hypothesis (categorical interaction)
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import ML regression components
from src.machine_learning.config_loader import MLRegressionConfig
from src.machine_learning.research.methods.method1_continuous import Method1ContinuousGLM
from src.machine_learning.research.methods.method2_categorical import Method2CategoricalGLM
from src.machine_learning.research.methods.method3_dml import Method3DML


class MLRegressionRunner:
    """Runner for ML regression analysis with hypothesis testing."""
    
    def __init__(self, config_path: Optional[Path] = None, use_unified: bool = True):
        """Initialize regression runner.
        
        Args:
            config_path: Path to configuration file
            use_unified: Whether to use unified config system
        """
        self.config = MLRegressionConfig(
            config_path=config_path,
            validate=False  # Skip validation for flexibility
        )
        self.logger = self._setup_logging()
        self.results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_file = Path('regression_analysis.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def run_method1_continuous(self, data_path: str, spatial_grid_step: int = 1) -> Dict[str, Any]:
        """Run Method 1: Continuous GLM for H1 hypothesis.
        
        Tests whether biodiversity-biodiversity relationship varies
        continuously with temperature seasonality (bio4).
        """
        self.logger.info("="*60)
        self.logger.info("Method 1: Continuous GLM (H1 Hypothesis)")
        self.logger.info("="*60)
        
        try:
            method1 = Method1ContinuousGLM(self.config)
            
            # Process the dataset with spatial sampling
            self.logger.info(f"Loading data from {data_path}")
            if spatial_grid_step > 1:
                self.logger.info(f"Using spatial grid sampling: every {spatial_grid_step}th point")
            
            # Use process_large_dataset which handles everything
            h2o_data = method1.load_data(data_path, spatial_grid_step=spatial_grid_step)
            h2o_data = method1.prepare_data_h2o(h2o_data)
            results = method1.fit_models(h2o_data)
            
            # Get interaction coefficient results
            interaction_results = method1.get_interaction_coefficient()
            results.update(interaction_results)
            
            # Log results
            self.logger.info(f"H1 Interaction coefficient: {interaction_results['interaction_coefficient']:.6f}")
            self.logger.info(f"H1 P-value: {interaction_results['interaction_p_value']:.6f}")
            self.logger.info(f"H1 Significant: {interaction_results['supports_h2']}")
            
            # Update key names for consistency
            results['interaction_coef'] = interaction_results['interaction_coefficient']
            results['interaction_pvalue'] = interaction_results['interaction_p_value']
            results['h1_supported'] = interaction_results['supports_h2']
            
            return results
            
        except Exception as e:
            self.logger.error(f"Method 1 failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def run_method2_categorical(self, data_path: str, spatial_grid_step: int = 1) -> Dict[str, Any]:
        """Run Method 2: Categorical GLM for H2 hypothesis.
        
        Tests whether biodiversity-biodiversity relationship is stronger
        in temperate regions with high plant diversity.
        """
        self.logger.info("="*60)
        self.logger.info("Method 2: Categorical GLM (H2 Hypothesis)")
        self.logger.info("="*60)
        
        try:
            method2 = Method2CategoricalGLM(self.config)
            
            # Load data with optional spatial sampling
            self.logger.info(f"Loading data from {data_path}")
            if spatial_grid_step > 1:
                self.logger.info(f"Using spatial grid sampling: every {spatial_grid_step}th point")
            h2o_data = method2.load_data(data_path, spatial_grid_step=spatial_grid_step)
            h2o_data = method2.prepare_data_h2o(h2o_data)
            
            # Fit GLM with categorical interaction
            self.logger.info("Fitting GLM with temperate × plant_high interaction...")
            results = method2.fit_model(h2o_data)
            
            # Test H2 hypothesis
            h2_results = method2.test_h2_contrast()
            results.update(h2_results)
            
            # Get marginal means
            emmeans = method2.compute_emmeans()
            results['marginal_means'] = emmeans.get('emmeans_dict', {})
            
            # Log results  
            self.logger.info(f"H2 Interaction coefficient: {h2_results.get('interaction_coefficient', 0):.6f}")
            self.logger.info(f"H2 P-value: {h2_results.get('p_value', 1.0):.6f}")
            self.logger.info(f"H2 Significant: {h2_results.get('h2_supported', False)}")
            
            # Update key names for consistency
            results['interaction_coef'] = h2_results.get('interaction_coefficient', 0)
            results['interaction_pvalue'] = h2_results.get('p_value', 1.0)
            results['h2_supported'] = h2_results.get('h2_supported', False)
            
            # Log marginal means
            if 'marginal_means' in results:
                self.logger.info("\nMarginal means (fungi richness):")
                for group, mean in results['marginal_means'].items():
                    self.logger.info(f"  {group}: {mean:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Method 2 failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def run_method3_dml(self, data_path: str, spatial_grid_step: int = 1) -> Dict[str, Any]:
        """Run Method 3: Double Machine Learning for causal inference.
        
        Tests all hypotheses using DML with cross-fitting and bootstrap inference.
        """
        self.logger.info("="*60)
        self.logger.info("Method 3: Double Machine Learning (All Hypotheses)")
        self.logger.info("="*60)
        
        try:
            method3 = Method3DML(self.config)
            
            # Load and prepare data with spatial sampling
            self.logger.info(f"Loading data from {data_path}")
            if spatial_grid_step > 1:
                self.logger.info(f"Using spatial grid sampling: every {spatial_grid_step}th point")
            
            # Load data with spatial sampling
            h2o_data = method3.load_data(data_path, spatial_grid_step=spatial_grid_step)
            h2o_data = method3.prepare_data_h2o(h2o_data)
            
            # Test all hypotheses
            results = {}
            for hypothesis in ['h0', 'h1', 'h2']:
                self.logger.info(f"\nTesting hypothesis {hypothesis.upper()}...")
                results[hypothesis] = method3.test_hypothesis(hypothesis)
            
            # Add comparison
            results['comparison'] = method3._compare_hypotheses(results)
            
            # Log summary
            self.logger.info("\n" + "="*60)
            self.logger.info("DML SUMMARY")
            self.logger.info("="*60)
            
            for hypothesis in ['h0', 'h1', 'h2']:
                hyp_results = results[hypothesis]
                self.logger.info(f"\n{hypothesis.upper()}:")
                self.logger.info(f"  Causal effect: {hyp_results['causal_effect']:.6f}")
                self.logger.info(f"  Bootstrap CI: [{hyp_results['bootstrap']['ci_lower']:.6f}, "
                               f"{hyp_results['bootstrap']['ci_upper']:.6f}]")
                self.logger.info(f"  Significant: {hyp_results['significant']}")
                self.logger.info(f"  {hyp_results['interpretation']}")
            
            # Log comparison
            comparison = results['comparison']
            self.logger.info(f"\nStrongest effect: {comparison['strongest_effect'].upper()}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Method 3 failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def run_diagnostics(self, data_path: str) -> Dict[str, Any]:
        """Run diagnostic checks on the data."""
        self.logger.info("="*60)
        self.logger.info("Running Diagnostic Checks")
        self.logger.info("="*60)
        
        try:
            import pandas as pd
            import numpy as np
            
            # Load data
            df = pd.read_parquet(data_path)
            self.logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            diagnostics = {
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'missing_summary': {},
                'dispersion_check': {},
                'correlation_matrix': {}
            }
            
            # Check missing data
            required_cols = [
                'fungi_richness', 'plants_richness', 'vertebrates_richness',
                'bio1', 'bio4', 'bio12', 'bio15', 'elevation'
            ]
            
            for col in required_cols:
                if col in df.columns:
                    missing_pct = df[col].isna().mean() * 100
                    diagnostics['missing_summary'][col] = missing_pct
                    self.logger.info(f"  {col}: {missing_pct:.2f}% missing")
            
            # Check overdispersion for count variables
            for col in ['fungi_richness', 'plants_richness', 'vertebrates_richness']:
                if col in df.columns:
                    mean_val = df[col].mean()
                    var_val = df[col].var()
                    ratio = var_val / mean_val if mean_val > 0 else 0
                    diagnostics['dispersion_check'][col] = {
                        'mean': mean_val,
                        'variance': var_val,
                        'ratio': ratio,
                        'overdispersed': ratio > 1.2
                    }
                    self.logger.info(f"  {col}: mean={mean_val:.2f}, var={var_val:.2f}, "
                                   f"ratio={ratio:.2f} {'(overdispersed)' if ratio > 1.2 else ''}")
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Diagnostics failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def run_all_methods(self, data_path: str, spatial_grid_step: int = 1) -> Dict[str, Any]:
        """Run all three methods and compare results."""
        self.logger.info("="*60)
        self.logger.info("RUNNING ALL METHODS")
        self.logger.info("="*60)
        
        all_results = {}
        
        # Method 1
        self.logger.info("\n" + "="*40)
        self.logger.info("Starting Method 1...")
        all_results['method1'] = self.run_method1_continuous(
            data_path, spatial_grid_step)
        
        # Method 2
        self.logger.info("\n" + "="*40)
        self.logger.info("Starting Method 2...")
        all_results['method2'] = self.run_method2_categorical(
            data_path, spatial_grid_step)
        
        # Method 3
        self.logger.info("\n" + "="*40)
        self.logger.info("Starting Method 3...")
        all_results['method3'] = self.run_method3_dml(
            data_path, spatial_grid_step)
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("ANALYSIS COMPLETE - SUMMARY")
        self.logger.info("="*60)
        
        # Compare H1 across methods
        if 'error' not in all_results['method1']:
            self.logger.info(f"\nH1 (Continuous interaction):")
            self.logger.info(f"  Method 1 GLM: p={all_results['method1']['interaction_pvalue']:.6f}, "
                           f"significant={all_results['method1']['h1_supported']}")
            if 'error' not in all_results['method3']:
                self.logger.info(f"  Method 3 DML: CI=[{all_results['method3']['h1']['bootstrap']['ci_lower']:.4f}, "
                               f"{all_results['method3']['h1']['bootstrap']['ci_upper']:.4f}], "
                               f"significant={all_results['method3']['h1']['significant']}")
        
        # Compare H2 across methods
        if 'error' not in all_results['method2']:
            self.logger.info(f"\nH2 (Categorical interaction):")
            self.logger.info(f"  Method 2 GLM: p={all_results['method2']['interaction_pvalue']:.6f}, "
                           f"significant={all_results['method2']['h2_supported']}")
            if 'error' not in all_results['method3']:
                self.logger.info(f"  Method 3 DML: CI=[{all_results['method3']['h2']['bootstrap']['ci_lower']:.4f}, "
                               f"{all_results['method3']['h2']['bootstrap']['ci_upper']:.4f}], "
                               f"significant={all_results['method3']['h2']['significant']}")
        
        return all_results


def main():
    """Main entry point for regression analysis."""
    parser = argparse.ArgumentParser(
        description='ML Regression Analysis for Biodiversity Hypothesis Testing'
    )
    parser.add_argument('input_data', help='Path to input parquet file')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output-dir', default='outputs/regression_results',
                       help='Output directory for results')
    parser.add_argument('--method', default='all',
                       choices=['all', 'glm-continuous', 'glm-categorical', 'dml'],
                       help='Which method to run')
    
    # Sampling option (just like ParquetLoader in biodiversity_analysis)
    parser.add_argument('--spatial-grid-step', type=int, default=1,
                       help='Spatial grid sampling: take every Nth point (1=no sampling, 10=every 10th)')
    
    parser.add_argument('--diagnostics-only', action='store_true',
                       help='Run only diagnostic checks')
    parser.add_argument('--use-legacy-config', action='store_true',
                       help='Use legacy config system instead of unified')
    
    args = parser.parse_args()
    
    # Determine config path
    config_path = None
    if args.config:
        config_path = Path(args.config)
    
    # Create runner
    runner = MLRegressionRunner(
        config_path=config_path,
        use_unified=not args.use_legacy_config
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run diagnostics if requested
        if args.diagnostics_only:
            results = runner.run_diagnostics(args.input_data)
        elif args.method == 'all':
            results = runner.run_all_methods(
                args.input_data, args.spatial_grid_step)
        elif args.method == 'glm-continuous':
            results = runner.run_method1_continuous(
                args.input_data, args.spatial_grid_step)
        elif args.method == 'glm-categorical':
            results = runner.run_method2_categorical(
                args.input_data, args.spatial_grid_step)
        elif args.method == 'dml':
            results = runner.run_method3_dml(
                args.input_data, args.spatial_grid_step)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # Save results
        import json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'regression_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        runner.logger.info(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        runner.logger.error(f"Analysis failed: {e}")
        runner.logger.error(traceback.format_exc())
        sys.exit(1)
    
    runner.logger.info("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
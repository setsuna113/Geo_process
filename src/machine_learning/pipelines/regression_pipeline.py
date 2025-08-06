"""Main regression analysis pipeline.

Integrates all components for biodiversity hypothesis testing.
Based on regression_analysis_decisions.md.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import time

from ..config_loader import MLRegressionConfig
from ..hypothesis import HypothesisTester
from ..models.glm import DHARMaDiagnostics

logger = logging.getLogger(__name__)


class RegressionPipeline:
    """Main pipeline for biodiversity regression analysis.
    
    Orchestrates the complete analysis workflow from data loading
    through hypothesis testing and result reporting.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize regression pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            self.config = MLRegressionConfig(config_path)
        else:
            self.config = MLRegressionConfig()
        
        # Initialize hypothesis tester
        self.hypothesis_tester = HypothesisTester(self.config)
        
        # Storage for results
        self.data = None
        self.results = {}
        self.execution_time = {}
        
        logger.info("Initialized RegressionPipeline")
        logger.info(f"Configuration loaded from: {config_path or 'default'}")
    
    def load_data(self, data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load biodiversity data.
        
        Args:
            data_source: Path to data file or DataFrame
            
        Returns:
            Loaded DataFrame
        """
        logger.info("Loading biodiversity data...")
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
            logger.info(f"Loaded DataFrame with {len(self.data)} samples")
        elif isinstance(data_source, str):
            # Determine file type and load
            data_path = Path(data_source)
            
            if data_path.suffix == '.parquet':
                self.data = pd.read_parquet(data_path)
            elif data_path.suffix == '.csv':
                self.data = pd.read_csv(data_path)
            elif data_path.suffix == '.pkl':
                self.data = pd.read_pickle(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
            logger.info(f"Loaded data from {data_path}")
            logger.info(f"Shape: {self.data.shape}")
        else:
            raise ValueError("data_source must be a DataFrame or file path")
        
        # Validate required columns
        required_cols = [
            'fungi_richness', 'plants_richness', 'vertebrates_richness',
            'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
            'longitude', 'latitude'
        ]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("Data validation passed")
        
        # Log data summary
        self._log_data_summary()
        
        return self.data
    
    def _log_data_summary(self):
        """Log summary statistics of the data."""
        logger.info("\nData Summary:")
        logger.info(f"  Total samples: {len(self.data)}")
        
        # Species richness summary
        for species in ['fungi', 'plants', 'vertebrates']:
            col = f'{species}_richness'
            logger.info(f"  {species.capitalize()} richness:")
            logger.info(f"    Mean: {self.data[col].mean():.1f}")
            logger.info(f"    Median: {self.data[col].median():.1f}")
            logger.info(f"    Range: [{self.data[col].min()}, {self.data[col].max()}]")
        
        # Climate variable ranges
        logger.info("  Climate variables:")
        for var in ['bio1', 'bio4', 'bio12', 'bio15']:
            logger.info(f"    {var}: [{self.data[var].min():.1f}, {self.data[var].max():.1f}]")
        
        # Geographic extent
        logger.info("  Geographic extent:")
        logger.info(f"    Longitude: [{self.data['longitude'].min():.2f}, {self.data['longitude'].max():.2f}]")
        logger.info(f"    Latitude: [{self.data['latitude'].min():.2f}, {self.data['latitude'].max():.2f}]")
        
        # Missing data
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100)
        if missing_pct.max() > 0:
            logger.info("  Missing data:")
            for col, pct in missing_pct[missing_pct > 0].items():
                logger.info(f"    {col}: {pct:.1f}%")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete regression analysis.
        
        Returns:
            Dictionary with all analysis results
        """
        if self.data is None:
            raise ValueError("Must load data first")
        
        logger.info("\n" + "="*70)
        logger.info("STARTING BIODIVERSITY REGRESSION ANALYSIS PIPELINE")
        logger.info("="*70)
        
        start_time = time.time()
        
        try:
            # Run all hypothesis tests
            self.results = self.hypothesis_tester.run_all_tests(self.data)
            
            # Add metadata
            self.results['metadata'] = {
                'n_samples': len(self.data),
                'n_features': len(self.data.columns),
                'config': self.config.config,
                'execution_time': time.time() - start_time
            }
            
            # Log completion
            logger.info("\n" + "="*70)
            logger.info("ANALYSIS COMPLETE")
            logger.info("="*70)
            logger.info(f"Total execution time: {self.results['metadata']['execution_time']:.1f} seconds")
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        
        return self.results
    
    def _print_summary(self):
        """Print analysis summary."""
        if 'summary' not in self.results:
            return
        
        summary = self.results['summary']
        
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*60)
        
        # H0: Independence
        if 'h0' in summary:
            logger.info("\nH0 (Independence):")
            logger.info(f"  Supported: {summary['h0']['supported']}")
            logger.info(f"  {summary['h0']['interpretation']}")
        
        # H1: Biodiversity begets biodiversity
        if 'h1' in summary:
            logger.info("\nH1 (Biodiversity begets biodiversity):")
            logger.info(f"  Supported: {summary['h1']['supported']}")
            logger.info(f"  {summary['h1']['interpretation']}")
        
        # H2: Temperate mismatch
        if 'h2' in summary:
            logger.info("\nH2 (Temperate mismatch):")
            logger.info(f"  Consensus: {summary['h2']['consensus']}")
            logger.info(f"  {summary['h2']['interpretation']}")
        
        # Conclusions
        if 'conclusions' in summary:
            logger.info("\nKey Conclusions:")
            for i, conclusion in enumerate(summary['conclusions'], 1):
                logger.info(f"  {i}. {conclusion}")
    
    def save_results(self, output_dir: str):
        """Save all results to directory.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        results_path = output_path / 'regression_results.json'
        self.hypothesis_tester.save_results(str(results_path))
        
        # Save visualizations
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        self.hypothesis_tester.create_visualizations(str(viz_dir))
        
        # Save summary report
        self._save_report(output_path / 'analysis_report.txt')
        
        logger.info(f"Results saved to {output_path}")
    
    def _save_report(self, report_path: Path):
        """Save text report of analysis.
        
        Args:
            report_path: Path to save report
        """
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BIODIVERSITY REGRESSION ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-"*30 + "\n")
            f.write(f"Total samples: {len(self.data)}\n")
            f.write(f"Features: {', '.join(self.data.columns[:10])}...\n\n")
            
            # Method results
            for method in ['method1', 'method2', 'method3']:
                if method in self.results:
                    f.write(f"\n{method.upper()} RESULTS\n")
                    f.write("-"*30 + "\n")
                    
                    if 'summary' in self.results[method]:
                        summary = self.results[method]['summary']
                        f.write(f"Method: {summary.get('method', 'Unknown')}\n")
                        
                        # Write key results based on method
                        if method == 'method3' and 'hypotheses' in summary:
                            for hyp, hyp_res in summary['hypotheses'].items():
                                f.write(f"\n{hyp.upper()}:\n")
                                f.write(f"  Effect: {hyp_res['causal_effect']:.4f}\n")
                                f.write(f"  P-value: {hyp_res['p_value']:.4f}\n")
                                f.write(f"  {hyp_res['interpretation']}\n")
            
            # Overall synthesis
            if 'summary' in self.results:
                f.write("\n\nOVERALL SYNTHESIS\n")
                f.write("="*70 + "\n")
                
                summary = self.results['summary']
                
                if 'conclusions' in summary:
                    f.write("\nKey Conclusions:\n")
                    for i, conclusion in enumerate(summary['conclusions'], 1):
                        f.write(f"{i}. {conclusion}\n")
                
                f.write("\n" + "="*70 + "\n")
                f.write(f"Analysis completed in {self.results['metadata']['execution_time']:.1f} seconds\n")
    
    def run_diagnostics_only(self) -> Dict[str, Any]:
        """Run only diagnostic checks without full analysis.
        
        Returns:
            Diagnostic results
        """
        if self.data is None:
            raise ValueError("Must load data first")
        
        logger.info("Running diagnostic checks only...")
        
        # Prepare data
        data_prepared = self.hypothesis_tester.prepare_data(self.data)
        
        # Check overdispersion
        dispersion_ratio = data_prepared['fungi_richness'].var() / data_prepared['fungi_richness'].mean()
        
        diagnostics = {
            'dispersion_ratio': dispersion_ratio,
            'recommended_distribution': 'negative_binomial' if dispersion_ratio > 1.5 else 'poisson',
            'n_samples': len(data_prepared),
            'n_complete_cases': len(data_prepared),
            'n_dropped': len(self.data) - len(data_prepared)
        }
        
        logger.info(f"Dispersion ratio: {dispersion_ratio:.2f}")
        logger.info(f"Recommended distribution: {diagnostics['recommended_distribution']}")
        
        return diagnostics


def run_regression_pipeline(
    data_path: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to run complete pipeline.
    
    Args:
        data_path: Path to data file
        config_path: Optional path to config file
        output_dir: Optional directory to save results
        
    Returns:
        Analysis results
    """
    # Initialize pipeline
    pipeline = RegressionPipeline(config_path)
    
    # Load data
    pipeline.load_data(data_path)
    
    # Run analysis
    results = pipeline.run_analysis()
    
    # Save results if requested
    if output_dir:
        pipeline.save_results(output_dir)
    
    return results
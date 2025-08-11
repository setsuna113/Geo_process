"""Main regression analysis pipeline.

Integrates all components for biodiversity hypothesis testing.
Orchestrates: Data Loading → HypothesisTester → Methods (1,2,3) → Models (GLM/GBM)
Based on regression_analysis_decisions.md.

Architecture:
    Script Layer → Pipeline Layer → Method Layer → Model Layer
    ↓                ↓                ↓              ↓
    Entry point      Orchestration    Business logic Training logic
    (run_ml_pipeline) (this file)     (method1/2/3)  (h2o_gbm/glm)
"""

import logging
import pandas as pd
import numpy as np
import h2o
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import json
import time

from ..config_loader import MLRegressionConfig
from ..hypothesis import HypothesisTester
from ..diagnostics import DHARMaDiagnostics
from ..utils.security_utils import PathValidator, DataLoader
from ..utils.h2o_utils import ensure_h2o_initialized
from ..preprocessing.mesf_preprocessor import MESFPreprocessor

# Import preprocessing functions
from ..preprocessing import (
    ColumnStandardizer,
    sample_spatial_grid,
    remove_missing,
    standardize_climate_2sd,
    add_elevation_squared,
    log_transform_biodiversity
)

logger = logging.getLogger(__name__)


class RegressionPipeline:
    """Main pipeline for biodiversity regression analysis.
    
    Orchestrates the complete analysis workflow from data loading
    through hypothesis testing and result reporting.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
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
    
    def load_and_preprocess_data(
        self,
        data_source: Union[str, Path],
        spatial_grid_step: Optional[int] = None,
        use_h2o: bool = True
    ) -> Union[pd.DataFrame, h2o.H2OFrame]:
        """Load and preprocess biodiversity data with all common transformations.
        
        This method:
        1. Loads data and standardizes columns (including fungi_richness creation)
        2. Applies spatial sampling if specified
        3. Adds MESF spatial eigenvectors
        4. Removes missing values
        5. Standardizes climate variables (2SD method)
        6. Adds polynomial features (elevation squared)
        7. Adds log-transformed biodiversity variables
        
        Args:
            data_source: Path to parquet file
            spatial_grid_step: Take every Nth point in spatial grid (None = use config)
            use_h2o: If True, return H2O Frame; if False, return pandas DataFrame
            
        Returns:
            Preprocessed data ready for methods
        """
        logger.info("="*60)
        logger.info("LOADING AND PREPROCESSING DATA")
        logger.info("="*60)
        
        # Get spatial config
        if spatial_grid_step is None:
            spatial_grid_step = self.config.get_data_config().get('spatial_grid_step', 1)
        
        # Step 1: Column standardization (includes fungi_richness addition)
        logger.info("Step 1: Standardizing columns and adding fungi_richness...")
        standardizer = ColumnStandardizer()
        standardized_path = standardizer.standardize_if_needed(data_source)
        
        # Step 2: Spatial sampling and MESF computation
        logger.info(f"Step 2: Spatial sampling (step={spatial_grid_step}) and MESF...")
        mesf_preprocessor = MESFPreprocessor()
        augmented_path = mesf_preprocessor.process_with_sampling(
            standardized_path,
            spatial_grid_step,
            self.config.get_spatial_config()
        )
        
        # Step 3: Load to H2O or pandas
        if use_h2o:
            ensure_h2o_initialized()
            data = h2o.import_file(augmented_path)
            logger.info(f"Loaded to H2O: {data.nrows:,} rows, {data.ncols} columns")
        else:
            data = pd.read_parquet(augmented_path)
            logger.info(f"Loaded to pandas: {len(data):,} rows, {len(data.columns)} columns")
        
        # Step 4: Remove missing values
        logger.info("Step 3: Removing missing values...")
        required_cols = [
            'fungi_richness', 'plants_richness', 'vertebrates_richness',
            'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
            'longitude', 'latitude'
        ]
        data = remove_missing(data, columns=required_cols)
        
        # Step 5: Standardize climate variables (2SD method)
        logger.info("Step 4: Standardizing climate variables (2SD method)...")
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
        data = standardize_climate_2sd(data, climate_vars)
        
        # Step 6: Add polynomial features
        logger.info("Step 5: Adding polynomial features...")
        data = add_elevation_squared(data, 'elevation_std', 'elevation_std_sq')
        
        # Step 7: Log transform biodiversity variables
        logger.info("Step 6: Log-transforming biodiversity variables...")
        data = log_transform_biodiversity(data, ['plants_richness', 'vertebrates_richness'])
        
        # Store preprocessed data
        self.data_preprocessed = data
        
        # Log summary
        logger.info("\nPreprocessing complete:")
        if use_h2o:
            logger.info(f"  Final shape: {data.nrows:,} rows × {data.ncols} columns")
            me_cols = [col for col in data.columns if col.startswith('ME')]
            logger.info(f"  MESF eigenvectors: {len(me_cols)}")
        else:
            logger.info(f"  Final shape: {data.shape}")
            me_cols = [col for col in data.columns if col.startswith('ME')]
            logger.info(f"  MESF eigenvectors: {len(me_cols)}")
        
        return data
    
    def load_data(self, data_source: Union[str, pd.DataFrame], 
                  max_file_size_mb: float = 10000,
                  validate_security: bool = True) -> pd.DataFrame:
        """Legacy load method for backwards compatibility.
        
        DEPRECATED: Use load_and_preprocess_data() instead.
        
        Args:
            data_source: Path to data file or DataFrame
            max_file_size_mb: Maximum allowed file size in MB
            validate_security: Whether to perform security validation
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If data validation fails
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        logger.info("Loading biodiversity data...")
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
            logger.info(f"Loaded DataFrame with {len(self.data)} samples")
        elif isinstance(data_source, str):
            if validate_security:
                # Use secure data loader with validation
                required_cols = [
                    'fungi_richness', 'plants_richness', 'vertebrates_richness',
                    'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
                    'longitude', 'latitude'
                ]
                
                loader = DataLoader(
                    max_file_size_mb=max_file_size_mb,
                    required_columns=required_cols
                )
                
                try:
                    self.data = loader.load_data(
                        data_source,
                        validate_shape=True
                    )
                    logger.info(f"Securely loaded data from {data_source}")
                except Exception as e:
                    logger.error(f"Failed to load data: {e}")
                    raise
            else:
                # Legacy loading without validation (backwards compatibility)
                logger.warning("Loading data without security validation")
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
        
        # Additional validation for required columns (if not done by DataLoader)
        if not validate_security:
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
    
    def _log_data_summary(self) -> None:
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
    
    def _print_summary(self) -> None:
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
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
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
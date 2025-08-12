#!/usr/bin/env python3
"""
Unified ML Orchestrator for Biodiversity Hypothesis Testing

Merges the functionality of RegressionPipeline and HypothesisTester into a single,
clean orchestrator that coordinates the entire analysis workflow.

Architecture:
    Entry Script (run_ml_pipeline.py)
        ↓
    ML Orchestrator (this file)
        ├── Data Loading & Preprocessing
        ├── Method 1: H0-H1 Testing (GLM)
        ├── Method 2: H2 Testing (Continuous & Categorical)
        ├── Method 3: H0-H1 Causal (DML)
        └── Results Synthesis & Reporting

Based on regression_analysis_decisions.md
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
import h2o

# Configuration and utilities
from src.machine_learning.config_loader import MLRegressionConfig
from src.machine_learning.utils.security_utils import PathValidator, DataLoader
from src.machine_learning.utils import H2OManager

# Methods for hypothesis testing
from src.machine_learning.methods import (
    Method1H0H1GLM,
    Method2H2Testing,
    Method3DML
)

# Preprocessing
from src.machine_learning.preprocessing import (
    ColumnStandardizer,
    sample_spatial_grid,
    remove_missing,
    standardize_climate_2sd,
    add_elevation_squared,
    log_transform_biodiversity
)
from src.machine_learning.preprocessing.mesf_preprocessor import MESFPreprocessor

# Evaluation and reporting
from src.machine_learning.visualization import create_regression_synthesis_plot
from src.machine_learning.reporting import RegressionReportGenerator

logger = logging.getLogger(__name__)


class MLOrchestrator:
    """Unified orchestrator for biodiversity ML hypothesis testing.
    
    Coordinates the complete workflow:
    1. Data loading and preprocessing
    2. Running all three methods
    3. Synthesizing results
    4. Generating reports
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize ML orchestrator.
        
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
        self.data_loader = DataLoader(self.path_validator)
        
        # Initialize H2O if needed
        if self.config.config.get('use_h2o', True):
            # Initialize H2O using manager
            self.h2o = H2OManager.get_or_init(
                max_mem_size=self.config.config.get('h2o_memory', '32G'),
                nthreads=self.config.config.get('h2o_threads', -1)
            )
        
        # Initialize methods
        self.method1 = Method1H0H1GLM(self.config)  # H0-H1 testing
        self.method2 = Method2H2Testing(self.config)  # H2 testing
        self.method3 = Method3DML(self.config)  # H0-H1 causal
        
        # Storage for data and results
        self.raw_data = None
        self.preprocessed_data = None
        self.results = {
            'method1': {},
            'method2': {},
            'method3': {},
            'synthesis': {}
        }
        
        logger.info("ML Orchestrator initialized successfully")
    
    def load_and_preprocess_data(
        self, 
        data_path: Union[str, Path],
        spatial_grid_step: Optional[int] = None,
        use_h2o: bool = True
    ) -> Union[pd.DataFrame, h2o.H2OFrame]:
        """Load and preprocess data for all methods.
        
        Args:
            data_path: Path to input data
            spatial_grid_step: Spatial sampling step (None = no sampling)
            use_h2o: Whether to convert to H2O frame
            
        Returns:
            Preprocessed data ready for methods
        """
        logger.info("="*70)
        logger.info("DATA LOADING AND PREPROCESSING")
        logger.info("="*70)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        data = self.data_loader.load_parquet(data_path)
        initial_rows = len(data)
        logger.info(f"Loaded {initial_rows:,} rows")
        
        # Remove missing data
        logger.info("Removing missing data...")
        data = remove_missing(data, target_col='fungi_richness')
        logger.info(f"After removing missing: {len(data):,} rows")
        
        # Spatial sampling if requested
        if spatial_grid_step and spatial_grid_step > 1:
            logger.info(f"Applying spatial grid sampling (step={spatial_grid_step})...")
            data = sample_spatial_grid(data, grid_step=spatial_grid_step)
            logger.info(f"After spatial sampling: {len(data):,} rows")
        
        # Standardize climate variables
        logger.info("Standardizing climate variables (2SD method)...")
        data = standardize_climate_2sd(data)
        
        # Add elevation squared
        logger.info("Adding elevation squared term...")
        data = add_elevation_squared(data)
        
        # Log transform biodiversity predictors
        logger.info("Log-transforming biodiversity predictors...")
        data = log_transform_biodiversity(data)
        
        # Add MESF spatial eigenvectors
        logger.info("Computing MESF spatial eigenvectors...")
        mesf_processor = MESFPreprocessor(self.config)
        data = mesf_processor.add_mesf_features(data)
        
        # Convert to H2O if requested
        if use_h2o:
            logger.info("Converting to H2O frame...")
            # Ensure H2O is initialized using manager
            self.h2o = H2OManager.get_or_init()
            data = h2o.H2OFrame(data)
            logger.info(f"Final H2O frame: {data.nrows:,} rows × {data.ncols} columns")
        else:
            logger.info(f"Final DataFrame: {len(data):,} rows × {len(data.columns)} columns")
        
        self.preprocessed_data = data
        return data
    
    def run_method1_h0_h1(self) -> Dict[str, Any]:
        """Run Method 1: H0-H1 Testing with GLM.
        
        Tests:
        - H0: Fungi independent of biodiversity
        - H1: Biodiversity begets biodiversity
        
        Returns:
            Method 1 results
        """
        logger.info("\n" + "="*70)
        logger.info("METHOD 1: H0-H1 TESTING (GLM)")
        logger.info("="*70)
        
        if self.preprocessed_data is None:
            raise ValueError("Must load and preprocess data first")
        
        # Fit models (Model 1 vs Model 2)
        model_results = self.method1.fit_models(self.preprocessed_data)
        
        # Get biodiversity coefficients
        h1_results = self.method1.get_biodiversity_coefficients()
        
        # Store results
        self.results['method1'] = {
            'models': model_results,
            'h0_h1_test': h1_results,
            'lr_test': model_results.get('lr_test_h0_h1'),
            'aic_comparison': {
                'model1_aic': model_results['null_model']['aic'],
                'model2_aic': model_results['full_model']['aic'],
                'delta_aic': model_results['null_model']['aic'] - model_results['full_model']['aic']
            }
        }
        
        # Log key findings
        logger.info("\nMethod 1 Results (H0-H1):")
        logger.info(f"  H1 supported: {h1_results['supports_h1']}")
        logger.info(f"  Plants coef: {h1_results['plants']['coefficient']:.4f} "
                   f"(p={h1_results['plants'].get('p_value', 'N/A'):.4f})")
        logger.info(f"  Vertebrates coef: {h1_results['vertebrates']['coefficient']:.4f} "
                   f"(p={h1_results['vertebrates'].get('p_value', 'N/A'):.4f})")
        logger.info(f"  LRT p-value: {model_results['lr_test_h0_h1']['p_value']:.4f}")
        logger.info(f"  ΔAIC: {self.results['method1']['aic_comparison']['delta_aic']:.2f}")
        
        return self.results['method1']
    
    def run_method2_h2(self) -> Dict[str, Any]:
        """Run Method 2: H2 Testing with two approaches.
        
        Tests H2 (Temperate Mismatch) using:
        - Model 3a: Continuous interaction
        - Model 3b: Categorical with emmeans
        
        Returns:
            Method 2 results
        """
        logger.info("\n" + "="*70)
        logger.info("METHOD 2: H2 TESTING (TEMPERATE MISMATCH)")
        logger.info("="*70)
        
        if self.preprocessed_data is None:
            raise ValueError("Must load and preprocess data first")
        
        # Fit both models (3a and 3b)
        model_results = self.method2.fit_models(self.preprocessed_data)
        
        # Compute emmeans for categorical model
        emmeans_results = self.method2.compute_emmeans()
        
        # Test H2 contrast
        h2_results = self.method2.test_h2_contrast()
        
        # Store results
        self.results['method2'] = {
            'models': model_results,
            'emmeans': emmeans_results,
            'h2_test': h2_results
        }
        
        # Log key findings
        logger.info("\nMethod 2 Results (H2):")
        logger.info("\nModel 3a (Continuous):")
        continuous_coef = model_results['model_3a_continuous']['interaction_coef']
        logger.info(f"  Interaction coef: {continuous_coef['coefficient']:.4f} "
                   f"(p={continuous_coef.get('p_value', 'N/A'):.4f})")
        
        logger.info("\nModel 3b (Categorical with emmeans):")
        logger.info(f"  H2 supported: {h2_results['h2_supported']}")
        logger.info(f"  Contrast estimate: {h2_results['contrast_estimate']:.4f} "
                   f"(p={h2_results['p_value']:.4f})")
        
        return self.results['method2']
    
    def run_method3_dml(self) -> Dict[str, Any]:
        """Run Method 3: H0-H1 Causal Testing with DML.
        
        Provides causal inference for biodiversity effects.
        
        Returns:
            Method 3 results
        """
        logger.info("\n" + "="*70)
        logger.info("METHOD 3: H0-H1 CAUSAL TESTING (DML)")
        logger.info("="*70)
        
        if self.preprocessed_data is None:
            raise ValueError("Must load and preprocess data first")
        
        # Test all hypotheses using DML
        hypothesis_results = self.method3.test_all_hypotheses(self.preprocessed_data)
        
        # Store results
        self.results['method3'] = {
            'hypotheses': hypothesis_results
        }
        
        # Log key findings
        logger.info("\nMethod 3 Results (Causal):")
        for hyp, res in hypothesis_results.items():
            if hyp in ['h0', 'h1']:  # Only H0-H1 for Method 3
                logger.info(f"  {hyp.upper()}: effect={res.get('causal_effect', 'N/A'):.4f} "
                           f"(p={res.get('p_value', 'N/A'):.4f})")
                logger.info(f"    {res.get('interpretation', '')}")
        
        return self.results['method3']
    
    def synthesize_results(self) -> Dict[str, Any]:
        """Synthesize results across all methods.
        
        Returns:
            Synthesis of findings
        """
        logger.info("\n" + "="*70)
        logger.info("SYNTHESIS OF RESULTS")
        logger.info("="*70)
        
        synthesis = {}
        
        # H0-H1 Synthesis (Method 1 and Method 3)
        if 'h0_h1_test' in self.results['method1']:
            h1_glm = self.results['method1']['h0_h1_test']['supports_h1']
            h1_glm_plants_p = self.results['method1']['h0_h1_test']['plants'].get('p_value')
            h1_glm_vertebrates_p = self.results['method1']['h0_h1_test']['vertebrates'].get('p_value')
            
            synthesis['h0_h1'] = {
                'glm_method1': {
                    'h1_supported': h1_glm,
                    'plants_significant': h1_glm_plants_p < 0.05 if h1_glm_plants_p else False,
                    'vertebrates_significant': h1_glm_vertebrates_p < 0.05 if h1_glm_vertebrates_p else False
                }
            }
            
            # Add DML results if available
            if 'hypotheses' in self.results['method3']:
                if 'h1' in self.results['method3']['hypotheses']:
                    h1_dml = self.results['method3']['hypotheses']['h1']
                    synthesis['h0_h1']['dml_method3'] = {
                        'h1_supported': h1_dml.get('significant', False),
                        'causal_effect': h1_dml.get('causal_effect')
                    }
            
            # Overall H0-H1 conclusion
            glm_support = synthesis['h0_h1']['glm_method1']['h1_supported']
            dml_support = synthesis['h0_h1'].get('dml_method3', {}).get('h1_supported', False)
            
            if glm_support and dml_support:
                synthesis['h0_h1']['conclusion'] = "STRONG EVIDENCE: Both GLM and causal analyses support biodiversity begets biodiversity"
            elif glm_support:
                synthesis['h0_h1']['conclusion'] = "MODERATE EVIDENCE: GLM supports biodiversity effects (causal not tested or not significant)"
            else:
                synthesis['h0_h1']['conclusion'] = "NO EVIDENCE: Biodiversity effects not supported"
        
        # H2 Synthesis (Method 2 only)
        if 'models' in self.results['method2']:
            continuous_sig = (
                self.results['method2']['models']['model_3a_continuous']['interaction_coef'].get('p_value', 1) < 0.05
            )
            categorical_supported = self.results['method2']['h2_test']['h2_supported']
            
            synthesis['h2'] = {
                'continuous_significant': continuous_sig,
                'categorical_supported': categorical_supported,
                'consensus': continuous_sig or categorical_supported
            }
            
            if continuous_sig and categorical_supported:
                synthesis['h2']['conclusion'] = "STRONG EVIDENCE: Both approaches support temperate mismatch"
            elif continuous_sig:
                synthesis['h2']['conclusion'] = "MODERATE EVIDENCE: Continuous interaction supports temperate mismatch"
            elif categorical_supported:
                synthesis['h2']['conclusion'] = "MODERATE EVIDENCE: Categorical analysis supports temperate mismatch"
            else:
                synthesis['h2']['conclusion'] = "NO EVIDENCE: Temperate mismatch not supported"
        
        self.results['synthesis'] = synthesis
        
        # Log synthesis
        logger.info("\nHYPOTHESIS TESTING SYNTHESIS:")
        
        if 'h0_h1' in synthesis:
            logger.info(f"\nH0-H1 (Biodiversity Effect):")
            logger.info(f"  {synthesis['h0_h1']['conclusion']}")
        
        if 'h2' in synthesis:
            logger.info(f"\nH2 (Temperate Mismatch):")
            logger.info(f"  {synthesis['h2']['conclusion']}")
        
        return synthesis
    
    def run_complete_analysis(
        self,
        data_path: Union[str, Path],
        spatial_grid_step: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run complete analysis workflow.
        
        Args:
            data_path: Path to input data
            spatial_grid_step: Spatial sampling step
            output_dir: Directory for output files
            
        Returns:
            Complete results dictionary
        """
        start_time = time.time()
        
        logger.info("\n" + "="*80)
        logger.info("BIODIVERSITY HYPOTHESIS TESTING - COMPLETE ANALYSIS")
        logger.info("="*80)
        
        try:
            # 1. Load and preprocess data
            self.load_and_preprocess_data(data_path, spatial_grid_step)
            
            # 2. Run all methods
            self.run_method1_h0_h1()
            self.run_method2_h2()
            self.run_method3_dml()
            
            # 3. Synthesize results
            self.synthesize_results()
            
            # 4. Add metadata
            self.results['metadata'] = {
                'data_path': str(data_path),
                'n_samples': self.preprocessed_data.nrows if hasattr(self.preprocessed_data, 'nrows') else len(self.preprocessed_data),
                'spatial_grid_step': spatial_grid_step,
                'execution_time': time.time() - start_time,
                'config': self.config.config
            }
            
            # 5. Save results if output directory provided
            if output_dir:
                self.save_results(output_dir)
                self.generate_reports(output_dir)
            
            logger.info("\n" + "="*80)
            logger.info("ANALYSIS COMPLETE")
            logger.info(f"Total execution time: {self.results['metadata']['execution_time']:.1f} seconds")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        
        return self.results
    
    def save_results(self, output_dir: Path) -> None:
        """Save results to JSON file.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy/H2O values for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        results_serializable = convert_for_json(self.results)
        
        output_path = output_dir / "ml_results.json"
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_reports(self, output_dir: Path) -> None:
        """Generate visualization and reports.
        
        Args:
            output_dir: Directory to save reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate synthesis plot
            viz_path = output_dir / "synthesis_plot.png"
            create_regression_synthesis_plot(self.results, save_path=str(viz_path))
            logger.info(f"Visualization saved to {viz_path}")
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
        
        try:
            # Generate report
            report_gen = RegressionReportGenerator(self.results, {})
            report_gen.generate_full_report(str(output_dir))
            logger.info(f"Report generated in {output_dir}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
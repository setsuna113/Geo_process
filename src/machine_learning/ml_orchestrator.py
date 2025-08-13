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
# pandas only imported locally for MESF operations
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

# Preprocessing - H2O optimized versions
from src.machine_learning.preprocessing.h2o_preprocessing import (
    remove_missing,
    standardize_climate_2sd,
    add_elevation_squared,
    log_transform_biodiversity,
    standardize_column_names,
    add_fungi_richness
)
from src.machine_learning.preprocessing.mesf_processor import MESFProcessor

# Diagnostics
from src.machine_learning.diagnostics import DHARMaDiagnostics

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
        self.data_loader = DataLoader(max_file_size_mb=10000)
        self.mesf_processor = MESFProcessor()
        
        # Initialize H2O if needed
        config_dict = self.config.config if hasattr(self.config, 'config') else {}
        self.h2o_initialized = False
        
        # Get H2O configuration from config file
        h2o_config = config_dict.get('h2o', {})
        h2o_enabled = h2o_config.get('enabled', True)
        
        # Initialize H2O
        if h2o_enabled:
            try:
                # Initialize H2O using manager with configuration from file
                self.h2o = H2OManager.get_or_init(
                    max_mem_size=h2o_config.get('max_mem_size', '32G'),
                    nthreads=h2o_config.get('nthreads', -1),
                    ice_root=h2o_config.get('ice_root', '/tmp/h2o_tmp')
                )
                self.h2o_initialized = True
                logger.info(f"H2O cluster initialized successfully with {h2o_config.get('max_mem_size', '32G')} memory")
            except Exception as e:
                logger.warning(f"Failed to initialize H2O cluster: {e}. Will fall back to non-H2O methods if needed.")
                self.h2o = None
                self.h2o_initialized = False
        
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
        use_h2o: Optional[bool] = None
    ) -> h2o.H2OFrame:
        """Load and preprocess data directly in H2O for all methods.
        
        Args:
            data_path: Path to input data
            spatial_grid_step: Spatial sampling step (None = no sampling)
            use_h2o: Deprecated, always returns H2O frame
            
        Returns:
            Preprocessed H2O frame ready for methods
        """
        logger.info("="*70)
        logger.info("DATA LOADING AND PREPROCESSING (H2O-Optimized)")
        logger.info("="*70)
        
        # Ensure H2O is initialized
        if not self.h2o_initialized:
            raise RuntimeError("H2O not initialized. Cannot proceed.")
        
        # Load data using DataLoader (now loads directly into H2O with spatial sampling)
        logger.info(f"Loading data from {data_path}")
        data = self.data_loader.load_data(data_path, spatial_grid_step=spatial_grid_step)
        initial_rows = data.nrows
        logger.info(f"Loaded {initial_rows:,} rows into H2O")
        
        # Apply preprocessing pipeline using abstracted H2O functions
        logger.info("Applying preprocessing pipeline...")
        
        # Remove missing data
        data = remove_missing(data)
        logger.info(f"After removing missing: {data.nrows:,} rows")
        
        # Standardize column names
        data = standardize_column_names(data)
        
        # Add fungi_richness if needed
        data = add_fungi_richness(data)
        
        # Standardize climate variables (2SD method)
        data = standardize_climate_2sd(data)
        
        # Add polynomial terms
        data = add_elevation_squared(data)
        
        # Log transform biodiversity predictors
        data = log_transform_biodiversity(data)
        
        # Add MESF spatial eigenvectors
        logger.info("Adding MESF spatial eigenvectors...")
        
        # Get MESF config (simplified - no complex params needed)
        spatial_config = self.config.config.get('spatial', {})
        n_eigenvectors = spatial_config.get('n_eigenvectors', 10)
        distance_threshold = spatial_config.get('distance_threshold', 100.0)
        
        # Get n_jobs from H2O config (unified thread control)
        h2o_config = self.config.config.get('h2o', {})
        n_jobs = h2o_config.get('nthreads', -1)
        
        # Get MESF eigenvectors using clean Nystrom-only implementation
        mesf_eigenvectors = self.mesf_processor.get_mesf_eigenvectors(
            data_path=str(data_path),
            spatial_grid_step=spatial_grid_step,
            n_eigenvectors=n_eigenvectors,
            distance_threshold=distance_threshold,
            n_jobs=n_jobs
        )
        if mesf_eigenvectors is not None:
            data = data.cbind(mesf_eigenvectors)
            logger.info(f"Added {mesf_eigenvectors.ncols} MESF eigenvectors to data")
        
        # Convert to H2O if requested or if H2O is initialized
        if use_h2o is None:
            use_h2o = self.h2o_initialized
            
        if use_h2o:
            logger.info("Converting to H2O frame...")
            # Ensure H2O is initialized and connected
            import h2o
            
            # Always verify H2O connection before creating frames
            h2o_connected = False
            try:
                # Try to actually use H2O, not just check cluster
                test_frame = h2o.H2OFrame([[1,2,3]])
                test_frame.rm()
                h2o_connected = True
                logger.debug("H2O connection verified")
            except:
                logger.info("H2O not connected, initializing...")
                try:
                    # Get H2O config from configuration
                    config_dict = self.config.config if hasattr(self.config, 'config') else {}
                    h2o_config = config_dict.get('h2o', {})
                    
                    h2o.init(
                        max_mem_size=h2o_config.get('max_mem_size', '32G'),
                        nthreads=h2o_config.get('nthreads', -1),
                        ice_root=h2o_config.get('ice_root', '/tmp/h2o_tmp')
                    )
                    self.h2o_initialized = True
                    logger.info(f"H2O initialized successfully with {h2o_config.get('max_mem_size', '32G')} memory")
                except Exception as e:
                    logger.error(f"Failed to initialize H2O: {e}")
                    raise
            
            # Create H2O frame
            data = h2o.H2OFrame(data)
            logger.info(f"Final H2O frame: {data.nrows:,} rows × {data.ncols} columns")
        else:
            logger.info(f"Final H2O Frame: {data.nrows:,} rows × {data.ncols} columns")
        
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
            'model_comparison': model_results.get('model_comparison'),
            'h0_h1_conclusion': model_results.get('h0_h1_conclusion')
        }
        
        # Log key findings
        logger.info("\nMethod 1 Results (H0-H1):")
        logger.info(f"  H1 supported: {h1_results['supports_h1']}")
        
        # Log coefficient info if available
        if 'plants' in h1_results:
            logger.info(f"  Plants coef: {h1_results['plants']['coefficient']:.4f} "
                       f"(p={h1_results['plants'].get('p_value', 'N/A')})")
        if 'vertebrates' in h1_results:
            logger.info(f"  Vertebrates coef: {h1_results['vertebrates']['coefficient']:.4f} "
                       f"(p={h1_results['vertebrates'].get('p_value', 'N/A')})")
        
        # Log LRT results if available
        if model_results.get('lr_test_h0_h1'):
            logger.info(f"  LRT p-value: {model_results['lr_test_h0_h1']['p_value']:.4f}")
        
        # Log model comparison if available
        if model_results.get('model_comparison'):
            comparison = model_results['model_comparison']
            if 'best_models' in comparison:
                logger.info(f"  Best model (AIC): {comparison['best_models'].get('aic', 'N/A')}")
        
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
                'config': self.config.config,
                'h2o_initialized': self.h2o_initialized,
                'warnings': getattr(self, 'processing_warnings', [])
            }
            
            # 5. Run diagnostics for each method
            self.run_diagnostics()
            
            # 6. Generate reports from each method
            if output_dir:
                self.generate_method_reports(output_dir)
                
            # 7. Save raw results if output directory provided
            if output_dir:
                self.save_results(output_dir)
            
            logger.info("\n" + "="*80)
            logger.info("ANALYSIS COMPLETE")
            logger.info(f"Total execution time: {self.results['metadata']['execution_time']:.1f} seconds")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Clean up resources on error
            self.cleanup()
            raise
        finally:
            # Always clean up resources
            self.cleanup()
        
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
    
    def run_diagnostics(self) -> None:
        """Run diagnostics for each method's models.
        
        Diagnostics are method-specific and stored in results.
        """
        logger.info("\n" + "-"*60)
        logger.info("RUNNING MODEL DIAGNOSTICS")
        logger.info("-"*60)
        
        # Method 1 diagnostics
        if self.results.get('method1'):
            try:
                logger.info("Running diagnostics for Method 1...")
                dharma = DHARMaDiagnostics()
                method1_diag = dharma.run_diagnostics(
                    self.method1.model_1a,
                    self.preprocessed_data
                )
                self.results['method1']['diagnostics'] = method1_diag
                logger.info(f"Method 1 diagnostics: {'PASSED' if method1_diag.get('model_adequate') else 'ISSUES FOUND'}")
            except Exception as e:
                logger.warning(f"Method 1 diagnostics failed: {e}")
        
        # Method 2 diagnostics  
        if self.results.get('method2'):
            try:
                logger.info("Running diagnostics for Method 2...")
                dharma = DHARMaDiagnostics()
                method2_diag = dharma.run_diagnostics(
                    self.method2.model_3b_categorical,
                    self.preprocessed_data
                )
                self.results['method2']['diagnostics'] = method2_diag
                logger.info(f"Method 2 diagnostics: {'PASSED' if method2_diag.get('model_adequate') else 'ISSUES FOUND'}")
            except Exception as e:
                logger.warning(f"Method 2 diagnostics failed: {e}")
        
        # Method 3 uses DML - different diagnostic approach
        if self.results.get('method3'):
            logger.info("Method 3 (DML) has built-in diagnostic checks")
    
    def generate_method_reports(self, output_dir: Path) -> None:
        """Generate reports from each method using their internal reporters.
        
        Each method controls its own reporting through hypothesis/evaluation modules.
        
        Args:
            output_dir: Directory to save reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "-"*60)
        logger.info("GENERATING METHOD REPORTS")
        logger.info("-"*60)
        
        # Method 1 report generation
        if self.results.get('method1'):
            try:
                logger.info("Generating Method 1 report...")
                method1_dir = output_dir / "method1"
                method1_dir.mkdir(exist_ok=True)
                
                method1_report = self.method1.generate_report(method1_dir)
                self.results['method1']['report'] = method1_report
                logger.info(f"Method 1 report generated with {len(method1_report.get('visualizations', []))} visualizations")
            except Exception as e:
                logger.warning(f"Method 1 report generation failed: {e}")
        
        # Method 2 report generation  
        if self.results.get('method2'):
            try:
                logger.info("Generating Method 2 report...")
                method2_dir = output_dir / "method2"
                method2_dir.mkdir(exist_ok=True)
                
                method2_report = self.method2.generate_report(method2_dir)
                self.results['method2']['report'] = method2_report
                logger.info(f"Method 2 report generated with {len(method2_report.get('visualizations', []))} visualizations")
            except Exception as e:
                logger.warning(f"Method 2 report generation failed: {e}")
        
        # Method 3 report generation
        if self.results.get('method3'):
            try:
                logger.info("Generating Method 3 report...")
                method3_dir = output_dir / "method3"
                method3_dir.mkdir(exist_ok=True)
                
                method3_report = self.method3.generate_report(method3_dir)
                self.results['method3']['report'] = method3_report
                logger.info(f"Method 3 report generated with {len(method3_report.get('visualizations', []))} visualizations")
            except Exception as e:
                logger.warning(f"Method 3 report generation failed: {e}")
        
        # Generate synthesis summary
        self.generate_synthesis_summary(output_dir)
    
    def generate_synthesis_summary(self, output_dir: Path) -> None:
        """Generate a synthesis summary across all methods.
        
        Args:
            output_dir: Directory to save summary
        """
        summary = {
            'title': 'Biodiversity Hypothesis Testing - Synthesis',
            'methods_completed': [],
            'hypotheses_tested': {},
            'overall_conclusions': {}
        }
        
        # Method 1 conclusions
        if self.results.get('method1', {}).get('report'):
            summary['methods_completed'].append('Method 1: GLM H0-H1')
            summary['hypotheses_tested']['H0_H1'] = {
                'method': 'Method 1',
                'conclusion': self.results['method1']['report'].get('conclusion', {})
            }
        
        # Method 2 conclusions
        if self.results.get('method2', {}).get('report'):
            summary['methods_completed'].append('Method 2: H2 Testing')
            summary['hypotheses_tested']['H2'] = {
                'method': 'Method 2',
                'conclusion': self.results['method2']['report'].get('conclusion', {})
            }
        
        # Method 3 conclusions
        if self.results.get('method3', {}).get('report'):
            summary['methods_completed'].append('Method 3: DML Causal')
            summary['hypotheses_tested']['H0_H1_causal'] = {
                'method': 'Method 3',
                'conclusion': self.results['method3']['report'].get('conclusion', {})
            }
        
        # Overall synthesis
        h1_support = 0
        if self.results.get('method1', {}).get('supports_h1'):
            h1_support += 1
        if self.results.get('method3', {}).get('hypothesis_tests', {}).get('classical', {}).get('h1_supported'):
            h1_support += 1
        
        h2_support = self.results.get('method2', {}).get('h2_results', {}).get('overall_conclusion', {}).get('h2_supported', False)
        
        if h1_support >= 1 and h2_support:
            overall = "Evidence supports both biodiversity-begets-biodiversity (H1) and temperate mismatch (H2)"
        elif h1_support >= 1:
            overall = "Evidence supports biodiversity-begets-biodiversity (H1) but not temperate mismatch (H2)"
        elif h2_support:
            overall = "Evidence supports temperate mismatch (H2) but mixed on biodiversity effects (H1)"
        else:
            overall = "Limited evidence for either hypothesis"
        
        summary['overall_conclusions'] = {
            'synthesis': overall,
            'h1_support_count': h1_support,
            'h2_supported': h2_support
        }
        
        # Save synthesis summary
        import json
        synthesis_path = output_dir / "synthesis_summary.json"
        with open(synthesis_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Synthesis summary saved to {synthesis_path}")
    
    def cleanup(self) -> None:
        """Clean up resources (H2O cluster, cache monitoring, etc.)."""
        try:
            # Log final cache statistics if mesf_processor exists
            if hasattr(self, 'mesf_processor'):
                try:
                    cache_stats = self.mesf_processor.get_cache_stats()
                    logger.info(f"Final cache status: {cache_stats['total_files']} files, "
                               f"{cache_stats['total_size_mb']:.1f}MB used")
                except:
                    pass
            
            # Shutdown H2O cluster if initialized
            if self.h2o_initialized and self.h2o:
                logger.info("Shutting down H2O cluster...")
                import h2o
                h2o.cluster().shutdown(prompt=False)
                self.h2o_initialized = False
                logger.info("H2O cluster shutdown complete")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def _reinitialize_h2o(self) -> bool:
        """Attempt to reinitialize H2O cluster after failure.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Attempting to reinitialize H2O cluster...")
            
            # First try to shutdown any existing cluster
            try:
                import h2o
                h2o.cluster().shutdown(prompt=False)
            except:
                pass
            
            # Wait a moment for cleanup
            import time
            time.sleep(2)
            
            # Don't fallback - raise the error
            logger.error("H2O cluster connection lost and no fallback allowed")
            raise
            return True
            
        except Exception as e:
            logger.error(f"Failed to reinitialize H2O: {e}")
            self.h2o_initialized = False
            return False
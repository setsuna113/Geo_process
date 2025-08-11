#!/usr/bin/env python
"""
Machine Learning Regression Analysis Runner
===========================================
Runs hypothesis testing using optimized H2O GBM with method-aware configurations.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import ML modules
from src.machine_learning.config_loader import MLRegressionConfig
from src.machine_learning.pipelines.regression_pipeline import RegressionPipeline
from src.machine_learning.research.biodiversity_analysis import BiodiversityRegressionAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regression_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_regression_analysis(
    input_path: str,
    config_path: str = "config/ml/ml_config.yml",
    output_dir: str = "outputs/regression_results",
    method: str = "all",
    spatial_sampling: int = 100,
    diagnostics_only: bool = False
) -> Dict[str, Any]:
    """
    Run regression analysis with method-aware optimizations.
    
    Args:
        input_path: Path to input parquet file
        config_path: Path to configuration file
        output_dir: Output directory for results
        method: Which method(s) to run (all, method1, method2, method3_dml)
        spatial_sampling: Spatial grid step for sampling (100 = every 100th point)
        diagnostics_only: Only run diagnostic checks
        
    Returns:
        Dictionary of results
    """
    logger.info("=" * 70)
    logger.info("BIODIVERSITY REGRESSION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Input data: {input_path}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Method: {method}")
    logger.info(f"Spatial sampling: every {spatial_sampling} points")
    logger.info(f"Diagnostics only: {diagnostics_only}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info("\n📋 Loading configuration...")
    config = MLRegressionConfig(Path(config_path))
    
    # Override spatial sampling if specified
    if spatial_sampling != 1:
        logger.info(f"Overriding spatial sampling to {spatial_sampling}")
        if not hasattr(config, 'regression_config'):
            config.regression_config = {}
        if 'data_processing' not in config.regression_config:
            config.regression_config['data_processing'] = {}
        config.regression_config['data_processing']['spatial_grid_step'] = spatial_sampling
    
    # Initialize analysis
    logger.info("\n🔬 Initializing biodiversity regression analysis...")
    analysis = BiodiversityRegressionAnalysis(config)
    
    # Load data
    logger.info(f"\n📊 Loading data from {input_path}...")
    start_time = time.time()
    
    try:
        # Load with spatial sampling
        data = analysis.load_data(input_path)
        load_time = time.time() - start_time
        logger.info(f"✅ Data loaded in {load_time:.2f} seconds")
        logger.info(f"   Shape: {data.shape}")
        logger.info(f"   Columns: {list(data.columns[:10])}...")
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise
    
    # Run diagnostics if requested
    if diagnostics_only:
        logger.info("\n🔍 Running diagnostic checks only...")
        diagnostics = analysis.run_diagnostics(data)
        
        # Save diagnostics
        import json
        diag_path = output_path / "diagnostics.json"
        with open(diag_path, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        logger.info(f"✅ Diagnostics saved to {diag_path}")
        return diagnostics
    
    # Determine which methods to run
    methods_to_run = []
    if method == "all":
        methods_to_run = ["method1", "method2", "method3_dml"]
    elif method == "glm-continuous":
        methods_to_run = ["method1"]
    elif method == "glm-categorical":
        methods_to_run = ["method2"]
    elif method == "dml":
        methods_to_run = ["method3_dml"]
    else:
        methods_to_run = [method]
    
    logger.info(f"\n🚀 Running methods: {methods_to_run}")
    
    # Track results
    all_results = {}
    
    # Run each method
    for method_name in methods_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            method_start = time.time()
            
            # Run method with proper configuration
            if method_name == "method1":
                logger.info("Method 1: GLM Continuous - Testing H1")
                results = analysis.run_method1_continuous(data)
                
            elif method_name == "method2":
                logger.info("Method 2: GLM Categorical - Testing H2")
                results = analysis.run_method2_categorical(data)
                
            elif method_name == "method3_dml":
                logger.info("Method 3: Double Machine Learning - Causal Inference")
                logger.info("⚠️ Using DML-safe optimizations (no validation split)")
                results = analysis.run_method3_dml(data)
                
            else:
                logger.warning(f"Unknown method: {method_name}")
                continue
            
            method_time = time.time() - method_start
            logger.info(f"✅ {method_name} completed in {method_time:.2f} seconds")
            
            # Store results
            all_results[method_name] = results
            
            # Save method results
            method_output = output_path / method_name
            method_output.mkdir(exist_ok=True)
            
            # Save results as JSON
            import json
            results_path = method_output / "results.json"
            with open(results_path, 'w') as f:
                # Convert non-serializable objects
                serializable_results = {}
                for key, value in results.items():
                    try:
                        json.dumps(value)
                        serializable_results[key] = value
                    except:
                        serializable_results[key] = str(value)
                
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"   Results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"❌ Error in {method_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method_name] = {"error": str(e)}
    
    # Generate summary report
    logger.info("\n📈 Generating summary report...")
    try:
        report = analysis.generate_report(all_results)
        report_path = output_path / "summary_report.html"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate report: {e}")
    
    # Generate interpretations
    logger.info("\n💡 Generating interpretations...")
    try:
        interpretations = analysis.generate_interpretations(all_results)
        interp_path = output_path / "interpretations.md"
        
        with open(interp_path, 'w') as f:
            f.write(interpretations)
        
        logger.info(f"✅ Interpretations saved to {interp_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate interpretations: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Methods run: {list(all_results.keys())}")
    logger.info(f"Output directory: {output_path}")
    
    # Print key findings
    logger.info("\n🔑 Key Findings:")
    for method_name, results in all_results.items():
        if "error" not in results:
            logger.info(f"\n{method_name}:")
            if "hypothesis_results" in results:
                for hyp, hyp_results in results["hypothesis_results"].items():
                    if isinstance(hyp_results, dict) and "p_value" in hyp_results:
                        logger.info(f"  {hyp}: p={hyp_results['p_value']:.4f}, "
                                  f"significant={hyp_results.get('significant', False)}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run biodiversity regression analysis with method-aware optimizations"
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to input parquet file"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/ml/ml_config.yml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/regression_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "method1", "method2", "method3_dml", "glm-continuous", "glm-categorical", "dml"],
        help="Which method(s) to run"
    )
    
    parser.add_argument(
        "--spatial-sampling",
        type=int,
        default=100,
        help="Spatial grid step for sampling (100 = every 100th point)"
    )
    
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Only run diagnostic checks"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_regression_analysis(
            input_path=args.input,
            config_path=args.config,
            output_dir=args.output_dir,
            method=args.method,
            spatial_sampling=args.spatial_sampling,
            diagnostics_only=args.diagnostics_only
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
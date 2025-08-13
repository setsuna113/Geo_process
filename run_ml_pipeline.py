#!/usr/bin/env python3
"""
Machine Learning Pipeline Entry Point

Clean, type-checked entry point for the biodiversity ML regression pipeline.
Implements the 3-method hypothesis testing framework.

Usage:
    python run_ml_pipeline.py data.parquet
    python run_ml_pipeline.py data.parquet --method all
    python run_ml_pipeline.py data.parquet --method method1
    python run_ml_pipeline.py data.parquet --spatial-grid-step 10
    python run_ml_pipeline.py data.parquet --output-dir results/
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.machine_learning.ml_orchestrator import MLOrchestrator
    from src.machine_learning.config_loader import MLRegressionConfig
except ImportError as e:
    # For help display, we don't need actual imports
    if "--help" not in sys.argv and "-h" not in sys.argv:
        print(f"Error: Failed to import ML modules: {e}")
        print("Make sure H2O is installed and the geo environment is activated.")
        sys.exit(1)


MethodType = Literal["all", "method1", "method2", "method3", "glm-continuous", "glm-categorical", "dml"]


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_file = log_dir / f"ml_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def validate_inputs(data_path: Path, method: str) -> None:
    """Validate input parameters.
    
    Args:
        data_path: Path to input data file
        method: Method to run
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If method is invalid
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    valid_methods = ["all", "method1", "method2", "method3", 
                     "glm-continuous", "glm-categorical", "dml"]
    
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method: {method}. "
            f"Valid options: {', '.join(valid_methods)}"
        )


def run_ml_pipeline(
    data_path: Path,
    method: MethodType = "all",
    spatial_grid_step: Optional[int] = None,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """Run the ML regression pipeline.
    
    Args:
        data_path: Path to input parquet/csv file
        method: Which method(s) to run
        spatial_grid_step: Spatial sampling step (1 = no sampling)
        output_dir: Output directory for results
        config_path: Path to configuration file
        log_level: Logging level
        
    Returns:
        Dictionary with analysis results
        
    Raises:
        FileNotFoundError: If data file not found
        ValueError: If invalid method specified
    """
    # Set up logging
    logger = setup_logging(log_level)
    
    logger.info("="*60)
    logger.info("MACHINE LEARNING REGRESSION PIPELINE")
    logger.info("="*60)
    
    # Validate inputs
    validate_inputs(data_path, method)
    
    # Load configuration
    if config_path:
        logger.info(f"Loading config from: {config_path}")
        config = MLRegressionConfig(config_path, validate=True)
        config_path_str = str(config_path)
    else:
        logger.info("Using default configuration")
        # Use the default config path
        config_path_str = "src/config/ml/ml_config.yml"
        config = MLRegressionConfig(config_path_str, validate=True)
    
    # Initialize H2O before creating orchestrator
    logger.info("Initializing H2O...")
    import h2o
    try:
        h2o.cluster()
        logger.info("H2O already running")
    except:
        h2o.init(max_mem_size='32G', nthreads=-1)
        logger.info("H2O initialized")
    
    # Initialize orchestrator
    logger.info("Initializing ML Orchestrator...")
    orchestrator = MLOrchestrator(config_path_str)
    
    # Set output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    # Map method names for backward compatibility
    method_map = {
        "glm-continuous": "method1",  # H0-H1 testing
        "glm-categorical": "method2",  # H2 testing
        "dml": "method3"  # H0-H1 causal
    }
    
    method = method_map.get(method, method)
    
    # Run analysis
    logger.info(f"Running method: {method}")
    
    if method == "all":
        logger.info("Running complete analysis (all methods)...")
        results = orchestrator.run_complete_analysis(
            data_path=data_path,
            spatial_grid_step=spatial_grid_step,
            output_dir=output_dir
        )
    else:
        # First load and preprocess data
        logger.info(f"Loading and preprocessing data from: {data_path}")
        orchestrator.load_and_preprocess_data(
            data_path=data_path,
            spatial_grid_step=spatial_grid_step
        )
        
        # Run specific method
        if method == "method1":
            logger.info("Running Method 1: H0-H1 Testing (GLM)...")
            results = orchestrator.run_method1_h0_h1()
        elif method == "method2":
            logger.info("Running Method 2: H2 Testing (Continuous & Categorical)...")
            results = orchestrator.run_method2_h2()
        elif method == "method3":
            logger.info("Running Method 3: H0-H1 Causal Testing (DML)...")
            results = orchestrator.run_method3_dml()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Generate method-specific reports if output directory specified
        if output_dir:
            orchestrator.generate_method_reports(output_dir)
    
    # Save results if output directory specified
    if output_dir:
        import json
        results_file = output_dir / f"results_{method}_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        # Convert any non-serializable objects
        def make_serializable(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return obj
        
        serializable_results = json.loads(
            json.dumps(results, default=make_serializable)
        )
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    # Log summary
    if 'summary' in results:
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        summary = results['summary']
        if 'h2' in summary:
            logger.info("H2 Hypothesis (Temperate mismatch):")
            logger.info(f"  Consensus: {summary['h2']['consensus']}")
            logger.info(f"  Method 1: {summary['h2'].get('method1_supported', 'N/A')}")
            logger.info(f"  Method 2: {summary['h2'].get('method2_supported', 'N/A')}")
            logger.info(f"  Method 3: {summary['h2'].get('method3_supported', 'N/A')}")
    
    logger.info("\n✅ Pipeline completed successfully!")
    
    return results


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Machine Learning Regression Pipeline for Biodiversity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods
  python run_ml_pipeline.py data.parquet
  
  # Run specific method
  python run_ml_pipeline.py data.parquet --method method1
  
  # With spatial sampling
  python run_ml_pipeline.py data.parquet --spatial-grid-step 10
  
  # Save results to specific directory
  python run_ml_pipeline.py data.parquet --output-dir results/

Methods:
  all            - Run all three methods (default)
  method1        - Continuous GLM (biodiversity begets biodiversity)
  method2        - Categorical GLM (temperate mismatch)
  method3        - Double Machine Learning (causal inference)
  glm-continuous - Alias for method1
  glm-categorical - Alias for method2
  dml            - Alias for method3
"""
    )
    
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to input data file (parquet or csv)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "method1", "method2", "method3", 
                 "glm-continuous", "glm-categorical", "dml"],
        help="Method to run (default: all)"
    )
    
    parser.add_argument(
        "--spatial-grid-step",
        type=int,
        default=None,
        help="Spatial grid sampling step (e.g., 10 = every 10th point)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_ml_pipeline(
            data_path=args.data_path,
            method=args.method,
            spatial_grid_step=args.spatial_grid_step,
            output_dir=args.output_dir,
            config_path=args.config,
            log_level=args.log_level
        )
        
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
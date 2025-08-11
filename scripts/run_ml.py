#!/usr/bin/env python3
"""
Main ML Pipeline Runner - Entry point for running biodiversity ML experiments.

This script orchestrates the ML pipeline which includes:
1. Data loading and preprocessing (shared)
2. Three analysis methods (continuous GLM, categorical GLM, DML)  
3. Two model types (GLM, GBM)
4. Diagnostics and interpretation
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import traceback
import yaml

# Set environment variable to skip database initialization
os.environ['SKIP_DB_INIT'] = 'true'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MLPipelineRunner:
    """ML Pipeline runner using the RegressionPipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize ML runner with configuration."""
        self.config_path = config_path or project_root / 'config.yml'
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_dir = Path(self.config.get('paths', {}).get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'ml_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'
        
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
    
    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment configuration from config.yml."""
        ml_config = self.config.get('machine_learning', {})
        
        # Load from standard experiments
        experiments = ml_config.get('experiments', {})
        defaults = ml_config.get('defaults', {})
        
        if experiment_name not in experiments:
            available = list(experiments.keys())
            raise ValueError(
                f"Experiment '{experiment_name}' not found. "
                f"Available experiments: {available}"
            )
        
        # Get experiment specific config
        exp_config = experiments[experiment_name].copy()
        
        # Merge with defaults dict
        if defaults and isinstance(defaults, dict):
            merged_config = defaults.copy()
            merged_config.update(exp_config)
            return merged_config
        
        return exp_config
    
    def run_pipeline(self, ml_config: Dict[str, Any]) -> bool:
        """Run ML experiment using the RegressionPipeline.
        
        Args:
            ml_config: Configuration for the ML experiment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import the main pipeline orchestrator
            from src.machine_learning.pipelines.regression_pipeline import run_regression_pipeline
            
            # Validate configuration
            input_parquet = ml_config.get('input_parquet')
            if not input_parquet:
                self.logger.error("No input parquet file specified")
                return False
            
            input_path = Path(input_parquet)
            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return False
            
            # Create output directory
            output_base = Path(self.config.get('output_paths', {}).get('results_dir', 'outputs'))
            if not output_base.exists():
                output_base = Path('outputs')
            
            exp_name = ml_config.get('experiment_name', f"ml_{datetime.now():%Y%m%d_%H%M%S}")
            output_dir = output_base / 'ml_results' / exp_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Starting ML pipeline: {exp_name}")
            self.logger.info(f"Input data: {input_path}")
            self.logger.info(f"Output directory: {output_dir}")
            
            # Get config path for ML configuration
            config_path = ml_config.get('config_path')
            
            # Run the regression pipeline
            self.logger.info("Running RegressionPipeline...")
            results = run_regression_pipeline(
                data_path=str(input_path),
                config_path=config_path,
                output_dir=str(output_dir)
            )
            
            # Log summary
            if results and 'summary' in results:
                self.logger.info("Pipeline completed successfully")
                summary = results['summary']
                if 'conclusions' in summary:
                    self.logger.info("Conclusions:")
                    for conclusion in summary['conclusions']:
                        self.logger.info(f"  - {conclusion}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ML pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run ML experiments on biodiversity data using the RegressionPipeline'
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiment', '-e',
        help='Named experiment from config.yml'
    )
    
    # Input data
    parser.add_argument(
        '--input', '-i',
        help='Input parquet file path (overrides experiment config)'
    )
    
    # Configuration
    parser.add_argument(
        '--config-file',
        type=Path,
        help='Path to config.yml file'
    )
    
    parser.add_argument(
        '--config-path',
        help='Path to ML configuration file (for RegressionPipeline)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize runner
    runner = MLPipelineRunner(config_path=args.config_file)
    
    # Build ML configuration
    ml_config = {}
    
    # Start with experiment config if specified
    if args.experiment:
        try:
            ml_config = runner.load_experiment_config(args.experiment)
            runner.logger.info(f"Loaded experiment config: {args.experiment}")
        except ValueError as e:
            runner.logger.error(str(e))
            sys.exit(1)
    
    # Override with command line arguments
    if args.input:
        ml_config['input_parquet'] = args.input
    
    if args.config_path:
        ml_config['config_path'] = args.config_path
    
    # Validate we have input
    if 'input_parquet' not in ml_config:
        runner.logger.error("No input parquet file specified. Use --input or --experiment")
        sys.exit(1)
    
    # Run ML pipeline
    success = runner.run_pipeline(ml_config)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
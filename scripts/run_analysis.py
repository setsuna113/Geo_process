#!/usr/bin/env python3
"""
Standalone Spatial Analysis Runner - Runs spatial analysis experiments on biodiversity data.

This is a standalone runner that doesn't depend on the main pipeline infrastructure.
It reads parquet files and runs spatial analysis experiments (SOM, GWPCA, MaxP).
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
import gc

# Skip database initialization for standalone mode
# Set environment variable that the database connection checks
os.environ['SKIP_DB_INIT'] = 'true'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.processors.data_preparation.standalone_analysis_datasets import create_analysis_dataset
from src.processors.data_preparation.analysis_checkpoint import create_checkpointer
from src.core.enhanced_progress_manager import EnhancedProgressManager
from src.pipelines.stages.analyzer_factory import AnalyzerFactory
from src.infrastructure.monitoring.unified_monitor import UnifiedMonitor


class StandaloneAnalysisRunner:
    """Standalone analysis runner that doesn't depend on pipeline infrastructure."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize analysis runner with configuration."""
        self.config_path = config_path or project_root / 'config.yml'
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize progress manager without database
        self.progress_manager = EnhancedProgressManager(
            experiment_id=None,  # Will be set per experiment
            db_manager=None,     # No database dependency
            use_database=False   # Force memory-only backend
        )
        
        # Initialize unified monitoring system (uses memory backend without database)
        self.monitor = UnifiedMonitor(self.config, db_manager=None)
    
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
        
        log_file = log_dir / f'analysis_standalone_{datetime.now():%Y%m%d_%H%M%S}.log'
        
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
    
    def _resolve_output_path(self, configured_path: str) -> Path:
        """Resolve output path with environment detection and fallbacks."""
        path = Path(configured_path)
        
        # Detect high-performance computing environments
        hpc_indicators = [
            os.getenv('SLURM_JOB_ID'),        # SLURM workload manager
            os.getenv('PBS_JOBID'),           # PBS/Torque
            os.getenv('SGE_JOB_ID'),          # Sun Grid Engine
            os.getenv('LSB_JOBID'),           # LSF
        ]
        is_hpc_environment = any(hpc_indicators)
        
        # Handle high-performance storage paths with dynamic detection
        if path.is_absolute() and path.parts[0] == '/' and len(path.parts) > 1:
            root_dir = '/' + path.parts[1]  # e.g., '/scratch' from '/scratch/user/data'
            
            # Check if this looks like an HPC storage path by checking if:
            # 1. We're in an HPC environment, AND
            # 2. The path exists and is writable, AND  
            # 3. The filesystem has HPC characteristics (large space, fast I/O)
            if is_hpc_environment and path.parent.exists():
                try:
                    # Test if directory is writable
                    test_file = path.parent / '.write_test'
                    test_file.touch()
                    test_file.unlink()
                    
                    # Check filesystem characteristics for HPC storage
                    import shutil
                    _, _, free_space = shutil.disk_usage(path.parent)
                    
                    # HPC storage typically has > 100GB free space
                    if free_space > 100 * 1024**3:  # 100GB in bytes
                        self.logger.info(f"Using HPC storage path: {path} (detected {free_space/1024**3:.1f}GB free)")
                        return path
                    else:
                        self.logger.warning(f"Path {root_dir} exists but appears to be local storage ({free_space/1024**3:.1f}GB free), using local outputs directory")
                        return Path('outputs')
                        
                except (PermissionError, OSError) as e:
                    self.logger.warning(f"Cannot write to {root_dir}: {e}, using local outputs directory")
                    return Path('outputs')
            else:
                if not is_hpc_environment:
                    self.logger.warning(f"Not in HPC environment, path {root_dir} may not be appropriate, using local outputs directory")
                else:
                    self.logger.warning(f"HPC storage {root_dir} not available, using local outputs directory")
                return Path('outputs')
        
        # Handle other absolute paths
        if path.is_absolute():
            # Ensure parent directory is writable
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = path.parent / '.write_test'
                test_file.touch()
                test_file.unlink()
                return path
            except (PermissionError, OSError) as e:
                self.logger.warning(f"Cannot write to {path}: {e}, using local outputs directory")
                return Path('outputs')
        
        # Relative paths are fine as-is
        return path
    
    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment configuration from config.yml."""
        analysis_config = self.config.get('spatial_analysis', {})
        experiments = analysis_config.get('experiments', {})
        defaults = analysis_config.get('defaults', {})
        
        if experiment_name not in experiments:
            available = list(experiments.keys())
            raise ValueError(
                f"Experiment '{experiment_name}' not found. "
                f"Available experiments: {available}"
            )
        
        # Get experiment specific config
        exp_config = experiments[experiment_name].copy()
        
        # Merge with defaults
        merged_config = defaults.copy()
        merged_config.update(exp_config)
        
        return merged_config
    
    def run_analysis_experiment(self, analysis_config: Dict[str, Any]) -> bool:
        """Run analysis experiment with given configuration."""
        try:
            # Validate configuration
            input_parquet = analysis_config.get('input_parquet')
            if not input_parquet:
                self.logger.error("No input parquet file specified")
                return False
            
            input_path = Path(input_parquet)
            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return False
            
            analysis_method = analysis_config.get('method')
            if not analysis_method:
                self.logger.error("No analysis method specified")
                return False
            
            # Create output directory with proper path validation
            output_base_config = self.config.get('output_paths', {}).get('results_dir', 'outputs')
            output_base = self._resolve_output_path(output_base_config)
            
            exp_name = analysis_config.get('experiment_name', f"{analysis_method}_{datetime.now():%Y%m%d_%H%M%S}")
            output_dir = output_base / 'analysis_results' / analysis_method / exp_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Starting analysis experiment: {exp_name}")
            self.logger.info(f"Method: {analysis_method}")
            self.logger.info(f"Input data: {input_path}")
            self.logger.info(f"Output directory: {output_dir}")
            
            # Set experiment ID for progress tracking
            self.progress_manager.set_experiment(exp_name)
            
            # Start unified monitoring (memory, CPU, system metrics)
            self.monitor.start(exp_name)
            
            # Create dataset
            self.logger.info("Loading dataset...")
            data_source = analysis_config.get('data_source', 'auto')
            chunk_size = analysis_config.get('chunk_size', 10000)
            
            dataset = create_analysis_dataset(
                input_path=input_path,
                data_source=data_source,
                chunk_size=chunk_size
            )
            
            # Get dataset info
            dataset_info = dataset.load_info()
            column_count = dataset_info.metadata.get('column_count', 'unknown')
            self.logger.info(f"Dataset: {dataset_info.record_count:,} records, "
                           f"{dataset_info.size_mb:.2f} MB, "
                           f"{column_count} columns")
            
            # Set up checkpointing if enabled
            checkpointer = None
            resume_info = None
            if analysis_config.get('enable_checkpointing', True):
                checkpointer = create_checkpointer(
                    output_dir=output_dir,
                    experiment_name=exp_name,
                    analysis_method=analysis_method
                )
                
                # Check for existing checkpoint if resume flag is set
                if analysis_config.get('resume', False):
                    latest_checkpoint = checkpointer.load_latest_checkpoint()
                    if latest_checkpoint:
                        self.logger.info("Found existing checkpoint, resuming...")
                        resume_info = checkpointer.resume_from_checkpoint(latest_checkpoint)
                    else:
                        self.logger.warning("Resume requested but no checkpoint found, starting fresh")
            
            # Create analyzer (database-free)
            self.logger.info("Creating analyzer...")
            
            # Use config wrapper that mimics the Config class
            config_wrapper = ConfigWrapper(self.config)
            analyzer = AnalyzerFactory.create(
                analysis_method,
                config_wrapper,
                db=None  # No database dependency
            )
            
            # Restore analyzer state if resuming from checkpoint
            if resume_info and 'method_state' in resume_info:
                self.logger.info("Restoring analyzer state from checkpoint...")
                analyzer.load_checkpoint_state(resume_info['method_state'])
            
            # Progress tracking is handled by UnifiedMonitor
            self.logger.info("Progress tracking initialized")
            
            # Get analysis parameters
            params = self._get_analysis_parameters(analysis_config, analysis_method)
            self.logger.info(f"Analysis parameters: {params}")
            
            # Perform analysis with checkpointing
            self.logger.info("Starting analysis computation...")
            results = self._run_analysis_with_checkpointing(
                analyzer=analyzer,
                dataset=dataset,
                params=params,
                checkpointer=checkpointer,
                resume_info=resume_info,
                analysis_config=analysis_config
            )
            
            # Save results
            if analysis_config.get('save_results', True):
                output_path = self._save_results(
                    results=results,
                    analysis_method=analysis_method,
                    exp_name=exp_name,
                    output_dir=output_dir,
                    analyzer=analyzer
                )
                self.logger.info(f"Results saved to: {output_path}")
            
            # Save metrics 
            metrics = self._extract_metrics(results, params, dataset_info)
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Cleanup
            dataset.cleanup()
            if checkpointer:
                checkpointer.cleanup_old_checkpoints(keep_last=3)
            
            # Stop unified monitoring
            self.monitor.stop()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Analysis experiment completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis experiment failed: {e}")
            self.logger.error(traceback.format_exc())
            # Ensure monitoring is stopped even on failure
            try:
                self.monitor.stop()
            except Exception as monitor_error:
                self.logger.debug(f"Error stopping monitor: {monitor_error}")
            return False
    
    def _setup_progress_tracking(self, analysis_method: str, exp_name: str):
        """Set up progress tracking for the analysis."""
        # Create main progress node
        self.progress_manager.create_node(
            node_id="analysis",
            name=f"{analysis_method} Analysis",
            level="pipeline",
            total_units=100
        )
        
        # Create method-specific node
        self.progress_manager.create_node(
            node_id=f"analysis/{analysis_method}",
            parent_id="analysis",
            name=f"{analysis_method.upper()} Processing",
            level="stage",
            total_units=100
        )
        
        self.logger.info(f"Progress tracking initialized for {analysis_method}")
    
    def _get_analysis_parameters(self, analysis_config: Dict[str, Any], analysis_method: str) -> Dict[str, Any]:
        """Extract analysis-specific parameters from configuration.
        
        IMPORTANT: This should ONLY pass generic parameters like output paths.
        Method-specific parameters (grid_size, learning_rate, etc.) are BANNED.
        Each analyzer reads its own config from its method-specific config file.
        """
        # ONLY pass generic parameters that all methods need
        params = {
            'method': analysis_method,  # Include the method name for metrics
            'save_results': analysis_config.get('save_results', True),
            'output_dir': analysis_config.get('output_dir', f'outputs/analysis_results/{analysis_method}'),
            'experiment_name': analysis_config.get('experiment_name', 'default'),
            'resume': analysis_config.get('resume', False)  # Pass resume flag
        }
        
        # Log warning if method-specific parameters are detected
        banned_params = ['grid_size', 'iterations', 'learning_rate', 'sigma', 
                        'convergence_threshold', 'max_iterations', 'cv_folds',
                        'observed_columns', 'predicted_columns', 'convergence_method']
        
        found_banned = [p for p in banned_params if p in analysis_config]
        if found_banned:
            self.logger.warning(f"BANNED method-specific parameters found in config.yml: {found_banned}")
            self.logger.warning(f"These should be in src/config/{analysis_method}/{analysis_method}_config.yml")
            self.logger.warning("Ignoring these parameters - using method-specific config file instead")
            
        
        if analysis_method == 'cpwopt':
            # CP-WOPT tensor factorization parameters
            cpwopt_params = ['rank', 'max_iterations', 'convergence_threshold', 
                           'initialization', 'use_minibatch', 'minibatch_size',
                           'regularization', 'random_state', 'verbose',
                           'preprocessing', 'generate_plots', 'save_factors']
            params.update({
                k: v for k, v in analysis_config.items() 
                if k in cpwopt_params and v is not None
            })
            # Add preprocessing params if not specified
            if 'preprocessing' not in params:
                params['preprocessing'] = {
                    'lat_bins': analysis_config.get('lat_bins', 20),
                    'lon_bins': analysis_config.get('lon_bins', 20),
                    'aggregation': analysis_config.get('aggregation', 'sum'),
                    'transform': analysis_config.get('transform', 'log1p')
                }
        
        return params
    
    def _run_analysis_with_checkpointing(
        self,
        analyzer,
        dataset,
        params: Dict[str, Any],
        checkpointer,
        resume_info: Optional[Dict[str, Any]],
        analysis_config: Dict[str, Any]
    ):
        """Run analysis with checkpointing support."""
        
        # For now, run analysis normally without chunked checkpointing
        # This can be enhanced later with more sophisticated chunked processing
        
        # Set up progress callback with error handling
        def progress_callback(message: str, progress: float):
            # Progress is given as a value between 0 and 1
            progress_percent = progress * 100
            try:
                # Try to update progress, but don't fail if it doesn't work
                if hasattr(self.progress_manager, 'update_progress'):
                    self.progress_manager.update_progress(
                        node_id=f"analysis/{params.get('method', 'analysis')}",
                        completed_units=int(progress_percent),
                        status="running",
                        metadata={"message": message, "progress": progress_percent}
                    )
            except Exception as e:
                # Log progress errors but don't fail the analysis
                self.logger.debug(f"Progress callback error (non-fatal): {e}")
            
            # Always log progress to console
            self.logger.info(f"Analysis progress: {message} - {progress_percent:.1f}%")
        
        # Set progress callback if analyzer supports it
        if hasattr(analyzer, 'set_progress_callback'):
            analyzer.set_progress_callback(progress_callback)
        
        # Run the analysis - pass the parquet file path, not the dataset object  
        input_path = analysis_config.get('input_parquet')
        results = analyzer.analyze(input_path, **params)
        
        # Update progress to complete (with error handling)
        try:
            if hasattr(self.progress_manager, 'update_progress'):
                self.progress_manager.update_progress(
                    node_id=f"analysis/{params.get('method', 'analysis')}",
                    completed_units=100,
                    status="completed"
                )
        except Exception as e:
            self.logger.debug(f"Final progress update error (non-fatal): {e}")
        
        self.logger.info("Analysis computation completed successfully!")
        
        return results
    
    def _save_results(self, results, analysis_method: str, exp_name: str, output_dir: Path, analyzer) -> Path:
        """Save analysis results to output directory."""
        if hasattr(analyzer, 'save_results') and callable(getattr(analyzer, 'save_results')):
            return analyzer.save_results(results, exp_name, output_dir)
        else:
            # Fallback: save results as JSON/pickle
            output_path = output_dir / f"{analysis_method}_results.json"
            
            # Try to save as JSON first
            try:
                if hasattr(results, 'to_dict'):
                    results_dict = results.to_dict()
                elif hasattr(results, '__dict__'):
                    results_dict = results.__dict__
                else:
                    results_dict = {'results': str(results)}
                
                with open(output_path, 'w') as f:
                    json.dump(results_dict, f, indent=2, default=str)
                    
            except Exception as e:
                self.logger.warning(f"Could not save as JSON: {e}, saving as pickle")
                pickle_path = output_dir / f"{analysis_method}_results.pkl"
                
                import pickle
                with open(pickle_path, 'wb') as f:
                    pickle.dump(results, f)
                
                return pickle_path
            
            return output_path
    
    def _extract_metrics(self, results, params: Dict[str, Any], dataset_info) -> Dict[str, Any]:
        """Extract metrics from analysis results."""
        metrics = {
            'analysis_method': params.get('method', 'unknown'),
            'parameters': params,
            'dataset_info': {
                'record_count': dataset_info.record_count,
                'column_count': dataset_info.metadata.get('column_count', 'unknown'),
                'size_mb': dataset_info.size_mb
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract method-specific metrics
        if hasattr(results, 'statistics'):
            metrics['analysis_stats'] = results.statistics
        elif hasattr(results, 'metadata'):
            metrics['analysis_metadata'] = results.metadata
        
        return metrics


class ConfigWrapper:
    """Wrapper to make dictionary config behave like Config class."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.settings = config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run standalone spatial analysis experiments on biodiversity data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Inspection:
  python run_analysis.py --list-methods
  python run_analysis.py --show-config cpwopt
  python run_analysis.py --show-env-vars cpwopt
  python run_analysis.py --compare-config cpwopt
  python run_analysis.py --export-config cpwopt --output cpwopt.json

Running Experiments:
  python run_analysis.py --experiment test_cpwopt
  python run_analysis.py --experiment test_som --resume
  CPWOPT_ALGORITHM_RANK=10 python run_analysis.py --experiment sparse_cpwopt
        """
    )
    
    # Create argument groups
    analysis_group = parser.add_argument_group('Analysis Options')
    config_group = parser.add_argument_group('Configuration Inspection')
    
    # Experiment selection
    analysis_group.add_argument(
        '--experiment', '-e',
        help='Named experiment from config.yml spatial_analysis.experiments'
    )
    
    # Input data override
    analysis_group.add_argument(
        '--input', '-i',
        help='Input parquet file path (overrides experiment config)'
    )
    
    # Method override
    analysis_group.add_argument(
        '--method', '-m',
        choices=['som', 'gwpca', 'maxp_regions', 'cpwopt', 'kmeans', 'kmeans_hierarchical'],
        help='Analysis method (overrides experiment config)'
    )
    
    # Configuration file
    analysis_group.add_argument(
        '--config-file',
        type=Path,
        help='Path to config.yml file'
    )
    
    # Logging level
    analysis_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    # Resume from checkpoint
    analysis_group.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint if available'
    )
    
    # Configuration inspection arguments
    config_group.add_argument(
        '--list-methods',
        action='store_true',
        help='List all available analysis methods'
    )
    config_group.add_argument(
        '--show-config',
        metavar='METHOD',
        help='Show complete configuration for a method'
    )
    config_group.add_argument(
        '--show-env-vars',
        metavar='METHOD',
        help='Show environment variables for a method'
    )
    config_group.add_argument(
        '--compare-config',
        metavar='METHOD',
        help='Show differences from default configuration'
    )
    config_group.add_argument(
        '--export-config',
        metavar='METHOD',
        help='Export current configuration to file'
    )
    config_group.add_argument(
        '--output',
        help='Output file for --export-config'
    )
    
    args = parser.parse_args()
    
    # Handle configuration inspection commands
    if args.list_methods:
        from src.config.method_registry import MethodConfigRegistry
        print("\nAvailable analysis methods:")
        print("=" * 30)
        for method in MethodConfigRegistry.list_methods():
            print(f"  - {method}")
        print("\nUse --show-config <method> to see configuration details")
        return
    
    if args.show_config:
        from src.utils.config_inspector import ConfigInspector
        ConfigInspector.show_method_config(args.show_config)
        return
    
    if args.show_env_vars:
        from src.utils.config_inspector import ConfigInspector
        ConfigInspector.show_env_vars(args.show_env_vars)
        return
    
    if args.compare_config:
        from src.utils.config_inspector import ConfigInspector
        ConfigInspector.compare_with_defaults(args.compare_config)
        return
    
    if args.export_config:
        from src.utils.config_inspector import ConfigInspector
        output_file = Path(args.output) if args.output else None
        ConfigInspector.export_config(args.export_config, output_file)
        return
    
    # Regular experiment execution requires --experiment
    if not args.experiment:
        parser.error("--experiment is required for running analysis")
        return
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize runner
    runner = StandaloneAnalysisRunner(config_path=args.config_file)
    
    try:
        # Load experiment configuration
        analysis_config = runner.load_experiment_config(args.experiment)
        runner.logger.info(f"Loaded experiment config: {args.experiment}")
        
        # Override with command line arguments
        if args.input:
            analysis_config['input_parquet'] = args.input
        
        if args.method:
            analysis_config['method'] = args.method
        
        if args.resume:
            analysis_config['resume'] = True
        
        # Validate required fields
        if 'input_parquet' not in analysis_config:
            runner.logger.error("No input parquet file specified. Use --input or configure in experiment")
            sys.exit(1)
        
        if 'method' not in analysis_config:
            runner.logger.error("No analysis method specified. Use --method or configure in experiment")
            sys.exit(1)
        
        # Add experiment name
        analysis_config['experiment_name'] = args.experiment
        
        # Run experiment
        success = runner.run_analysis_experiment(analysis_config)
        sys.exit(0 if success else 1)
        
    except ValueError as e:
        runner.logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        runner.logger.error(f"Unexpected error: {e}")
        runner.logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
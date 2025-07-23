#!/usr/bin/env python3
"""
Unified resampling pipeline script.

This script extends the functionality of process_richness_datasets.py to support:
1. Multiple dataset resampling to uniform resolution
2. Database storage of resampled datasets  
3. Coordinate alignment and merging of resampled datasets
4. Scalable pipeline for N datasets → uniform database → SOM analysis

Inherits argument parsing and memory management from the original script.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.setup import setup_database
from src.pipelines.unified_resampling.pipeline_orchestrator import UnifiedResamplingPipeline
from src.pipelines.unified_resampling.validation_checks import ValidationChecks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments with extensions for resampling pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified resampling pipeline for multi-dataset spatial analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Memory control options (inherited from original script)
    parser.add_argument('--max-samples', type=int, default=500000,
                        help='Maximum samples for in-memory processing')
    parser.add_argument('--sampling-strategy', choices=['random', 'stratified', 'grid'],
                        default='stratified', help='Subsampling strategy for large datasets')
    parser.add_argument('--memory-limit', type=float, default=8.0,
                        help='Memory limit in GB')
    parser.add_argument('--batch-processing', action='store_true',
                        help='Use batch processing for very large datasets')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Chunk size for batch processing')
    
    # SOM parameters (inherited)
    parser.add_argument('--som-grid-size', type=int, nargs=2, default=[10, 10],
                        help='SOM grid dimensions [width height]')
    parser.add_argument('--som-iterations', type=int, default=1000,
                        help='Number of SOM training iterations')
    
    # NEW: Resampling-specific options
    parser.add_argument('--target-resolution', type=float, default=None,
                        help='Override target resolution (degrees) from config')
    parser.add_argument('--resampling-engine', choices=['numpy', 'gdal'], default=None,
                        help='Override resampling engine from config')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip datasets that are already resampled')
    parser.add_argument('--validate-inputs', action='store_true', default=True,
                        help='Validate input datasets and configuration')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configuration and estimate requirements without processing')
    
    # Pipeline control options
    parser.add_argument('--skip-resampling', action='store_true',
                        help='Skip resampling phase (use existing resampled data)')
    parser.add_argument('--skip-som', action='store_true',
                        help='Skip SOM analysis (only perform resampling and merging)')
    parser.add_argument('--cleanup-intermediate', action='store_true',
                        help='Clean up intermediate resampled data after completion')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experimental run (auto-generated if not provided)')
    parser.add_argument('--experiment-description', type=str, default=None,
                        help='Description for this experimental run')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='outputs/unified_resampling',
                        help='Output directory for results')
    parser.add_argument('--progress-interval', type=int, default=10,
                        help='Progress reporting interval (seconds)')
    
    args = parser.parse_args()
    
    # Validate inputs (inherited logic)
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.memory_limit <= 0:
        parser.error("--memory-limit must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.som_grid_size[0] <= 0 or args.som_grid_size[1] <= 0:
        parser.error("--som-grid-size dimensions must be positive")
    if args.som_iterations <= 0:
        parser.error("--som-iterations must be positive")
    
    # NEW: Validate resampling-specific inputs
    if args.target_resolution is not None and args.target_resolution <= 0:
        parser.error("--target-resolution must be positive")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    
    return args


def setup_progress_monitoring(pipeline, args):
    """Setup progress monitoring with different output options."""
    
    def console_progress_callback(workflow_state):
        """Console progress callback."""
        progress = workflow_state.get('overall_progress', 0)
        current_step = workflow_state.get('current_step', 'unknown')
        status_message = workflow_state.get('status_message', '')
        
        # Print concise progress updates
        if progress >= 100:
            logger.info(f"✅ Pipeline completed: {current_step}")
        else:
            logger.info(f"🔄 {progress:.1f}% - {current_step}: {status_message}")
    
    # Register progress callback
    pipeline.resampling_workflow.register_progress_callback(console_progress_callback)
    
    # Setup periodic progress reports if requested
    if args.progress_interval > 0:
        import threading
        import time
        
        def periodic_progress_report():
            """Print periodic progress reports."""
            while not getattr(threading.current_thread(), "stop_requested", False):
                try:
                    workflow_state = pipeline.resampling_workflow.get_workflow_status()
                    if workflow_state.get('status') in ['completed', 'failed']:
                        break
                    
                    report = pipeline.resampling_workflow.create_progress_report()
                    logger.info(f"\n--- Progress Report ---\n{report}\n--- End Report ---")
                    
                    time.sleep(args.progress_interval)
                except Exception as e:
                    logger.warning(f"Progress report failed: {e}")
                    break
        
        progress_thread = threading.Thread(target=periodic_progress_report, daemon=True)
        progress_thread.start()
        
        return progress_thread
    
    return None


def main():
    """Main processing pipeline with resampling integration."""
    args = parse_arguments()
    
    logger.info("🚀 Starting unified resampling pipeline")
    logger.info(f"Memory limit: {args.memory_limit} GB")
    logger.info(f"Max samples: {args.max_samples:,}")
    logger.info(f"Target resolution override: {args.target_resolution}")
    
    # Initialize configuration and database
    config = Config()
    
    # Apply command-line overrides to config
    if args.target_resolution is not None:
        config.config.setdefault('resampling', {})['target_resolution'] = args.target_resolution
    
    if args.resampling_engine is not None:
        config.config.setdefault('resampling', {})['engine'] = args.resampling_engine
    
    # Update processing configuration (inherited logic)
    processing_config = config.config.setdefault('processing', {})
    subsampling_config = processing_config.setdefault('subsampling', {})
    subsampling_config.update({
        'enabled': True,
        'max_samples': args.max_samples,
        'strategy': args.sampling_strategy,
        'memory_limit_gb': args.memory_limit
    })
    
    # Update SOM configuration (inherited logic)
    som_config = config.config.setdefault('som_analysis', {})
    som_config.update({
        'max_pixels_in_memory': min(args.max_samples, 1000000),
        'use_memory_mapping': args.memory_limit <= 16.0,
        'batch_training': {
            'enabled': args.batch_processing,
            'batch_size': args.chunk_size
        },
        'default_grid_size': args.som_grid_size,
        'iterations': args.som_iterations
    })
    
    # Initialize database
    db = DatabaseManager()
    
    # Ensure database is ready
    logger.info("Checking database setup...")
    try:
        if not db.test_connection():
            logger.error("Database connection failed")
            return False
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False
    
    # Validation phase
    if args.validate_inputs:
        logger.info("🔍 Validating pipeline configuration...")
        validator = ValidationChecks(config)
        
        # Validate configuration
        is_valid, error_msg = validator.validate_resampling_config()
        if not is_valid:
            logger.error(f"Resampling configuration invalid: {error_msg}")
            return False
        
        is_valid, error_msg = validator.validate_datasets_config()
        if not is_valid:
            logger.error(f"Datasets configuration invalid: {error_msg}")
            return False
        
        # Validate database
        is_valid, error_msg = validator.validate_database_connection(db)
        if not is_valid:
            logger.error(f"Database validation failed: {error_msg}")
            return False
        
        # System requirements check
        requirements_met, warnings = validator.validate_system_requirements()
        for warning in warnings:
            if warning.startswith("CRITICAL"):
                logger.error(warning)
            else:
                logger.warning(warning)
        
        if not requirements_met:
            logger.error("Critical system requirements not met")
            return False
        
        # Print validation report
        report = validator.create_validation_report()
        logger.info(f"\n--- Validation Report ---\n{report}\n--- End Report ---")
        
        logger.info("✅ All validations passed")
    
    # Dry run mode
    if args.dry_run:
        logger.info("🔍 Dry run mode - estimating processing requirements...")
        
        from src.pipelines.unified_resampling.dataset_processor import DatasetProcessor
        dataset_processor = DatasetProcessor(config, db)
        
        datasets_config = config.get('datasets.target_datasets', [])
        requirements = dataset_processor.estimate_processing_requirements(datasets_config)
        
        logger.info("\n--- Processing Estimates ---")
        logger.info(f"Datasets to process: {requirements['datasets_count']}")
        logger.info(f"Target resolution: {requirements['target_resolution']}°")
        logger.info(f"Estimated dimensions per dataset: {requirements['estimated_dimensions']['width']} x {requirements['estimated_dimensions']['height']}")
        logger.info(f"Estimated memory usage: {requirements['memory_estimates']['total_gb']:.2f} GB")
        logger.info(f"Estimated processing time: {requirements['time_estimates']['total_hours']:.1f} hours")
        logger.info(f"Estimated disk usage: {requirements['disk_space_estimates']['database_storage_mb']:.0f} MB")
        logger.info("--- End Estimates ---\n")
        
        logger.info("✅ Dry run completed - no data was processed")
        return True
    
    try:
        # Initialize pipeline
        pipeline = UnifiedResamplingPipeline(config, db)
        
        # Setup progress monitoring
        progress_thread = setup_progress_monitoring(pipeline, args)
        
        # Generate experiment name if not provided
        experiment_name = args.experiment_name
        if not experiment_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"UnifiedResampling_{timestamp}"
        
        # Run the complete pipeline
        logger.info(f"🎬 Starting pipeline execution: {experiment_name}")
        
        results = pipeline.run_complete_pipeline(
            experiment_name=experiment_name,
            description=args.experiment_description,
            skip_existing=args.skip_existing
        )
        
        # Stop progress monitoring
        if progress_thread:
            progress_thread.stop_requested = True
            progress_thread.join(timeout=1)
        
        # Print results summary
        logger.info("\n--- Pipeline Results Summary ---")
        logger.info(f"Experiment ID: {results['experiment_id']}")
        logger.info(f"Datasets processed: {len(results['resampled_datasets'])}")
        logger.info(f"Merged dataset shape: {results['merged_dataset']['shape']}")
        logger.info(f"SOM results saved to: {results['som_analysis']['saved_path']}")
        logger.info("--- End Summary ---\n")
        
        # Cleanup if requested
        if args.cleanup_intermediate:
            logger.info("🧹 Cleaning up intermediate data...")
            pipeline.cleanup_intermediate_data(keep_resampled=False)
        
        logger.info("🎉 Unified resampling pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
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
import psutil

# Add project root to path for imports (script is in src/pipelines/unified_resampling/scripts/)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

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
    parser.add_argument('--test-mode', action='store_true',
                        help='Force test mode (use defaults.py, ignore config.yml)')
    
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


def pre_flight_checks(config: Config, args) -> bool:
    """Comprehensive pre-flight checks with intelligent mode detection."""
    print("🔍 Running pre-flight checks...")
    
    # 1. Mode detection and reporting
    is_test_mode = config._is_test_mode()
    if is_test_mode:
        print("🧪 TEST MODE DETECTED")
        print(f"   Database: localhost:{config.get('database.port')} (test)")
        print(f"   Data directory: {config.get('paths.data_dir')} (defaults.py)")
    else:
        print("🏭 PRODUCTION MODE DETECTED")  
        print(f"   Database: {config.get('database.host')}:{config.get('database.port')} (config.yml)")
        print(f"   Data directory: {config.get('paths.data_dir')} (config.yml)")
    
    # 2. Dataset file validation (using individual paths)
    datasets = config.get('datasets.target_datasets', [])
    if not datasets:
        print("❌ No target datasets configured")
        return False
    
    missing_files = []
    found_files = []
    
    for dataset in datasets:
        if not dataset.get('enabled', True):
            continue
            
        dataset_path = Path(dataset.get('path', ''))
        dataset_name = dataset.get('name', 'unknown')
        
        if dataset_path.exists():
            size = dataset_path.stat().st_size / (1024**2)  # MB
            found_files.append(f"{dataset_name}: {dataset_path.name} ({size:.1f}MB)")
        else:
            missing_files.append(f"{dataset_name}: {dataset_path}")
    
    if found_files:
        print("✅ Dataset files found:")
        for file_info in found_files:
            print(f"   - {file_info}")
    
    if missing_files:
        print("❌ Missing dataset files:")
        for file_info in missing_files:
            print(f"   - {file_info}")
        return False
    
    # 3. Database connectivity check
    try:
        print("🔍 Testing database connection...")
        from src.database.connection import DatabaseManager
        db = DatabaseManager()
        if db.test_connection():
            print(f"✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database error: {e}")
        if is_test_mode:
            print("💡 Start PostgreSQL: sudo systemctl start postgresql")
        return False
    
    # 4. System resource validation
    if args.dry_run:
        print("🔍 Dry run mode - skipping resource checks")
    else:
        # Check available memory
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        required_gb = args.memory_limit
        
        print(f"🖥️  System memory: {available_gb:.1f}GB available")
        if available_gb < required_gb:
            print(f"⚠️  Warning: Requested {required_gb}GB but only {available_gb:.1f}GB available")
        else:
            print(f"✅ Memory check passed ({required_gb}GB requested)")
    
    print("✅ All pre-flight checks passed!")
    return True


def main():
    """Main processing pipeline with resampling integration."""
    args = parse_arguments()
    
    logger.info("🚀 Starting unified resampling pipeline")
    logger.info(f"Memory limit: {args.memory_limit} GB")
    logger.info(f"Max samples: {args.max_samples:,}")
    logger.info(f"Target resolution override: {args.target_resolution}")
    
    # Initialize configuration and database
    # If --test-mode is specified, force test mode detection
    if args.test_mode:
        import sys
        sys.modules['pytest'] = type(sys)('pytest')  # Force test mode
        print("🧪 Test mode forced via --test-mode flag")
    
    config = Config()
    
    # Run comprehensive pre-flight checks
    if not pre_flight_checks(config, args):
        logger.error("❌ Pre-flight checks failed. Exiting.")
        return False
    
    # Apply command-line overrides to config
    if args.target_resolution is not None:
        config.settings.setdefault('resampling', {})['target_resolution'] = args.target_resolution
    
    if args.resampling_engine is not None:
        config.settings.setdefault('resampling', {})['engine'] = args.resampling_engine
    
    # Update processing configuration (inherited logic)
    processing_config = config.settings.setdefault('processing', {})
    subsampling_config = processing_config.setdefault('subsampling', {})
    subsampling_config.update({
        'enabled': True,
        'max_samples': args.max_samples,
        'strategy': args.sampling_strategy,
        'memory_limit_gb': args.memory_limit
    })
    
    # Update SOM configuration (inherited logic)
    som_config = config.settings.setdefault('som_analysis', {})
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
        
        # Validate database and create schema if needed
        is_valid, error_msg = validator.validate_database_connection(db)
        if not is_valid:
            if "does not exist" in error_msg:
                logger.info("🔧 Creating missing database schema...")
                try:
                    from src.database.schema import DatabaseSchema
                    schema = DatabaseSchema()
                    schema.create_all_tables()
                    logger.info("✅ Database schema created successfully")
                    # Re-validate after schema creation
                    is_valid, error_msg = validator.validate_database_connection(db)
                    if not is_valid:
                        logger.error(f"Database validation still failed after schema creation: {error_msg}")
                        return False
                except Exception as e:
                    logger.error(f"Failed to create database schema: {e}")
                    return False
            else:
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
    sys.exit(0 if success else 1)# src/pipelines/unified_resampling/scripts/run_unified_resampling.py
#!/usr/bin/env python3
"""
Enhanced unified resampling pipeline script with full orchestration.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineOrchestrator
from src.pipelines.stages.load_stage import DataLoadStage
from src.pipelines.stages.resample_stage import ResampleStage
from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.analysis_stage import AnalysisStage
from src.pipelines.stages.export_stage import ExportStage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production-ready unified resampling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Existing arguments preserved
    parser.add_argument('--max-samples', type=int, default=500000,
                        help='Maximum samples for in-memory processing')
    parser.add_argument('--memory-limit', type=float, default=8.0,
                        help='Memory limit in GB')
    parser.add_argument('--target-resolution', type=float, default=None,
                        help='Override target resolution (degrees)')
    
    # Pipeline control
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experimental run')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--skip-stages', nargs='*', default=[],
                        help='Stages to skip')
    
    # Monitoring
    parser.add_argument('--monitor-interval', type=int, default=5,
                        help='Monitoring interval in seconds')
    parser.add_argument('--progress-report', action='store_true',
                        help='Show detailed progress reports')
    
    return parser.parse_args()


def setup_pipeline_callbacks(orchestrator, args):
    """Setup monitoring callbacks."""
    
    def memory_warning_callback(usage):
        logger.warning(f"Memory warning: {usage['process_rss_gb']:.2f}GB used")
    
    def memory_critical_callback(usage):
        logger.error(f"CRITICAL memory usage: {usage['process_rss_gb']:.2f}GB")
        # Could trigger emergency cleanup here
    
    def progress_callback(progress):
        if args.progress_report:
            logger.info(f"Progress: {progress['overall_progress']:.1f}% - "
                       f"Stage: {progress.get('current_stage', 'None')}")
    
    # Register callbacks
    orchestrator.memory_monitor.register_warning_callback(memory_warning_callback)
    orchestrator.memory_monitor.register_critical_callback(memory_critical_callback)
    orchestrator.progress_tracker.register_callback(progress_callback)


def main():
    """Main pipeline execution."""
    args = parse_arguments()
    
    logger.info("🚀 Starting production unified resampling pipeline")
    
    # Initialize configuration and database
    config = Config()
    
    # Apply overrides
    if args.target_resolution:
        config.settings.setdefault('resampling', {})['target_resolution'] = args.target_resolution
    
    config.settings.setdefault('pipeline', {})['memory_limit_gb'] = args.memory_limit
    
    # Update memory-aware settings
    config.settings.setdefault('processing', {}).setdefault('subsampling', {}).update({
        'max_samples': args.max_samples,
        'memory_limit_gb': args.memory_limit
    })
    
    # Initialize database
    db = DatabaseManager()
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config, db)
        
        # Configure pipeline stages
        stages = [
            DataLoadStage,
            ResampleStage,
            MergeStage,
            AnalysisStage,
            ExportStage
        ]
        
        # Skip requested stages
        if args.skip_stages:
            stages = [s for s in stages if s().name not in args.skip_stages]
        
        orchestrator.configure_pipeline(stages)
        
        # Setup callbacks
        setup_pipeline_callbacks(orchestrator, args)
        
        # Validate pipeline
        is_valid, errors = orchestrator.validate_pipeline()
        if not is_valid:
            logger.error(f"Pipeline validation failed: {errors}")
            return False
        
        # Generate experiment name
        experiment_name = args.experiment_name
        if not experiment_name:
            experiment_name = f"UnifiedResampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run pipeline
        logger.info(f"Starting experiment: {experiment_name}")
        
        results = orchestrator.run_pipeline(
            experiment_name=experiment_name,
            resume_from_checkpoint=args.resume,
            description="Production unified resampling with monitoring and recovery"
        )
        
        # Print results summary
        logger.info("\n" + "="*50)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Experiment ID: {results.get('experiment_id')}")
        logger.info(f"Total execution time: {results.get('execution_time', 0):.2f}s")
        logger.info(f"Stages completed: {results.get('stages_completed')}/{results.get('total_stages')}")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Ensure cleanup
        if 'orchestrator' in locals():
            orchestrator.stop_pipeline()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
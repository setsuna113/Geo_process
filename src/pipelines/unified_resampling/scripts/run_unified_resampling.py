# src/pipelines/unified_resampling/scripts/run_unified_resampling.py
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
    
    # Analysis method selection
    parser.add_argument('--analysis-method', 
                       choices=['som', 'maxp_regions', 'gwpca', 'all'], 
                       default='som',
                       help='Spatial analysis method to use')
    
    # Data source for analysis
    parser.add_argument('--analysis-data-source',
                       choices=['database', 'csv'],
                       default='database',
                       help='Data source for analysis stage')
    
    # Analysis-specific parameters
    parser.add_argument('--som-grid-size', type=int, nargs=2, default=[8, 8],
                       help='SOM grid dimensions [width height]')
    parser.add_argument('--maxp-regions', type=int, default=10,
                       help='Number of regions for MaxP')
    parser.add_argument('--gwpca-components', type=int, default=3,
                       help='Number of components for GWPCA')
    
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
    
    config.settings.setdefault('analysis', {})['data_source'] = args.analysis_data_source
    
    # Configure analysis-specific parameters
    if args.analysis_method in ['som', 'all']:
        config.settings.setdefault('som_analysis', {})['default_grid_size'] = args.som_grid_size
    
    if args.analysis_method in ['maxp_regions', 'all']:
        config.settings.setdefault('maxp_regions_analysis', {})['n_regions'] = args.maxp_regions
    
    if args.analysis_method in ['gwpca', 'all']:
        config.settings.setdefault('gwpca_analysis', {})['n_components'] = args.gwpca_components

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
            ExportStage  # Now before analysis
        ]
        
        # Add analysis stages based on selection
        if args.analysis_method == 'all':
            # Run all three analyses
            stages.extend([
                lambda: AnalysisStage('som'),
                lambda: AnalysisStage('maxp_regions'),
                lambda: AnalysisStage('gwpca')
            ])
        else:
            # Run selected analysis
            stages.append(lambda: AnalysisStage(args.analysis_method))
        
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
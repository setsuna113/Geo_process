# Add these imports to the existing imports
from src.core.progress_manager import get_progress_manager, console_progress_callback
from src.core.checkpoint_manager import get_checkpoint_manager
from src.core.signal_handler import get_signal_handler

# Update the main() function to use enhanced orchestrator
def main():
    """Main pipeline execution with process control support."""
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
    
    # Register progress callbacks if requested
    if args.progress_report:
        progress_manager = get_progress_manager()
        progress_manager.register_callback('any', console_progress_callback)
    
    try:
        # Create enhanced orchestrator (using the new pipeline_orchestrator.py)
        from src.pipelines.unified_resampling.pipeline_orchestrator import UnifiedResamplingPipeline
        pipeline = UnifiedResamplingPipeline(config, db)
        
        # Generate experiment name
        experiment_name = args.experiment_name
        if not experiment_name:
            experiment_name = f"UnifiedResampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine resume checkpoint
        resume_checkpoint = None
        if args.resume:
            # Find latest checkpoint for this experiment
            checkpoints = get_checkpoint_manager().list_checkpoints(
                processor_name='UnifiedResamplingPipeline',
                level='pipeline',
                status='valid'
            )
            if checkpoints:
                resume_checkpoint = checkpoints[0]['checkpoint_id']
                logger.info(f"Found checkpoint to resume from: {resume_checkpoint}")
        
        # Determine phases to run
        phases_to_run = None
        if args.skip_stages:
            all_phases = ['resampling', 'merging', 'analysis', 'export']
            phases_to_run = [p for p in all_phases if p not in args.skip_stages]
        
        # Run pipeline with enhanced features
        logger.info(f"Starting experiment: {experiment_name}")
        
        results = pipeline.run_complete_pipeline(
            experiment_name=experiment_name,
            description="Production unified resampling with process control",
            skip_existing=True,
            resume_from_checkpoint=resume_checkpoint,
            phases_to_run=phases_to_run
        )
        
        # Print results summary
        logger.info("\n" + "="*50)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Experiment ID: {results.get('experiment_id')}")
        logger.info(f"Pipeline ID: {results.get('pipeline_id')}")
        
        if 'resampled_datasets' in results:
            logger.info(f"Datasets processed: {len(results['resampled_datasets'])}")
        
        if 'som_analysis' in results:
            logger.info(f"SOM results saved to: {results['som_analysis']['saved_path']}")
        
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Ensure cleanup
        logger.info("Pipeline execution finished")


# Add argument for checkpoint resume
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
                        help='Resume from latest checkpoint if available')
    parser.add_argument('--skip-stages', nargs='*', default=[],
                        choices=['resampling', 'merging', 'analysis', 'export'],
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
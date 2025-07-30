"""Example script showing ML stage integration in the pipeline."""

import logging
from pathlib import Path
from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineOrchestrator
from src.pipelines.stages.load_stage import DataLoadStage
from src.pipelines.stages.resample_stage import ResampleStage
from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.export_stage import ExportStage
from src.pipelines.stages.ml_stage import MLStage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run pipeline with ML stage."""
    logger.info("Starting pipeline with ML stage example")
    
    # Initialize database connection
    db = DatabaseManager(config.config)
    
    # Create pipeline orchestrator
    orchestrator = PipelineOrchestrator(config.config, db)
    
    # First run the data processing pipeline
    logger.info("Running data processing pipeline...")
    
    # Register data processing stages
    data_stages = [
        DataLoadStage(),
        ResampleStage(),
        MergeStage(),
        ExportStage()
    ]
    
    for stage in data_stages:
        orchestrator.register_stage(stage)
    
    # Run data pipeline
    success = orchestrator.run(
        experiment_name="biodiversity_data_processing",
        resume_from_checkpoint=False
    )
    
    if not success:
        logger.error("Data pipeline failed")
        return
    
    # Get the exported parquet path
    export_results = orchestrator.context.stage_outputs.get('export', {})
    parquet_path = export_results.get('data', {}).get('parquet_path')
    
    if not parquet_path:
        logger.error("No parquet file exported")
        return
    
    logger.info(f"Data exported to: {parquet_path}")
    
    # Now run ML as a separate, standalone pipeline
    logger.info("\nRunning ML pipeline...")
    
    # Create new orchestrator for ML
    ml_orchestrator = PipelineOrchestrator(config.config, db)
    
    # Configure ML stage with the exported parquet
    ml_config = {
        'input_parquet': parquet_path,  # Explicitly specify input
        'model_type': 'lightgbm',  # or 'linear_regression'
        'target_column': 'total_richness',
        'cv_strategy': {
            'type': 'spatial_block',
            'n_splits': 5,
            'block_size': 100
        },
        'imputation_strategy': {
            'type': 'spatial_knn',
            'n_neighbors': 10,
            'spatial_weight': 0.6
        },
        'perform_cv': True,
        'save_model': True,
        'save_predictions': True
    }
    
    # Register only ML stage
    ml_stage = MLStage(ml_config=ml_config)
    ml_orchestrator.register_stage(ml_stage)
    
    # Validate ML pipeline
    is_valid, errors = ml_orchestrator.validate_pipeline()
    if not is_valid:
        logger.error(f"ML pipeline validation failed: {errors}")
        return
    
    # Run ML pipeline
    try:
        ml_success = ml_orchestrator.run(
            experiment_name="biodiversity_ml_analysis",
            resume_from_checkpoint=False
        )
        
        if ml_success:
            logger.info("ML pipeline completed successfully!")
            
            # Access ML results
            ml_outputs = ml_orchestrator.context.stage_outputs.get('machine_learning', {})
            ml_data = ml_outputs.get('data', {})
            ml_metrics = ml_outputs.get('metrics', {})
            
            if ml_data:
                logger.info(f"Model saved to: {ml_data.get('model_path')}")
                logger.info(f"Predictions saved to: {ml_data.get('predictions_path')}")
                logger.info(f"Feature importance saved to: {ml_data.get('importance_path')}")
                logger.info(f"Metrics saved to: {ml_data.get('metrics_path')}")
            
            if ml_metrics:
                logger.info(f"\nML Performance:")
                logger.info(f"  Training R²: {ml_metrics.get('train_r2', 0):.3f}")
                if 'cv_r2_mean' in ml_metrics:
                    logger.info(f"  CV R² (mean ± std): {ml_metrics['cv_r2_mean']:.3f} ± {ml_metrics.get('cv_r2_std', 0):.3f}")
        else:
            logger.error("ML pipeline failed")
            
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
    finally:
        # Cleanup both orchestrators
        orchestrator.cleanup()
        ml_orchestrator.cleanup()
        db.close()


if __name__ == "__main__":
    main()
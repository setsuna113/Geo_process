"""Example of running ML stage standalone on any parquet file."""

import logging
from pathlib import Path
import argparse
from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineOrchestrator
from src.pipelines.stages.ml_stage import MLStage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ml_on_parquet(parquet_path: str, model_type: str = 'linear_regression', 
                      target_column: str = 'total_richness'):
    """Run ML analysis on a parquet file."""
    logger.info(f"Running ML analysis on: {parquet_path}")
    
    # Initialize database connection (needed for orchestrator)
    db = DatabaseManager(config.config)
    
    # Create ML orchestrator
    orchestrator = PipelineOrchestrator(config.config, db)
    
    # Configure ML stage
    ml_config = {
        'input_parquet': parquet_path,
        'model_type': model_type,
        'target_column': target_column,
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
    
    # Create and register ML stage
    ml_stage = MLStage(ml_config=ml_config)
    orchestrator.register_stage(ml_stage)
    
    # Validate
    is_valid, errors = orchestrator.validate_pipeline()
    if not is_valid:
        logger.error(f"Validation failed: {errors}")
        return False
    
    # Run ML pipeline
    try:
        success = orchestrator.run(
            experiment_name=f"ml_analysis_{Path(parquet_path).stem}",
            resume_from_checkpoint=False
        )
        
        if success:
            logger.info("ML analysis completed successfully!")
            
            # Get results
            ml_outputs = orchestrator.context.stage_outputs.get('machine_learning', {})
            ml_data = ml_outputs.get('data', {})
            ml_metrics = ml_outputs.get('metrics', {})
            
            # Print results
            logger.info("\n=== ML Results ===")
            if ml_data:
                for key, path in ml_data.items():
                    if path:
                        logger.info(f"{key}: {path}")
            
            if ml_metrics:
                logger.info("\n=== Performance Metrics ===")
                logger.info(f"Model: {ml_metrics.get('model_type')}")
                logger.info(f"Features: {ml_metrics.get('features_created')}")
                logger.info(f"Samples: {ml_metrics.get('records_processed')}")
                
                # Training metrics
                logger.info(f"\nTraining Performance:")
                logger.info(f"  R²: {ml_metrics.get('train_r2', 0):.3f}")
                logger.info(f"  RMSE: {ml_metrics.get('train_rmse', 0):.2f}")
                
                # CV metrics
                if 'cv_r2_mean' in ml_metrics:
                    logger.info(f"\nCross-Validation Performance:")
                    logger.info(f"  R² (mean ± std): {ml_metrics['cv_r2_mean']:.3f} ± {ml_metrics.get('cv_r2_std', 0):.3f}")
                    logger.info(f"  RMSE (mean ± std): {ml_metrics.get('cv_rmse_mean', 0):.2f} ± {ml_metrics.get('cv_rmse_std', 0):.2f}")
            
            return True
        else:
            logger.error("ML analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during ML analysis: {e}", exc_info=True)
        return False
    finally:
        orchestrator.cleanup()
        db.close()


def main():
    parser = argparse.ArgumentParser(description='Run ML analysis on biodiversity parquet file')
    parser.add_argument('parquet_path', help='Path to input parquet file')
    parser.add_argument('--model', choices=['linear_regression', 'lightgbm'], 
                        default='linear_regression', help='ML model type')
    parser.add_argument('--target', default='total_richness', 
                        help='Target column name for prediction')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.parquet_path).exists():
        logger.error(f"File not found: {args.parquet_path}")
        return
    
    # Run ML analysis
    success = run_ml_on_parquet(
        args.parquet_path,
        model_type=args.model,
        target_column=args.target
    )
    
    if success:
        logger.info("\nML analysis complete!")
    else:
        logger.error("\nML analysis failed!")


if __name__ == "__main__":
    main()
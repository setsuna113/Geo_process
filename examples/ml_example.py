"""Example script demonstrating ML module usage for biodiversity analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import ML components
from src.machine_learning import (
    LinearRegressionAnalyzer,
    RichnessFeatureBuilder,
    SpatialFeatureBuilder,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV
)
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples=1000):
    """Create sample biodiversity data for demonstration."""
    np.random.seed(42)
    
    # Generate spatial coordinates
    lat = np.random.uniform(-60, 60, n_samples)
    lon = np.random.uniform(-180, 180, n_samples)
    
    # Generate richness data with spatial pattern
    # Higher richness near equator
    equator_effect = 1 - np.abs(lat) / 60
    plants_richness = np.random.poisson(50 * equator_effect + 10)
    terrestrial_richness = np.random.poisson(30 * equator_effect + 5)
    
    # Add some missing values
    missing_mask = np.random.random(n_samples) < 0.1
    plants_richness[missing_mask] = np.nan
    
    # Create DataFrame
    data = pd.DataFrame({
        'latitude': lat,
        'longitude': lon,
        'plants_richness': plants_richness,
        'terrestrial_richness': terrestrial_richness
    })
    
    return data


def main():
    """Run ML pipeline example."""
    logger.info("ML Module Example - Biodiversity Richness Prediction")
    
    # 1. Create or load data
    logger.info("\n1. Creating sample data...")
    data = create_sample_data(n_samples=2000)
    logger.info(f"Created {len(data)} samples with columns: {list(data.columns)}")
    logger.info(f"Missing values: {data.isna().sum().to_dict()}")
    
    # 2. Handle missing values with spatial imputation
    logger.info("\n2. Imputing missing values...")
    imputer = SpatialKNNImputer(n_neighbors=5, spatial_weight=0.7)
    data_imputed = imputer.fit_transform(data)
    logger.info(f"After imputation: {data_imputed.isna().sum().to_dict()}")
    
    # 3. Feature engineering
    logger.info("\n3. Engineering features...")
    
    # Option A: Use individual feature builders
    richness_builder = RichnessFeatureBuilder(config.config)
    spatial_builder = SpatialFeatureBuilder(config.config)
    
    richness_features = richness_builder.fit_transform(data_imputed)
    spatial_features = spatial_builder.fit_transform(data_imputed)
    
    # Combine features
    features = pd.concat([richness_features, spatial_features], axis=1)
    
    # Option B: Use composite builder (automatic)
    composite_builder = CompositeFeatureBuilder(config.config)
    features_auto = composite_builder.fit_transform(data_imputed)
    
    logger.info(f"Created {len(features.columns)} features")
    logger.info(f"Feature names: {features.columns.tolist()[:10]}...")  # Show first 10
    
    # 4. Prepare target variable (predict total richness from other features)
    target = features['total_richness'].copy()
    feature_cols = [col for col in features.columns if col not in ['total_richness', 'plants_richness', 'terrestrial_richness']]
    X = features[feature_cols]
    
    # 5. Set up spatial cross-validation
    logger.info("\n4. Setting up spatial cross-validation...")
    cv = SpatialBlockCV(
        n_splits=5,
        block_size=100,  # 100km blocks
        random_state=42
    )
    
    # 6. Train and evaluate model
    logger.info("\n5. Training ML model...")
    model = LinearRegressionAnalyzer(config.config)
    
    # Perform cross-validation
    cv_results = model.cross_validate(
        X, target, 
        cv_strategy=cv,
        metrics=['r2', 'rmse', 'mae']
    )
    
    logger.info(f"\nCross-validation results:")
    logger.info(f"Mean R²: {cv_results.mean_metrics['r2']:.3f} ± {cv_results.std_metrics['r2']:.3f}")
    logger.info(f"Mean RMSE: {cv_results.mean_metrics['rmse']:.2f} ± {cv_results.std_metrics['rmse']:.2f}")
    logger.info(f"Mean MAE: {cv_results.mean_metrics['mae']:.2f} ± {cv_results.std_metrics['mae']:.2f}")
    
    # 7. Fit final model on all data
    logger.info("\n6. Training final model on all data...")
    final_result = model.analyze(
        pd.concat([X, target.rename('target')], axis=1),
        target_column='target',
        feature_columns=feature_cols
    )
    
    # 8. Feature importance
    logger.info("\n7. Feature importance:")
    importance = model.get_feature_importance()
    if importance:
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in top_features:
            logger.info(f"  {feature}: {score:.4f}")
    
    # 9. Save model
    output_dir = Path("outputs/ml_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "biodiversity_model.pkl"
    model.save_model(model_path)
    logger.info(f"\n8. Model saved to: {model_path}")
    
    # 10. Make predictions on new data
    logger.info("\n9. Making predictions on new data...")
    new_data = create_sample_data(n_samples=100)
    new_data_imputed = imputer.transform(new_data)
    new_features = composite_builder.transform(new_data_imputed)
    X_new = new_features[feature_cols]
    
    predictions = model.predict(X_new)
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Save predictions
    results_df = new_data.copy()
    results_df['predicted_total_richness'] = predictions
    results_df['actual_total_richness'] = new_data_imputed['plants_richness'] + new_data_imputed['terrestrial_richness']
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info(f"Predictions saved to: {output_dir / 'predictions.csv'}")
    
    logger.info("\nExample complete!")


if __name__ == "__main__":
    main()
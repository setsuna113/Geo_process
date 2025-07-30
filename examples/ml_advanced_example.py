"""Advanced ML example with LightGBM and comprehensive spatial validation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Import ML components
from src.machine_learning import (
    LinearRegressionAnalyzer,
    LightGBMAnalyzer,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV,
    SpatialBufferCV,
    EnvironmentalBlockCV
)
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_real_data(parquet_path: str = None):
    """
    Load real biodiversity data from parquet file.
    
    If no path provided, creates synthetic data.
    """
    if parquet_path and Path(parquet_path).exists():
        logger.info(f"Loading data from {parquet_path}")
        data = pd.read_parquet(parquet_path)
    else:
        logger.info("Creating synthetic data for demonstration")
        data = create_realistic_synthetic_data(n_samples=5000)
    
    return data


def create_realistic_synthetic_data(n_samples=5000):
    """Create more realistic synthetic biodiversity data."""
    np.random.seed(42)
    
    # Generate spatial coordinates with clustering
    n_clusters = 20
    cluster_centers = np.random.uniform(-60, 60, (n_clusters, 2))
    cluster_centers[:, 1] = np.random.uniform(-180, 180, n_clusters)
    
    lat = []
    lon = []
    cluster_ids = []
    
    for i in range(n_samples):
        cluster = np.random.randint(0, n_clusters)
        cluster_ids.append(cluster)
        lat.append(cluster_centers[cluster, 0] + np.random.normal(0, 5))
        lon.append(cluster_centers[cluster, 1] + np.random.normal(0, 5))
    
    lat = np.array(lat)
    lon = np.array(lon)
    
    # Generate richness with multiple factors
    # 1. Latitude effect (higher near equator)
    equator_effect = 1 - (np.abs(lat) / 60) ** 1.5
    
    # 2. Longitude effect (continental patterns)
    continental_effect = 0.5 + 0.5 * np.sin(np.radians(lon) / 2)
    
    # 3. Cluster effect (local patterns)
    cluster_effects = np.random.uniform(0.5, 1.5, n_clusters)
    local_effect = np.array([cluster_effects[cid] for cid in cluster_ids])
    
    # Combine effects
    base_richness = 50 * equator_effect * continental_effect * local_effect
    
    # Add environmental noise
    environmental_noise = np.random.normal(1, 0.2, n_samples)
    
    # Generate species richness
    plants_richness = np.random.poisson(base_richness * environmental_noise)
    terrestrial_richness = np.random.poisson(0.6 * base_richness * environmental_noise)
    
    # Add missing values with spatial pattern
    missing_prob = 0.1 + 0.1 * (np.abs(lat) / 60)  # More missing at poles
    missing_mask = np.random.random(n_samples) < missing_prob
    plants_richness = plants_richness.astype(float)
    plants_richness[missing_mask] = np.nan
    
    # Create DataFrame
    data = pd.DataFrame({
        'latitude': lat,
        'longitude': lon,
        'plants_richness': plants_richness,
        'terrestrial_richness': terrestrial_richness,
        'cluster_id': cluster_ids
    })
    
    return data


def compare_cv_strategies(X, y, coordinates):
    """Compare different spatial CV strategies."""
    cv_strategies = {
        'Spatial Blocks (100km)': SpatialBlockCV(n_splits=5, block_size=100, random_state=42),
        'Spatial Blocks (200km)': SpatialBlockCV(n_splits=5, block_size=200, random_state=42),
        'Spatial Buffer (50km)': SpatialBufferCV(n_splits=5, buffer_distance=50, random_state=42),
        'Environmental (Latitude)': EnvironmentalBlockCV(n_splits=5, stratify_by='latitude')
    }
    
    results = {}
    
    for cv_name, cv_strategy in cv_strategies.items():
        logger.info(f"\nEvaluating {cv_name}...")
        
        # Simple model for quick comparison
        model = LinearRegressionAnalyzer(config.config)
        
        # Evaluate
        cv_results = model.cross_validate(
            X, y, 
            cv_strategy=cv_strategy,
            metrics=['r2', 'rmse']
        )
        
        results[cv_name] = {
            'mean_r2': cv_results.mean_metrics['r2'],
            'std_r2': cv_results.std_metrics['r2'],
            'mean_rmse': cv_results.mean_metrics['rmse'],
            'std_rmse': cv_results.std_metrics['rmse']
        }
    
    return results


def compare_models(X, y, cv_strategy):
    """Compare different ML models."""
    models = {
        'Ridge Regression': LinearRegressionAnalyzer(config.config),
    }
    
    # Add LightGBM if available
    try:
        models['LightGBM'] = LightGBMAnalyzer(config.config)
    except:
        logger.warning("LightGBM not available")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        cv_results = model.cross_validate(
            X, y,
            cv_strategy=cv_strategy,
            metrics=['r2', 'rmse', 'mae']
        )
        
        results[model_name] = {
            'cv_results': cv_results,
            'model': model
        }
    
    return results


def plot_spatial_cv_folds(data, cv_strategy, output_dir):
    """Visualize spatial CV fold assignments."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Get fold assignments
        coords = data[['latitude', 'longitude']].values
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(data, lat_lon=coords)):
            if fold_idx >= 6:
                break
                
            ax = axes[fold_idx]
            
            # Plot training points
            ax.scatter(
                data.iloc[train_idx]['longitude'],
                data.iloc[train_idx]['latitude'],
                c='blue', alpha=0.5, s=1, label='Train'
            )
            
            # Plot test points
            ax.scatter(
                data.iloc[test_idx]['longitude'],
                data.iloc[test_idx]['latitude'],
                c='red', alpha=0.8, s=2, label='Test'
            )
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Fold {fold_idx + 1}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'spatial_cv_folds.png', dpi=150)
        plt.close()
        
        logger.info(f"Spatial CV visualization saved to {output_dir / 'spatial_cv_folds.png'}")
        
    except ImportError:
        logger.warning("Matplotlib not available for plotting")


def main():
    """Run advanced ML pipeline."""
    output_dir = Path("outputs/ml_advanced_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Advanced ML Example - Biodiversity Prediction with Spatial Validation")
    
    # 1. Load data
    logger.info("\n1. Loading data...")
    data = load_real_data()  # Will create synthetic if no real data
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    logger.info(f"Missing values:\n{data.isna().sum()}")
    
    # 2. Handle missing values
    logger.info("\n2. Imputing missing values...")
    imputer = SpatialKNNImputer(
        n_neighbors=10,
        spatial_weight=0.6,
        missing_values=np.nan
    )
    
    # Fit on non-cluster columns (cluster_id is just for synthetic data)
    impute_cols = [col for col in data.columns if col != 'cluster_id']
    data_imputed = data.copy()
    data_imputed[impute_cols] = imputer.fit_transform(data[impute_cols])
    
    # 3. Feature engineering with composite builder
    logger.info("\n3. Engineering features...")
    feature_builder = CompositeFeatureBuilder(config.config)
    
    # Auto-discover and use all registered feature builders
    features = feature_builder.fit_transform(data_imputed)
    
    logger.info(f"Feature summary:")
    summary = feature_builder.get_feature_summary()
    for category, info in summary['categories'].items():
        logger.info(f"  {category}: {info['count']} features")
    
    # 4. Prepare for modeling
    target = features['richness_total_richness'].copy()  # Composite builder adds prefix
    
    # Remove target-related features
    feature_cols = [
        col for col in features.columns 
        if not any(x in col for x in ['total_richness', 'plants_richness', 'terrestrial_richness'])
    ]
    X = features[feature_cols]
    
    logger.info(f"\nModeling setup:")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Samples: {X.shape[0]}")
    logger.info(f"  Target: {target.name}")
    
    # 5. Compare CV strategies
    logger.info("\n4. Comparing cross-validation strategies...")
    cv_comparison = compare_cv_strategies(X, target, data_imputed[['latitude', 'longitude']])
    
    logger.info("\nCV Strategy Comparison:")
    for strategy, metrics in cv_comparison.items():
        logger.info(f"  {strategy}:")
        logger.info(f"    R²: {metrics['mean_r2']:.3f} ± {metrics['std_r2']:.3f}")
        logger.info(f"    RMSE: {metrics['mean_rmse']:.2f} ± {metrics['std_rmse']:.2f}")
    
    # 6. Visualize spatial CV
    best_cv = SpatialBlockCV(n_splits=5, block_size=100, random_state=42)
    plot_spatial_cv_folds(data_imputed, best_cv, output_dir)
    
    # 7. Compare models
    logger.info("\n5. Comparing ML models...")
    model_comparison = compare_models(X, target, best_cv)
    
    logger.info("\nModel Comparison:")
    for model_name, results in model_comparison.items():
        cv_results = results['cv_results']
        logger.info(f"  {model_name}:")
        logger.info(f"    R²: {cv_results.mean_metrics['r2']:.3f} ± {cv_results.std_metrics['r2']:.3f}")
        logger.info(f"    RMSE: {cv_results.mean_metrics['rmse']:.2f} ± {cv_results.std_metrics['rmse']:.2f}")
        logger.info(f"    MAE: {cv_results.mean_metrics['mae']:.2f} ± {cv_results.std_metrics['mae']:.2f}")
    
    # 8. Train best model
    best_model_name = max(model_comparison.items(), key=lambda x: x[1]['cv_results'].mean_metrics['r2'])[0]
    logger.info(f"\n6. Training best model: {best_model_name}")
    
    best_model = model_comparison[best_model_name]['model']
    best_model.fit(X, target)
    
    # 9. Feature importance analysis
    logger.info("\n7. Analyzing feature importance...")
    importance = best_model.get_feature_importance()
    
    if importance:
        # Save feature importance
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        logger.info("Top 15 important features:")
        for _, row in importance_df.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 10. Model diagnostics
    logger.info("\n8. Model diagnostics...")
    predictions = best_model.predict(X)
    residuals = target - predictions
    
    diagnostics = {
        'residual_mean': residuals.mean(),
        'residual_std': residuals.std(),
        'residual_skew': residuals.skew(),
        'r2_training': best_model.evaluate(X, target, metrics=['r2'])['r2']
    }
    
    logger.info(f"  Residual mean: {diagnostics['residual_mean']:.4f}")
    logger.info(f"  Residual std: {diagnostics['residual_std']:.2f}")
    logger.info(f"  Residual skewness: {diagnostics['residual_skew']:.2f}")
    logger.info(f"  Training R²: {diagnostics['r2_training']:.3f}")
    
    # 11. Save model and results
    model_path = output_dir / f'{best_model_name.lower().replace(" ", "_")}_model.pkl'
    best_model.save_model(model_path)
    logger.info(f"\n9. Model saved to: {model_path}")
    
    # Save predictions
    results_df = data_imputed.copy()
    results_df['predicted_richness'] = predictions
    results_df['residual'] = residuals
    results_df.to_parquet(output_dir / 'predictions.parquet')
    
    logger.info(f"Results saved to: {output_dir / 'predictions.parquet'}")
    logger.info("\nAdvanced example complete!")


if __name__ == "__main__":
    main()
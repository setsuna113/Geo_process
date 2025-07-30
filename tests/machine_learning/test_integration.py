"""Integration tests for the ML module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.machine_learning import (
    LinearRegressionAnalyzer,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV
)
from src.pipelines.stages.ml_stage import MLStage
from src.config import config


class TestMLIntegration:
    """Test complete ML workflows."""
    
    @pytest.fixture
    def sample_biodiversity_data(self):
        """Create realistic biodiversity data."""
        np.random.seed(42)
        n_samples = 500
        
        # Spatial coordinates
        lat = np.random.uniform(-45, 45, n_samples)
        lon = np.random.uniform(-120, 120, n_samples)
        
        # Biodiversity data with spatial pattern
        equator_effect = 1 - (np.abs(lat) / 45) ** 1.5
        
        plants = np.random.poisson(60 * equator_effect + 10)
        animals = np.random.poisson(40 * equator_effect + 5)
        birds = np.random.poisson(25 * equator_effect + 3)
        
        # Add some missing values
        plants[np.random.random(n_samples) < 0.1] = np.nan
        
        data = pd.DataFrame({
            'latitude': lat,
            'longitude': lon,
            'plants_richness': plants,
            'animals_richness': animals,
            'birds_richness': birds
        })
        
        return data
    
    def test_complete_workflow(self, sample_biodiversity_data):
        """Test complete ML workflow from raw data to predictions."""
        # 1. Imputation
        imputer = SpatialKNNImputer(n_neighbors=5, spatial_weight=0.7)
        data_clean = imputer.fit_transform(sample_biodiversity_data)
        
        assert data_clean.isna().sum().sum() == 0
        
        # 2. Feature engineering
        feature_builder = CompositeFeatureBuilder(config)
        features = feature_builder.fit_transform(data_clean)
        
        assert len(features.columns) > len(data_clean.columns)
        assert 'richness_total_richness' in features.columns
        assert 'spatial_distance_to_equator' in features.columns
        
        # 3. Model training with CV
        target = features['richness_total_richness']
        feature_cols = [
            col for col in features.columns 
            if 'richness' not in col.lower()
        ]
        X = features[feature_cols]
        
        model = LinearRegressionAnalyzer(config)
        cv = SpatialBlockCV(n_splits=3, block_size=100)
        
        cv_results = model.cross_validate(X, target, cv_strategy=cv)
        
        assert hasattr(cv_results, 'mean_metrics')
        assert cv_results.mean_metrics['r2'] > 0  # Should have some predictive power
        
        # 4. Final model training
        model.fit(X, target)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert not np.any(np.isnan(predictions))
        
        # 5. Feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(feature_cols)
        
        # Spatial features should be important
        spatial_importance = sum(
            score for feat, score in importance.items() 
            if 'spatial' in feat
        )
        assert spatial_importance > 0
    
    def test_model_persistence(self, sample_biodiversity_data, tmp_path):
        """Test saving and loading complete ML pipeline."""
        # Prepare data
        imputer = SpatialKNNImputer(n_neighbors=5)
        data_clean = imputer.fit_transform(sample_biodiversity_data)
        
        feature_builder = CompositeFeatureBuilder(config)
        features = feature_builder.fit_transform(data_clean)
        
        target = features['richness_total_richness']
        X = features.drop(columns=[col for col in features.columns if 'richness' in col])
        
        # Train model
        model = LinearRegressionAnalyzer(config)
        model.fit(X, target)
        original_predictions = model.predict(X)
        
        # Save all components
        model_path = tmp_path / "model.pkl"
        imputer_path = tmp_path / "imputer.pkl"
        builder_path = tmp_path / "feature_builder.pkl"
        
        model.save_model(model_path)
        
        import pickle
        with open(imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        with open(builder_path, 'wb') as f:
            pickle.dump(feature_builder, f)
        
        # Load and test
        new_model = LinearRegressionAnalyzer(config)
        new_model.load_model(model_path)
        
        with open(imputer_path, 'rb') as f:
            loaded_imputer = pickle.load(f)
        with open(builder_path, 'rb') as f:
            loaded_builder = pickle.load(f)
        
        # Process new data through loaded pipeline
        new_data = sample_biodiversity_data.iloc[:100].copy()
        new_data.loc[5, 'plants_richness'] = np.nan  # Add missing value
        
        new_clean = loaded_imputer.transform(new_data)
        new_features = loaded_builder.transform(new_clean)
        new_X = new_features[X.columns]
        
        new_predictions = new_model.predict(new_X)
        
        assert len(new_predictions) == len(new_data)
        assert not np.any(np.isnan(new_predictions))
    
    def test_ml_stage_integration(self, sample_biodiversity_data, tmp_path):
        """Test MLStage integration."""
        # Save sample data
        data_path = tmp_path / "test_data.parquet"
        sample_biodiversity_data.to_parquet(data_path)
        
        # Create mock context
        from types import SimpleNamespace
        context = SimpleNamespace(
            experiment_id='test_experiment',
            output_dir=tmp_path,
            config={'grid': {'grid_id': 'test_grid'}},
            stage_outputs={}  # No dependencies on other stages
        )
        
        # Create and execute ML stage
        ml_config = {
            'input_parquet': str(data_path),  # Explicitly specify input
            'model_type': 'linear_regression',
            'target_column': 'total_richness',
            'cv_strategy': {'type': 'spatial_block', 'n_splits': 3},
            'perform_cv': True,
            'save_model': True,
            'save_predictions': True
        }
        
        ml_stage = MLStage(ml_config=ml_config)
        
        # Validate stage
        is_valid, errors = ml_stage.validate()
        assert is_valid, f"Validation errors: {errors}"
        
        # Execute stage
        result = ml_stage.execute(context)
        
        assert result.success
        assert 'model_path' in result.data
        assert 'predictions_path' in result.data
        
        # Check outputs exist
        assert Path(result.data['model_path']).exists()
        assert Path(result.data['predictions_path']).exists()
        
        # Check metrics
        assert 'cv_r2_mean' in result.metrics
        assert result.metrics['cv_r2_mean'] > 0
    
    def test_registry_integration(self):
        """Test that ML components are properly registered."""
        from src.core.registry import ComponentRegistry
        
        # Check model registration
        ml_models = ComponentRegistry.get_available_ml_models()
        assert 'linear_regression' in ml_models
        
        # Check feature builder registration
        feature_builders = ComponentRegistry.get_available_feature_builders()
        assert len(feature_builders) > 0
        assert any('richness' in str(fb) for fb in feature_builders)
        
        # Check CV strategy registration
        cv_strategies = ComponentRegistry.get_available_cv_strategies()
        assert len(cv_strategies) > 0
        
        # Check imputation strategy registration
        imputation_strategies = ComponentRegistry.get_available_imputation_strategies()
        assert len(imputation_strategies) > 0
    
    def test_error_handling(self):
        """Test error handling in ML components."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']  # Non-numeric
        })
        
        model = LinearRegressionAnalyzer(config)
        
        # Should handle non-numeric data gracefully
        with pytest.raises((ValueError, TypeError)):
            model.fit(invalid_data[['col2']], pd.Series([1, 2, 3]))
        
        # Test with mismatched dimensions
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1, 2])  # Wrong length
        
        with pytest.raises(ValueError):
            model.fit(X, y)
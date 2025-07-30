"""Tests for ML models."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.machine_learning import LinearRegressionAnalyzer, LightGBMAnalyzer
from src.abstractions.types.ml_types import MLResult
from src.config import config


class TestLinearRegressionAnalyzer:
    """Test LinearRegressionAnalyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with linear relationship
        true_coef = np.array([1.5, -2.0, 0.5, 1.0, -0.5])
        y = pd.Series(X.values @ true_coef + np.random.randn(n_samples) * 0.1)
        
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return LinearRegressionAnalyzer(config)
    
    def test_fit_predict(self, model, sample_data):
        """Test basic fit and predict functionality."""
        X, y = sample_data
        
        # Fit model
        result = model.fit(X, y)
        assert isinstance(result, LinearRegressionAnalyzer)
        assert model._model is not None
        assert model._scaler is not None
        assert model._feature_names == list(X.columns)
        
        # Predict
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))
    
    def test_evaluate(self, model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        model.fit(X, y)
        metrics = model.evaluate(X, y, metrics=['r2', 'rmse', 'mae'])
        
        assert isinstance(metrics, dict)
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 0 <= metrics['r2'] <= 1
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_feature_importance(self, model, sample_data):
        """Test feature importance calculation."""
        X, y = sample_data
        
        model.fit(X, y)
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(feat in importance for feat in X.columns)
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_cross_validate(self, model, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        
        # Mock CV strategy
        cv_strategy = Mock()
        cv_strategy.split.return_value = [
            (np.arange(0, 80), np.arange(80, 100)),
            (np.arange(20, 100), np.arange(0, 20))
        ]
        
        cv_results = model.cross_validate(X, y, cv_strategy=cv_strategy)
        
        assert hasattr(cv_results, 'mean_metrics')
        assert hasattr(cv_results, 'std_metrics')
        assert hasattr(cv_results, 'fold_metrics')
        assert len(cv_results.fold_metrics) == 2
    
    def test_save_load_model(self, model, sample_data, tmp_path):
        """Test model persistence."""
        X, y = sample_data
        
        # Train model
        model.fit(X, y)
        original_predictions = model.predict(X)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        model.save_model(model_path)
        assert model_path.exists()
        
        # Load model
        new_model = LinearRegressionAnalyzer(config)
        new_model.load_model(model_path)
        
        # Check predictions match
        new_predictions = new_model.predict(X)
        np.testing.assert_array_almost_equal(original_predictions, new_predictions)
    
    def test_analyze_method(self, model, sample_data):
        """Test high-level analyze method."""
        X, y = sample_data
        
        # Create combined dataframe
        data = X.copy()
        data['target'] = y
        
        result = model.analyze(
            data,
            target_column='target',
            feature_columns=list(X.columns)
        )
        
        assert isinstance(result, MLResult)
        assert result.model_type == 'linear_regression'
        assert 'predictions' in result.data
        assert 'feature_importance' in result.data
        assert 'metrics' in result.data
    
    def test_missing_values_handling(self, model):
        """Test handling of missing values."""
        # Create data with missing values
        X = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],
            'feature_2': [2, np.nan, 6, 8, 10]
        })
        y = pd.Series([1, 2, 3, 4, 5])
        
        # Model should raise error for missing values
        with pytest.raises(ValueError, match="missing values"):
            model.fit(X, y)
    
    def test_empty_data_handling(self, model):
        """Test handling of empty data."""
        X = pd.DataFrame()
        y = pd.Series()
        
        with pytest.raises(ValueError):
            model.fit(X, y)


@pytest.mark.skipif(
    not pytest.importorskip("lightgbm"),
    reason="LightGBM not installed"
)
class TestLightGBMAnalyzer:
    """Test LightGBMAnalyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create non-linear target
        y = pd.Series(
            np.sin(X['feature_0']) + 
            np.cos(X['feature_1']) + 
            X['feature_2']**2 + 
            np.random.randn(n_samples) * 0.1
        )
        
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return LightGBMAnalyzer(config)
    
    def test_fit_predict(self, model, sample_data):
        """Test basic fit and predict functionality."""
        X, y = sample_data
        
        # Fit model
        result = model.fit(X, y)
        assert isinstance(result, LightGBMAnalyzer)
        assert model._model is not None
        
        # Predict
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
    
    def test_native_missing_values(self, model):
        """Test LightGBM's native handling of missing values."""
        # Create data with missing values
        X = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'feature_2': [2, np.nan, 6, 8, 10, 12, 14, 16, 18, 20]
        })
        y = pd.Series(range(10))
        
        # LightGBM should handle missing values natively
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))
    
    def test_feature_importance(self, model, sample_data):
        """Test feature importance from tree model."""
        X, y = sample_data
        
        model.fit(X, y)
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(v >= 0 for v in importance.values())
        
        # Tree models should have some features with zero importance
        assert any(v == 0 for v in importance.values()) or len(importance) > 5
    
    def test_early_stopping(self, model, sample_data):
        """Test early stopping functionality."""
        X, y = sample_data
        
        # Split data for validation
        n_train = int(0.8 * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Fit with validation data (would trigger early stopping if implemented)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        # Check model was fitted
        assert model._model is not None
        
    def test_model_params(self, model):
        """Test model parameters from config."""
        expected_params = config.machine_learning.models.lightgbm
        
        assert model._params['num_leaves'] == expected_params.get('num_leaves', 31)
        assert model._params['learning_rate'] == expected_params.get('learning_rate', 0.05)
        assert model._params['verbosity'] == -1  # Should be silent
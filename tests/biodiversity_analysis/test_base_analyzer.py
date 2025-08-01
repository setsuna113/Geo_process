"""Integration tests for base biodiversity analyzer."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

from src.biodiversity_analysis import BaseBiodiversityAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult


class MockBiodiversityAnalyzer(BaseBiodiversityAnalyzer):
    """Mock analyzer for testing base functionality."""
    
    def analyze(self, data_path: str, **parameters) -> AnalysisResult:
        """Simple mock analysis."""
        # Load and preprocess data
        data = self.load_data(data_path)
        data = self.preprocess_data(data)
        
        # Mock analysis
        self.update_progress("Running analysis", 0.5)
        
        # Create mock labels
        labels = np.random.randint(0, 3, size=data.n_samples)
        
        # Create result
        result_data = {
            'labels': labels,
            'statistics': {
                'n_clusters': 3,
                'cluster_sizes': np.bincount(labels).tolist()
            }
        }
        
        return self.create_result(
            success=True,
            data=result_data,
            runtime_seconds=1.0,
            n_samples=data.n_samples,
            n_features=data.n_features,
            parameters=parameters
        )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate mock parameters."""
        issues = []
        
        if 'n_clusters' in parameters:
            if parameters['n_clusters'] < 2:
                issues.append("n_clusters must be at least 2")
        
        return len(issues) == 0, issues
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default mock parameters."""
        return {
            'n_clusters': 3,
            'method': 'mock'
        }


class TestBaseBiodiversityAnalyzer:
    """Test base analyzer functionality."""
    
    @pytest.fixture
    def sample_biodiversity_data(self):
        """Create sample biodiversity data."""
        np.random.seed(42)
        n_samples = 100
        
        # Create zero-inflated species data
        data = {
            'longitude': np.random.uniform(-10, 10, n_samples),
            'latitude': np.random.uniform(40, 50, n_samples),
            'species_1': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
            'species_2': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
            'species_3': np.random.binomial(1, 0.3, n_samples),
            'env_temp': np.random.normal(15, 3, n_samples),
            'env_precip': np.random.normal(800, 100, n_samples)
        }
        
        # Add some NaN values
        data['species_1'][::10] = np.nan
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_parquet_file(self, sample_biodiversity_data):
        """Create temporary parquet file."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = Path(f.name)
        
        sample_biodiversity_data.to_parquet(temp_path)
        yield str(temp_path)
        temp_path.unlink()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        assert analyzer.method_name == 'mock'
        assert analyzer.version == '1.0.0'
        assert analyzer.config is not None
        assert analyzer.data_loader is not None
    
    def test_load_data(self, sample_parquet_file):
        """Test data loading."""
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        data = analyzer.load_data(sample_parquet_file)
        
        assert data.n_samples == 100
        assert data.n_features == 5  # All numeric except coordinates
        assert data.coordinates.shape == (100, 2)
        assert data.zero_inflated  # Should detect zero inflation
    
    def test_preprocess_data(self, sample_parquet_file):
        """Test data preprocessing."""
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        # Configure preprocessing
        analyzer.data_config = {
            'missing_value_strategy': 'median',
            'handle_zero_inflation': True,
            'remove_constant_features': True,
            'normalization_method': 'standard'
        }
        
        data = analyzer.load_data(sample_parquet_file)
        original_shape = data.features.shape
        
        # Check for NaN before preprocessing
        assert np.any(np.isnan(data.features))
        
        processed_data = analyzer.preprocess_data(data)
        
        # Check NaN handled
        assert not np.any(np.isnan(processed_data.features))
        
        # Check normalization applied
        assert 'normalization' in processed_data.metadata
        
        # Check zero inflation handled
        if data.zero_inflated:
            assert 'zero_inflation_transform' in processed_data.metadata
    
    def test_analyze_integration(self, sample_parquet_file):
        """Test full analysis integration."""
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        # Track progress updates
        progress_updates = []
        
        def progress_callback(message, progress):
            progress_updates.append((message, progress))
        
        analyzer.progress_callback = progress_callback
        
        # Run analysis
        result = analyzer.analyze(sample_parquet_file, n_clusters=3)
        
        # Check result
        assert isinstance(result, AnalysisResult)
        assert result.labels is not None
        assert len(result.labels) == 100
        assert result.metadata.analysis_type == 'MOCK'
        assert result.statistics['n_clusters'] == 3
        
        # Check progress was tracked
        assert len(progress_updates) > 0
        assert any('Loading data' in msg for msg, _ in progress_updates)
        assert any('Preprocessing' in msg for msg, _ in progress_updates)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        # Valid parameters
        valid, issues = analyzer.validate_parameters({'n_clusters': 3})
        assert valid
        assert len(issues) == 0
        
        # Invalid parameters
        valid, issues = analyzer.validate_parameters({'n_clusters': 1})
        assert not valid
        assert 'n_clusters must be at least 2' in issues
    
    def test_config_integration(self):
        """Test biodiversity config integration."""
        from src.config import get_biodiversity_config
        
        # Register custom config
        config_manager = get_biodiversity_config()
        config_manager.register_custom_method('mock', {
            'n_clusters': 5,
            'custom_param': 'test'
        })
        
        # Create analyzer
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        # Should have merged config
        assert analyzer.method_params['n_clusters'] == 5
        assert analyzer.method_params['custom_param'] == 'test'
    
    def test_error_handling(self):
        """Test error handling."""
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        # Non-existent file
        with pytest.raises(ValueError, match="File not found"):
            analyzer.load_data("nonexistent.parquet")
    
    def test_runtime_measurement(self):
        """Test runtime measurement utility."""
        import time
        analyzer = MockBiodiversityAnalyzer(method_name='mock')
        
        def slow_function():
            time.sleep(0.1)
            return "result"
        
        result, runtime = analyzer.measure_runtime(slow_function)
        
        assert result == "result"
        assert 0.09 < runtime < 0.15  # Allow some variance
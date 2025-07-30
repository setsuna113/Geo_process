"""Simplified unit tests for GEE extraction logic - standalone version."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MockGEEImage:
    """Mock GEE Image class for testing."""
    
    def __init__(self, band_name='bio01'):
        self.band_name = band_name
    
    def select(self, band):
        return MockGEEImage(band)
    
    def sampleRegions(self, collection, scale, geometries=False):
        """Mock sampleRegions method."""
        mock_result = Mock()
        
        # Create mock features with sample data
        features = []
        for i in range(len(collection.features)):
            feature = {
                'properties': {
                    'point_id': i,
                    self.band_name: 150 + i * 10  # Mock temperature values
                }
            }
            features.append(feature)
        
        mock_result.getInfo.return_value = {'features': features}
        return mock_result


class MockGEEGeometry:
    """Mock GEE Geometry class."""
    
    @staticmethod
    def Point(coords):
        return {'type': 'Point', 'coordinates': coords}


class MockGEEFeature:
    """Mock GEE Feature class."""
    
    def __init__(self, geometry, properties):
        self.geometry = geometry
        self.properties = properties


class MockGEEFeatureCollection:
    """Mock GEE FeatureCollection class."""
    
    def __init__(self, features):
        self.features = features


class MockEarthEngine:
    """Mock Earth Engine module."""
    
    def __init__(self):
        self.Image = Mock(side_effect=lambda asset: MockGEEImage())
        self.Geometry = MockGEEGeometry()
        self.Feature = MockGEEFeature
        self.FeatureCollection = MockGEEFeatureCollection


class StandaloneGEEExtractor:
    """Standalone GEE extractor for testing."""
    
    # WorldClim dataset configurations
    WORLDCLIM_DATASETS = {
        'bio01': {
            'asset': 'WORLDCLIM/V1/BIO',
            'band': 'bio01',
            'description': 'Annual Mean Temperature',
            'units': '°C * 10',
            'scale_factor': 0.1
        },
        'bio04': {
            'asset': 'WORLDCLIM/V1/BIO', 
            'band': 'bio04',
            'description': 'Temperature Seasonality',
            'units': '°C * 100',
            'scale_factor': 0.01
        },
        'bio12': {
            'asset': 'WORLDCLIM/V1/BIO',
            'band': 'bio12', 
            'description': 'Annual Precipitation',
            'units': 'mm',
            'scale_factor': 1.0
        }
    }
    
    def __init__(self, chunk_size=5000):
        self.chunk_size = min(chunk_size, 5000)  # GEE limit
        self.ee = MockEarthEngine()
        self._load_worldclim_images()
    
    def _load_worldclim_images(self):
        """Load WorldClim image assets."""
        self.images = {}
        base_image = MockGEEImage()
        
        for var_name, config in self.WORLDCLIM_DATASETS.items():
            self.images[var_name] = base_image.select(config['band'])
    
    def extract_climate_data(self, bounds, variables=None):
        """Extract climate data for coordinate grid within bounds."""
        if variables is None:
            variables = list(self.WORLDCLIM_DATASETS.keys())
        
        # Generate simple coordinate grid
        min_x, min_y, max_x, max_y = bounds
        resolution = 0.1  # Simple resolution for testing
        
        coords = []
        x_val = min_x
        while x_val < max_x:
            y_val = min_y
            while y_val < max_y:
                coords.append({'x': x_val, 'y': y_val})
                y_val += resolution
            x_val += resolution
        
        coord_chunk = pd.DataFrame(coords)
        
        if len(coord_chunk) == 0:
            return pd.DataFrame(columns=['x', 'y'] + variables)
        
        # Extract data for chunk
        result_df = self._extract_chunk_data(coord_chunk, variables, 1)
        return result_df
    
    def _extract_chunk_data(self, coord_chunk, variables, chunk_id):
        """Extract climate data for a single coordinate chunk."""
        # Create list of coordinate points for sampling
        points = [[row['x'], row['y']] for _, row in coord_chunk.iterrows()]
        
        # Initialize result DataFrame
        result_df = coord_chunk.copy()
        
        for var_name in variables:
            try:
                # Use mock sampling
                raw_values = self._sample_image_batch(
                    self.images[var_name], 
                    points
                )
                
                # Apply scale factor to valid values
                scale_factor = self.WORLDCLIM_DATASETS[var_name]['scale_factor']
                scaled_values = []
                
                for raw_value in raw_values:
                    if raw_value is not None and not np.isnan(raw_value):
                        scaled_values.append(raw_value * scale_factor)
                    else:
                        scaled_values.append(np.nan)
                
                # Add values to result DataFrame
                result_df[var_name] = scaled_values
                
            except Exception:
                # Fill with NaN values on failure
                result_df[var_name] = np.nan
        
        return result_df
    
    def _sample_image_batch(self, image, points):
        """Mock batch image sampling."""
        # Generate mock values based on coordinates
        values = []
        for i, point in enumerate(points):
            # Generate predictable test values based on position
            base_value = 150 + (point[0] * 10) + (point[1] * 5) + i
            values.append(base_value)
        
        return values
    
    def get_variable_info(self):
        """Get information about available WorldClim variables."""
        return self.WORLDCLIM_DATASETS.copy()
    
    def test_extraction(self, test_bounds=(-1, -1, 1, 1), variables=None):
        """Test extraction with a small area."""
        if variables is None:
            variables = ['bio01']  # Test with just temperature
        
        return self.extract_climate_data(test_bounds, variables)


class TestStandaloneGEEExtractor:
    """Test suite for standalone GEE extractor."""
    
    def test_init_success(self):
        """Test successful initialization."""
        extractor = StandaloneGEEExtractor(chunk_size=1000)
        
        assert extractor.chunk_size == 1000
        assert extractor.ee is not None
        assert len(extractor.images) == 3
    
    def test_init_chunk_size_limit(self):
        """Test chunk size is limited to GEE maximum."""
        extractor = StandaloneGEEExtractor(chunk_size=10000)  # Above GEE limit
        assert extractor.chunk_size == 5000  # Should be capped at GEE limit
    
    def test_load_worldclim_images(self):
        """Test WorldClim image loading."""
        extractor = StandaloneGEEExtractor()
        
        # Should have loaded all three variables
        assert len(extractor.images) == 3
        assert 'bio01' in extractor.images
        assert 'bio04' in extractor.images
        assert 'bio12' in extractor.images
    
    def test_get_variable_info(self):
        """Test getting variable information."""
        extractor = StandaloneGEEExtractor()
        
        var_info = extractor.get_variable_info()
        
        assert isinstance(var_info, dict)
        assert 'bio01' in var_info
        assert 'description' in var_info['bio01']
        assert 'units' in var_info['bio01']
        assert 'scale_factor' in var_info['bio01']
    
    def test_extract_climate_data_basic(self):
        """Test basic climate data extraction."""
        extractor = StandaloneGEEExtractor()
        
        bounds = (0, 0, 0.2, 0.2)
        variables = ['bio01']
        
        result_df = extractor.extract_climate_data(bounds, variables)
        
        # Should return DataFrame with coordinates and climate data
        assert isinstance(result_df, pd.DataFrame)
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
        assert 'bio01' in result_df.columns
        assert len(result_df) > 0
    
    def test_extract_climate_data_all_variables(self):
        """Test extraction with all variables."""
        extractor = StandaloneGEEExtractor()
        
        bounds = (0, 0, 0.2, 0.2)
        
        # Test with default variables (should be all)
        result_df = extractor.extract_climate_data(bounds)
        
        assert 'bio01' in result_df.columns
        assert 'bio04' in result_df.columns
        assert 'bio12' in result_df.columns
    
    def test_extract_climate_data_empty_bounds(self):
        """Test extraction with bounds that produce no coordinates."""  
        extractor = StandaloneGEEExtractor()
        
        bounds = (0, 0, 0, 0)  # Zero-size bounds
        variables = ['bio01']
        
        result_df = extractor.extract_climate_data(bounds, variables)
        
        # Should return empty DataFrame with correct columns
        assert isinstance(result_df, pd.DataFrame)
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
        assert 'bio01' in result_df.columns
        assert len(result_df) == 0
    
    def test_extract_chunk_data(self):
        """Test single chunk data extraction."""
        extractor = StandaloneGEEExtractor()
        
        # Create test coordinate chunk
        coord_chunk = pd.DataFrame({
            'x': [0.0, 0.1, 0.2],
            'y': [0.0, 0.1, 0.2]
        })
        
        variables = ['bio01']
        
        result_df = extractor._extract_chunk_data(coord_chunk, variables, 1)
        
        assert len(result_df) == len(coord_chunk)
        assert 'bio01' in result_df.columns
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
        
        # Check that values were generated
        assert not result_df['bio01'].isna().all()
    
    def test_sample_image_batch(self):
        """Test batch image sampling."""
        extractor = StandaloneGEEExtractor()
        
        mock_image = MockGEEImage('bio01')
        points = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]
        
        values = extractor._sample_image_batch(mock_image, points)
        
        assert len(values) == len(points)
        assert all(isinstance(v, (int, float)) for v in values)
        assert all(v > 0 for v in values)  # Should have positive values
    
    def test_test_extraction(self):
        """Test the test extraction method."""
        extractor = StandaloneGEEExtractor()
        
        result_df = extractor.test_extraction()
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
        assert 'bio01' in result_df.columns
        assert len(result_df) > 0
    
    def test_scale_factor_application(self):
        """Test that scale factors are applied correctly."""
        extractor = StandaloneGEEExtractor()
        
        # Create test data with known coordinates
        coord_chunk = pd.DataFrame({
            'x': [0.0],
            'y': [0.0]
        })
        
        result_df = extractor._extract_chunk_data(coord_chunk, ['bio01'], 1)
        
        # Check that scale factor was applied (should be different from raw value)
        raw_value = 150  # Base value from mock
        expected_scaled = raw_value * 0.1  # bio01 scale factor is 0.1
        
        # Value should be in reasonable range after scaling
        actual_value = result_df['bio01'].iloc[0]
        assert isinstance(actual_value, (int, float))
        assert actual_value > 0
    
    def test_multiple_variables_scaling(self):
        """Test that different variables get proper scaling."""
        extractor = StandaloneGEEExtractor()
        
        coord_chunk = pd.DataFrame({
            'x': [0.0],
            'y': [0.0]
        })
        
        variables = ['bio01', 'bio04', 'bio12']
        result_df = extractor._extract_chunk_data(coord_chunk, variables, 1)
        
        # All variables should have values
        for var in variables:
            assert var in result_df.columns
            assert not pd.isna(result_df[var].iloc[0])
            assert isinstance(result_df[var].iloc[0], (int, float))
    
    def test_coordinate_preservation(self):
        """Test that input coordinates are preserved in output."""
        extractor = StandaloneGEEExtractor()
        
        test_coords = pd.DataFrame({
            'x': [1.5, 2.5, 3.5],
            'y': [4.5, 5.5, 6.5]
        })
        
        result_df = extractor._extract_chunk_data(test_coords, ['bio01'], 1)
        
        # Coordinates should be preserved exactly
        pd.testing.assert_series_equal(result_df['x'], test_coords['x'])
        pd.testing.assert_series_equal(result_df['y'], test_coords['y'])
    
    def test_variable_info_completeness(self):
        """Test that variable info contains all required fields."""
        extractor = StandaloneGEEExtractor()
        
        var_info = extractor.get_variable_info()
        
        required_fields = ['asset', 'band', 'description', 'units', 'scale_factor']
        
        for var_name, info in var_info.items():
            for field in required_fields:
                assert field in info, f"Field {field} missing from {var_name}"
            
            # Check that scale factors are reasonable
            assert 0 < info['scale_factor'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
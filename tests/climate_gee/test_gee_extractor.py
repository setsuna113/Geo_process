"""Unit tests for GEEClimateExtractor with mocked GEE API."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile

# Mock database connections before importing climate_gee modules
with patch('src.database.connection.DatabaseManager'):
    with patch('psycopg2.pool.ThreadedConnectionPool'):
        from src.climate_gee.gee_extractor import GEEClimateExtractor
        from src.climate_gee.auth import GEEAuthenticator
        from src.climate_gee.coordinate_generator import CoordinateGenerator


class MockGEEImage:
    """Mock GEE Image class for testing."""
    
    def __init__(self, band_name='bio01'):
        self.band_name = band_name
        self._values = {}
    
    def select(self, band):
        mock_img = MockGEEImage(band)
        return mock_img
    
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
    
    def sample(self, geometry, scale):
        """Mock sample method for individual sampling."""
        mock_feature = Mock()
        mock_feature.getInfo.return_value = {
            'properties': {
                self.band_name: 150  # Mock value
            }
        }
        mock_feature.first.return_value = mock_feature
        return mock_feature


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


@pytest.fixture
def mock_authenticated_gee():
    """Create mock authenticated GEE authenticator."""
    auth = Mock(spec=GEEAuthenticator)
    auth.is_authenticated.return_value = True
    auth.ee = MockEarthEngine()
    return auth


@pytest.fixture
def mock_coordinate_generator():
    """Create mock coordinate generator."""
    coord_gen = Mock(spec=CoordinateGenerator)
    
    # Mock coordinate generation
    def mock_generate_chunks(bounds, chunk_size):
        # Generate simple test coordinates
        coords = []
        for x in np.arange(bounds[0], bounds[2], 0.1):
            for y in np.arange(bounds[1], bounds[3], 0.1):
                coords.append({'x': x, 'y': y})
                if len(coords) >= chunk_size:
                    yield pd.DataFrame(coords)
                    coords = []
        if coords:
            yield pd.DataFrame(coords)
    
    coord_gen.generate_coordinate_chunks = mock_generate_chunks
    return coord_gen


class TestGEEClimateExtractor:
    """Test suite for GEEClimateExtractor class."""
    
    def test_init_success(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test successful initialization."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator,
            chunk_size=1000
        )
        
        assert extractor.auth == mock_authenticated_gee
        assert extractor.coord_gen == mock_coordinate_generator
        assert extractor.chunk_size == 1000
        assert extractor.ee is not None
    
    def test_init_unauthenticated(self, mock_coordinate_generator):
        """Test initialization with unauthenticated GEE."""
        auth = Mock(spec=GEEAuthenticator)
        auth.is_authenticated.return_value = False
        
        with pytest.raises(ValueError, match="GEE authenticator must be authenticated"):
            GEEClimateExtractor(auth, mock_coordinate_generator)
    
    def test_init_chunk_size_limit(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test chunk size is limited to GEE maximum."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator,
            chunk_size=10000  # Above GEE limit
        )
        
        assert extractor.chunk_size == 5000  # Should be capped at GEE limit
    
    def test_load_worldclim_images(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test WorldClim image loading."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        # Should have loaded all three variables
        assert len(extractor.images) == 3
        assert 'bio01' in extractor.images
        assert 'bio04' in extractor.images
        assert 'bio12' in extractor.images
    
    def test_get_variable_info(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test getting variable information."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        var_info = extractor.get_variable_info()
        
        assert isinstance(var_info, dict)
        assert 'bio01' in var_info
        assert 'description' in var_info['bio01']
        assert 'units' in var_info['bio01']
        assert 'scale_factor' in var_info['bio01']
    
    def test_extract_climate_data_basic(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test basic climate data extraction."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator,
            chunk_size=10
        )
        
        bounds = (0, 0, 0.2, 0.2)
        variables = ['bio01']
        
        result_df = extractor.extract_climate_data(bounds, variables)
        
        # Should return DataFrame with coordinates and climate data
        assert isinstance(result_df, pd.DataFrame)
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
        assert 'bio01' in result_df.columns
        assert len(result_df) > 0
    
    def test_extract_climate_data_all_variables(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test extraction with all variables."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator,
            chunk_size=5
        )
        
        bounds = (0, 0, 0.1, 0.1)
        
        # Test with default variables (should be all)
        result_df = extractor.extract_climate_data(bounds)
        
        assert 'bio01' in result_df.columns
        assert 'bio04' in result_df.columns
        assert 'bio12' in result_df.columns
    
    def test_extract_climate_data_invalid_variables(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test extraction with invalid variables."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        bounds = (0, 0, 0.1, 0.1)
        invalid_variables = ['bio99', 'invalid_var']
        
        with pytest.raises(ValueError, match="Invalid variables"):
            extractor.extract_climate_data(bounds, invalid_variables)
    
    def test_extract_chunk_data(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test single chunk data extraction."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        # Create test coordinate chunk
        coord_chunk = pd.DataFrame({
            'x': [0.0, 0.1, 0.2],
            'y': [0.0, 0.1, 0.2]
        })
        
        variables = ['bio01']
        
        result_df = extractor._extract_chunk_data(coord_chunk, variables, None, 1)
        
        assert len(result_df) == len(coord_chunk)
        assert 'bio01' in result_df.columns
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
    
    def test_sample_image_batch(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test batch image sampling."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        mock_image = MockGEEImage('bio01')
        points = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]
        
        values = extractor._sample_image_batch(mock_image, points)
        
        assert len(values) == len(points)
        assert all(isinstance(v, (int, float)) or np.isnan(v) for v in values)
    
    def test_sample_image_individual_fallback(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test individual sampling fallback."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        # Create mock image that fails batch sampling
        mock_image = Mock()
        mock_image.sampleRegions.side_effect = Exception("Batch sampling failed")
        mock_image.sample.return_value.first.return_value.getInfo.return_value = {
            'properties': {'bio01': 200}
        }
        
        points = [[0.0, 0.0], [0.1, 0.1]]
        
        values = extractor._sample_image_individual(mock_image, points)
        
        assert len(values) == len(points)
        assert all(isinstance(v, (int, float)) or np.isnan(v) for v in values)
    
    def test_test_extraction(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test the test extraction method."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        result_df = extractor.test_extraction()
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
        assert 'bio01' in result_df.columns
        assert len(result_df) > 0
    
    def test_scale_factor_application(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test that scale factors are applied correctly."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        # Create test data with known values
        coord_chunk = pd.DataFrame({
            'x': [0.0],
            'y': [0.0]
        })
        
        # Mock the sampling to return known raw values
        with patch.object(extractor, '_sample_image_batch') as mock_sample:
            mock_sample.return_value = [1500]  # Raw value for bio01
            
            result_df = extractor._extract_chunk_data(coord_chunk, ['bio01'], None, 1)
            
            # bio01 scale factor is 0.1, so 1500 * 0.1 = 150
            assert result_df['bio01'].iloc[0] == 150.0
    
    def test_nan_value_handling(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test handling of NaN values from GEE."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        coord_chunk = pd.DataFrame({
            'x': [0.0, 0.1],
            'y': [0.0, 0.1]
        })
        
        # Mock sampling to return mix of valid and NaN values
        with patch.object(extractor, '_sample_image_batch') as mock_sample:
            mock_sample.return_value = [150, np.nan]
            
            result_df = extractor._extract_chunk_data(coord_chunk, ['bio01'], None, 1)
            
            assert result_df['bio01'].iloc[0] == 15.0  # 150 * 0.1
            assert np.isnan(result_df['bio01'].iloc[1])
    
    def test_error_handling_chunk_failure(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test error handling when chunk processing fails."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator
        )
        
        bounds = (0, 0, 0.1, 0.1)
        
        # Mock chunk extraction to fail
        with patch.object(extractor, '_extract_chunk_data') as mock_extract:
            mock_extract.side_effect = Exception("Chunk processing failed")
            
            with pytest.raises(RuntimeError, match="No chunks processed successfully"):
                extractor.extract_climate_data(bounds, ['bio01'])
    
    def test_error_handling_partial_failure(self, mock_authenticated_gee, mock_coordinate_generator):
        """Test handling when some chunks fail but others succeed."""
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            mock_coordinate_generator,
            chunk_size=2
        )
        
        bounds = (0, 0, 0.3, 0.1)  # Should create multiple chunks
        
        # Mock first chunk to fail, second to succeed
        with patch.object(extractor, '_extract_chunk_data') as mock_extract:
            def side_effect(chunk, variables, output_dir, chunk_id):
                if chunk_id == 1:
                    raise Exception("First chunk failed")
                else:
                    return chunk.copy()  # Return successful result
            
            mock_extract.side_effect = side_effect
            
            result_df = extractor.extract_climate_data(bounds, ['bio01'])
            
            # Should still return results from successful chunks
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) > 0


@pytest.fixture
def temp_config_file():
    """Create temporary config file for testing."""
    config_data = {
        'resampling': {
            'target_resolution': 0.05
        },
        'processing_bounds': {
            'test_region': [-1, -1, 1, 1]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


class TestGEEExtractorIntegration:
    """Integration tests for GEE extractor with mocked components."""
    
    @patch('src.climate_gee.gee_extractor.GEEAuthenticator')
    @patch('src.climate_gee.gee_extractor.CoordinateGenerator')
    def test_create_gee_extractor_function(self, mock_coord_gen_class, mock_auth_class, temp_config_file):
        """Test the create_gee_extractor convenience function."""
        # Setup mocks
        mock_auth = Mock()
        mock_auth.is_authenticated.return_value = True
        mock_auth.ee = MockEarthEngine()
        
        mock_coord_gen = Mock()
        mock_coord_gen.generate_coordinate_chunks.return_value = [
            pd.DataFrame({'x': [0.0], 'y': [0.0]})
        ]
        
        # Mock the setup functions
        with patch('src.climate_gee.gee_extractor.setup_gee_auth') as mock_setup_auth:
            with patch('src.climate_gee.gee_extractor.create_from_config') as mock_create_coord:
                mock_setup_auth.return_value = mock_auth
                mock_create_coord.return_value = mock_coord_gen
                
                from src.climate_gee.gee_extractor import create_gee_extractor
                
                extractor = create_gee_extractor(
                    config_path=temp_config_file,
                    chunk_size=1000
                )
                
                assert isinstance(extractor, GEEClimateExtractor)
                assert extractor.chunk_size == 1000
    
    def test_real_coordinate_alignment(self, mock_authenticated_gee):
        """Test with real coordinate generator to verify alignment."""
        from src.climate_gee.coordinate_generator import CoordinateGenerator
        
        # Use real coordinate generator
        coord_gen = CoordinateGenerator(target_resolution=0.1)
        
        extractor = GEEClimateExtractor(
            mock_authenticated_gee,
            coord_gen,
            chunk_size=10
        )
        
        bounds = (0, 0, 0.2, 0.2)
        
        # Test extraction
        result_df = extractor.extract_climate_data(bounds, ['bio01'])
        
        # Verify coordinate alignment
        assert len(result_df) > 0
        x_unique = sorted(result_df['x'].unique())
        y_unique = sorted(result_df['y'].unique())
        
        # Check expected coordinate spacing
        if len(x_unique) > 1:
            spacing = x_unique[1] - x_unique[0]
            np.testing.assert_almost_equal(spacing, 0.1, decimal=5)
        
        if len(y_unique) > 1:
            spacing = y_unique[1] - y_unique[0]
            np.testing.assert_almost_equal(spacing, 0.1, decimal=5)
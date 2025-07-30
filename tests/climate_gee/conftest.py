"""Shared test fixtures for climate_gee module tests."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Mock database connections before importing climate_gee modules
with patch('src.database.connection.DatabaseManager'):
    with patch('psycopg2.pool.ThreadedConnectionPool'):
        from src.climate_gee.auth import GEEAuthenticator
        from src.climate_gee.coordinate_generator import CoordinateGenerator


@pytest.fixture
def temp_config():
    """Create temporary config file for testing."""
    config_data = {
        'resampling': {
            'target_resolution': 0.025
        },
        'processing_bounds': {
            'global': [-180, -90, 180, 90],
            'test_region': [-1, -1, 1, 1],
            'europe': [-10, 35, 40, 70]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def test_bounds():
    """Standard test bounds for small area testing."""
    return (0, 0, 0.1, 0.1)


@pytest.fixture
def large_test_bounds():
    """Larger test bounds for multi-chunk testing."""
    return (0, 0, 1, 1)


@pytest.fixture
def coordinate_generator():
    """Create real coordinate generator for testing."""
    return CoordinateGenerator(target_resolution=0.05)


@pytest.fixture
def mock_gee_authenticator():
    """Create mock GEE authenticator."""
    auth = Mock(spec=GEEAuthenticator)
    auth.is_authenticated.return_value = True
    
    # Mock Earth Engine module
    ee = Mock()
    ee.Image = Mock()
    ee.Geometry = Mock()
    ee.Feature = Mock()
    ee.FeatureCollection = Mock()
    
    auth.ee = ee
    return auth


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir
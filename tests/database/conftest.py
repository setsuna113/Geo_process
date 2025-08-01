"""Pytest configuration for database tests."""

import pytest
import logging
from src.database.setup import setup_database, reset_database
from src.database.connection import db

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Setup test database once per test session."""
    print("\n🧪 Setting up test database...")
    
    # Ensure database is properly set up
    success = setup_database()
    if not success:
        pytest.exit("Failed to setup test database")
    
    yield
    
    print("\n🧹 Test session cleanup...")
    # Optionally clean up after all tests
    # db.close_pool()

@pytest.fixture
def clean_database():
    """Provide clean database for individual tests that need it."""
    reset_database()
    yield
    # Test-specific cleanup if needed

@pytest.fixture
def sample_grid_data():
    """Provide sample grid data for tests."""
    return {
        'name': 'test_grid',
        'grid_type': 'cubic',
        'resolution': 1000,
        'bounds': 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))',
        'metadata': {'test': True}
    }

@pytest.fixture
def sample_species_data():
    """Provide sample species data for tests."""
    return {
        'species_name': 'Quercus alba',
        'scientific_name': 'Quercus alba',
        'genus': 'Quercus',
        'family': 'Fagaceae',
        'category': 'plant',
        'range_type': 'distribution',
        'geometry_wkt': 'POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))',
        'source_file': 'test_oak.gpkg',
        'confidence': 0.95,
        'area_km2': 25.0
    }

@pytest.fixture
def sample_raster_data():
    """Provide sample raster data for tests."""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    return {
        'name': f'test_raster_{unique_id}',
        'file_path': f'/tmp/test_raster_{unique_id}.tif',
        'data_type': 'Float32',
        'pixel_size_degrees': 0.016666666666667,
        'spatial_extent_wkt': 'POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))',
        'nodata_value': -9999.0,
        'band_count': 1,
        'file_size_mb': 125.5,
        'checksum': 'abc123def456',
        'last_modified': '2024-01-01T00:00:00Z',
        'source_dataset': 'Test Dataset',
        'variable_name': 'temperature',
        'units': 'degrees_celsius',
        'description': 'Test temperature data',
        'temporal_info': {'temporal_resolution': 'annual'},
        'metadata': {'test': True}
    }

@pytest.fixture
def sample_raster_tiles():
    """Provide sample raster tile data for tests."""
    return [
        {
            'tile_x': 0,
            'tile_y': 0,
            'tile_size_pixels': 1000,
            'tile_bounds_wkt': 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))',
            'file_byte_offset': 0,
            'file_byte_length': 1024,
            'tile_stats': {'min': 0, 'max': 100, 'mean': 50}
        },
        {
            'tile_x': 1,
            'tile_y': 0,
            'tile_size_pixels': 1000,
            'tile_bounds_wkt': 'POLYGON((10 0, 20 0, 20 10, 10 10, 10 0))',
            'file_byte_offset': 1024,
            'file_byte_length': 1024,
            'tile_stats': {'min': 10, 'max': 90, 'mean': 45}
        }
    ]

# Add marks for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
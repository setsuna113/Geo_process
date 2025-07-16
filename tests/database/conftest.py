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
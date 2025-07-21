"""Shared fixtures for grid system tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
import yaml
import shutil

from src.grid_systems import BoundsManager, GridFactory, BoundsDefinition
from src.database.connection import DatabaseManager
from src.database.schema import schema
from src.config.config import Config


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config_file(test_data_dir):
    """Create a real test config file."""
    config_data = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'geoprocess_db',
            'user': 'jason',
            'password': '123456',
            'pool_size': 5,
            'max_overflow': 10
        },
        'grids': {
            'default_bounds': [-180, -90, 180, 90],
            'cubic': {
                'use_postgis': False,  # Use Python implementation for tests
                'resolutions': [10000, 25000, 50000],
                'crs': 'EPSG:4326'
            },
            'hexagonal': {
                'chunk_size': 1000,
                'resolutions': [10000, 25000],
                'crs': 'EPSG:4326'
            }
        },
        'processing_bounds': {
            'custom': {
                'test_region': [0, 0, 10, 10]
            }
        }
    }
    
    config_path = test_data_dir / "test_config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_path


@pytest.fixture  
def test_config(test_config_file):
    """Create a real Config instance."""
    return Config(test_config_file)


@pytest.fixture
def test_db(test_config):
    """Create a real test database connection."""
    try:
        # Initialize database connection with real config
        db = DatabaseManager()
        
        # Test connection
        if not db.test_connection():
            raise Exception("Cannot connect to database")
        
        yield db
        
        # Clean up test data but keep schema
        try:
            with db.get_connection() as conn:
                cur = conn.cursor()
                # Clean up test data
                tables = [
                    'grid_cells',
                    'grids',
                    'experiments'
                ]
                for table in tables:
                    cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                conn.commit()
        except Exception as e:
            print(f"Warning: Failed to clean up test data: {e}")
            
    except Exception as e:
        pytest.skip(f"Could not connect to test database: {e}")


@pytest.fixture
def bounds_manager():
    """Create bounds manager instance."""
    return BoundsManager()


@pytest.fixture
def grid_factory(test_config):
    """Create grid factory instance with real config."""
    return GridFactory()


@pytest.fixture
def sample_bounds():
    """Sample bounds for testing."""
    return {
        'small': BoundsDefinition('small', (0, 0, 1, 1)),
        'medium': BoundsDefinition('medium', (-10, -10, 10, 10)),
        'large': BoundsDefinition('large', (-180, -90, 180, 90)),
        'custom': BoundsDefinition('custom', (10.5, 45.2, 15.7, 48.9))
    }
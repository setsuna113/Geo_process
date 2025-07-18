"""Shared fixtures for grid system tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.grid_systems import BoundsManager, GridFactory, BoundsDefinition
from src.database.connection import DatabaseManager
from src.database.schema import schema
from src.config import config


@pytest.fixture
def test_db():
    """Create a test database."""
    # Use in-memory database or test database
    test_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_biodiversity',
        'user': 'test_user',
        'password': 'test_pass'
    }
    
    db = DatabaseConnection(test_config)
    schema.db = db
    
    # Create tables
    schema.create_all_tables()
    
    yield db
    
    # Cleanup
    schema.drop_all_tables()
    db.close()


@pytest.fixture
def bounds_manager():
    """Create bounds manager instance."""
    return BoundsManager()


@pytest.fixture
def grid_factory():
    """Create grid factory instance."""
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


@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration."""
    test_config = {
        'grids': {
            'default_bounds': [-180, -90, 180, 90],
            'cubic': {
                'use_postgis': False  # Use Python implementation for tests
            },
            'hexagonal': {
                'chunk_size': 1000
            }
        },
        'processing_bounds': {
            'custom': {
                'test_region': [0, 0, 10, 10]
            }
        }
    }
    
    monkeypatch.setattr(config, '_config', test_config)
    monkeypatch.setattr(config, 'get', lambda key, default=None: 
                       test_config.get(key, default))
    
    return test_config
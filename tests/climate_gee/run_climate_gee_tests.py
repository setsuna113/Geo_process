#!/usr/bin/env python3
"""Test runner for climate_gee module tests."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable to indicate test mode
os.environ['TESTING'] = '1'

# Mock database components before any imports
def mock_database_imports():
    """Mock database-related imports to avoid connection issues."""
    import unittest.mock as mock
    
    # Create mock modules
    mock_psycopg2 = mock.MagicMock()
    mock_psycopg2.pool.ThreadedConnectionPool = mock.MagicMock()
    
    # Install the mocks
    sys.modules['psycopg2'] = mock_psycopg2
    sys.modules['psycopg2.pool'] = mock_psycopg2.pool
    
    # Mock the database manager to avoid connection attempts
    with patch('src.database.connection.DatabaseManager') as mock_db:
        mock_db.return_value = mock.MagicMock()
        return mock_db

# Apply database mocks
mock_database_imports()


def run_climate_gee_tests():
    """Run all climate_gee tests."""
    test_dir = Path(__file__).parent
    
    # Run pytest with verbose output
    args = [
        str(test_dir),
        '-v',
        '--tb=short',
        '-x',  # Stop on first failure
        '--disable-warnings'
    ]
    
    print("=" * 60)
    print("Running Climate GEE Module Tests")
    print("=" * 60)
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✅ All climate_gee tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_climate_gee_tests()
    sys.exit(exit_code)
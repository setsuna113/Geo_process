# src/config/test_config.py
"""Test configuration that extends the main config for testing purposes."""

from typing import Dict, Any
from .config import Config


class TestConfig(Config):
    """Test configuration that uses real database but with cleanup."""
    
    def __init__(self):
        super().__init__()
        
        # Override with test-specific settings while keeping existing database
        self.settings.update({
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'geoprocess_db',  # Use existing database
                'user': 'jason',
                'password': '123456'
            },
            'spatial_analysis': {
                'normalize_data': True,
                'save_results': True,
                'output_dir': 'test_output',
                'som': {
                    'grid_size': [3, 3],
                    'iterations': 50,
                    'sigma': 1.0,
                    'learning_rate': 0.5
                }
            },
            'testing': {
                'cleanup_after_test': True,
                'use_real_database': True,
                'test_data_retention_hours': 1  # Keep test data for 1 hour for debugging
            }
        })
    
    def get_test_tables(self) -> Dict[str, str]:
        """Return SQL for creating test tables."""
        return {
            'normalization_parameters': """
                CREATE TABLE IF NOT EXISTS normalization_parameters (
                    id SERIAL PRIMARY KEY,
                    method VARCHAR(50) NOT NULL,
                    parameters JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'som_results': """
                CREATE TABLE IF NOT EXISTS som_results (
                    id SERIAL PRIMARY KEY,
                    grid_size INTEGER[],
                    iterations INTEGER,
                    quantization_error FLOAT,
                    topographic_error FLOAT,
                    labels INTEGER[],
                    spatial_data JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
    
    def get_cleanup_queries(self) -> Dict[str, str]:
        """Return SQL queries for cleaning up test data."""
        retention_hours = self.settings['testing']['test_data_retention_hours']
        return {
            'normalization_parameters': f"""
                DELETE FROM normalization_parameters 
                WHERE created_at > NOW() - INTERVAL '{retention_hours} hours'
            """,
            'som_results': f"""
                DELETE FROM som_results 
                WHERE created_at > NOW() - INTERVAL '{retention_hours} hours'
            """
        }

#!/usr/bin/env python3
"""
Test database setup utilities for integration tests.
"""
import psycopg2
import logging
from typing import Dict, Any
from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.setup import setup_database

logger = logging.getLogger(__name__)

class TestDatabaseManager:
    """Manages test database creation and cleanup."""
    
    def __init__(self):
        self.test_db_name = "test_biodiversity_integration"
        self.admin_config = self._get_admin_config()
        
    def _get_admin_config(self) -> Dict[str, Any]:
        """Get admin database configuration for creating test database."""
        config = Config()
        admin_config = config.database.copy()
        admin_config['database'] = 'postgres'  # Connect to default postgres db first
        return admin_config
    
    def create_test_database(self) -> bool:
        """Create a test database for integration tests."""
        try:
            # Connect to default postgres database to create test database
            conn = psycopg2.connect(**self.admin_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Drop test database if it exists
            cursor.execute(f"DROP DATABASE IF EXISTS {self.test_db_name}")
            
            # Create test database
            cursor.execute(f"CREATE DATABASE {self.test_db_name}")
            
            cursor.close()
            conn.close()
            
            print(f"✅ Created test database: {self.test_db_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create test database: {e}")
            return False
    
    def setup_test_schema(self) -> bool:
        """Set up schema in test database."""
        try:
            # Create config for test database
            test_config = Config()
            test_config.settings['database']['database'] = self.test_db_name
            
            # Initialize database manager with test config
            db_manager = DatabaseManager()
            db_manager.pool = None  # Reset pool
            
            # Override the global config temporarily
            import src.database.connection as db_module
            original_config = db_module.config
            db_module.config = test_config
            
            # Reinitialize the database manager
            db_manager._create_pool()
            
            # Setup schema
            result = setup_database(reset=True)
            
            # Restore original config
            db_module.config = original_config
            
            if result:
                print("✅ Test database schema setup complete")
            else:
                print("❌ Test database schema setup failed")
                
            return result
            
        except Exception as e:
            print(f"❌ Failed to setup test schema: {e}")
            return False
    
    def cleanup_test_database(self) -> bool:
        """Clean up test database."""
        try:
            conn = psycopg2.connect(**self.admin_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Terminate connections to test database
            cursor.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{self.test_db_name}' AND pid <> pg_backend_pid()
            """)
            
            # Drop test database
            cursor.execute(f"DROP DATABASE IF EXISTS {self.test_db_name}")
            
            cursor.close()
            conn.close()
            
            print(f"✅ Cleaned up test database: {self.test_db_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to cleanup test database: {e}")
            return False
    
    def get_test_config(self) -> Config:
        """Get configuration for test database."""
        config = Config()
        config.settings['database']['database'] = self.test_db_name
        return config

def ensure_test_database() -> TestDatabaseManager:
    """Ensure test database is ready and return manager."""
    manager = TestDatabaseManager()
    
    if not manager.create_test_database():
        raise RuntimeError("Failed to create test database")
    
    if not manager.setup_test_schema():
        raise RuntimeError("Failed to setup test database schema")
    
    return manager

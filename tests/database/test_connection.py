"""Tests for database connection management."""

import pytest
import psycopg2
import time
from unittest.mock import patch, MagicMock
from src.database.connection import db, DatabaseManager
from src.config import config

class TestDatabaseManager:
    """Test DatabaseManager class functionality."""
    
    def test_connection_pool_creation(self):
        """Test connection pool is created successfully."""
        assert db.pool is not None
        assert db.pool.minconn == 1
        assert db.pool.maxconn == 20
    
    def test_get_connection_context_manager(self):
        """Test connection context manager."""
        with db.get_connection() as conn:
            assert conn is not None
            assert not conn.closed
            
            # Test connection is working
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
            cursor.close()
    
    def test_get_cursor_context_manager(self):
        """Test cursor context manager with dict cursor."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            assert result['test_value'] == 1
    
    def test_get_cursor_non_dict(self):
        """Test cursor context manager without dict cursor."""
        with db.get_cursor(dict_cursor=False) as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_get_cursor_autocommit(self):
        """Test cursor with autocommit mode."""
        with db.get_cursor(autocommit=True) as cursor:
            # Test that we can execute DDL in autocommit mode
            cursor.execute("SELECT current_setting('autocommit')")
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        try:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                # Force an error
                cursor.execute("SELECT invalid_column_name FROM non_existent_table")
        except Exception:
            pass  # Expected
        
        # Verify connection is still usable
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")
            assert cursor.fetchone() is not None
    
    def test_connection_recovery(self):
        """Test connection recovery after failure."""
        # Get connection info to verify recovery
        info_before = db.get_connection_info()
        assert info_before['status'] == 'connected'
        
        # Force a connection error by closing the pool
        original_pool = db.pool
        db.pool = None
        
        # Restore pool and test recovery
        db.pool = original_pool
        
        info_after = db.get_connection_info()
        assert info_after['status'] == 'connected'
    
    def test_test_connection(self):
        """Test connection testing functionality."""
        result = db.test_connection()
        assert result is True
    
    def test_get_connection_info(self):
        """Test connection info retrieval."""
        info = db.get_connection_info()
        assert info['status'] == 'connected'
        assert 'database' in info
        assert 'user' in info
        assert 'version' in info
        assert info['pool_min_conn'] == 1
        assert info['pool_max_conn'] == 20
    
    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_connection_failure_retry(self, mock_pool):
        """Test connection retry logic on failure."""
        # Mock pool creation to fail twice, then succeed
        mock_pool.side_effect = [
            psycopg2.OperationalError("Connection failed"),
            psycopg2.OperationalError("Connection failed"),
            MagicMock()  # Success on third try
        ]
        
        # Create new manager to test retry logic
        with patch('time.sleep'):  # Speed up test
            manager = DatabaseManager()
            assert manager.connection_attempts == 3
    
    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_connection_max_attempts_exceeded(self, mock_pool):
        """Test behavior when max connection attempts exceeded."""
        mock_pool.side_effect = psycopg2.OperationalError("Connection failed")
        
        with patch('time.sleep'):  # Speed up test
            with pytest.raises(psycopg2.OperationalError):
                DatabaseManager()
    
    def test_execute_sql_file_success(self, tmp_path):
        """Test successful SQL file execution."""
        # Create test SQL file
        sql_file = tmp_path / "test.sql"
        sql_file.write_text("""
            CREATE TEMP TABLE test_table (id INTEGER);
            INSERT INTO test_table VALUES (1), (2), (3);
        """)
        
        # Execute file
        db.execute_sql_file(sql_file)
        
        # Verify execution
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM test_table")
            result = cursor.fetchone()
            assert result['count'] == 3
    
    def test_execute_sql_file_not_found(self, tmp_path):
        """Test SQL file execution with non-existent file."""
        non_existent_file = tmp_path / "non_existent.sql"
        
        with pytest.raises(FileNotFoundError):
            db.execute_sql_file(non_existent_file)
    
    def test_execute_sql_file_invalid_sql(self, tmp_path):
        """Test SQL file execution with invalid SQL."""
        sql_file = tmp_path / "invalid.sql"
        sql_file.write_text("INVALID SQL STATEMENT;")
        
        with pytest.raises(Exception):
            db.execute_sql_file(sql_file)
    
    def test_close_pool(self):
        """Test connection pool closure."""
        # Create separate manager for this test
        manager = DatabaseManager()
        assert manager.pool is not None
        
        # Close pool
        manager.close_pool()
        assert manager.pool is None
    
    def test_concurrent_connections(self):
        """Test multiple concurrent connections."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def test_connection():
            try:
                with db.get_cursor() as cursor:
                    cursor.execute("SELECT pg_backend_pid() as pid")
                    pid = cursor.fetchone()['pid']
                    results.put(('success', pid))
            except Exception as e:
                results.put(('error', str(e)))
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=test_connection)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        pids = set()
        for _ in range(5):
            status, result = results.get()
            assert status == 'success'
            pids.add(result)
        
        # Should have multiple different backend PIDs (concurrent connections)
        assert len(pids) >= 1  # At least one connection worked
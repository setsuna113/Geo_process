"""Test mode operations for database schema."""

from typing import Dict, List, Any
import json
import logging
from ..connection import db

logger = logging.getLogger(__name__)


class TestOperations:
    """Test mode operations for database cleanup and management."""
    
    def cleanup_test_data(self, force: bool = False) -> Dict[str, int]:
        """Clean up test data with multiple safety checks."""
        # Validate test mode
        db.validate_test_mode_operation()
        
        from src.config import config
        testing_config = config.testing
        
        if not testing_config.get('cleanup_after_test', True) and not force:
            logger.info("Test cleanup disabled in config")
            return {}
        
        allowed_tables = testing_config.get('allowed_cleanup_tables', [])
        retention_hours = testing_config.get('test_data_retention_hours', 1)
        markers = testing_config.get('test_data_markers', {})
        
        cleanup_results = {}
        
        with db.get_cursor() as cursor:
            for table in allowed_tables:
                try:
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' AND table_name = %s
                        )
                    """, (table,))
                    
                    if not cursor.fetchone()['exists']:
                        continue
                    
                    # Build safe cleanup query based on table
                    deleted_count = self._cleanup_table_test_data(
                        cursor, table, retention_hours, markers
                    )
                    
                    cleanup_results[table] = deleted_count
                    
                    if deleted_count > 0:
                        logger.info(f"🧹 Cleaned {deleted_count} test records from {table}")
                        
                except Exception as e:
                    logger.error(f"Failed to cleanup {table}: {e}")
                    cleanup_results[table] = 0
        
        total_cleaned = sum(cleanup_results.values())
        logger.info(f"🧪 Test cleanup complete: {total_cleaned} total records removed")
        
        return cleanup_results

    def _cleanup_table_test_data(self, cursor, table: str, 
                                retention_hours: int, markers: Dict) -> int:
        """Clean up test data from a specific table."""
        conditions = []
        params = []
        
        # Time-based condition
        if 'created_at' in self._get_table_columns(cursor, table):
            conditions.append("created_at > NOW() - INTERVAL %s")
            params.append(f'{retention_hours} hours')
        
        # Test prefix condition
        prefix = markers.get('name_prefix', 'TEST_')
        name_columns = self._get_name_columns(cursor, table)
        
        if name_columns and markers.get('require_test_prefix', True):
            prefix_conditions = []
            for col in name_columns:
                prefix_conditions.append(f"{col} LIKE %s")
                params.append(f'{prefix}%')
            
            if prefix_conditions:
                conditions.append(f"({' OR '.join(prefix_conditions)})")
        
        # Created by condition
        if 'created_by' in self._get_table_columns(cursor, table):
            conditions.append("created_by = %s")
            params.append(markers.get('created_by', 'pytest'))
        
        # Metadata marker condition for JSON fields
        if 'metadata' in self._get_table_columns(cursor, table):
            conditions.append("metadata->%s IS NOT NULL")
            params.append(markers.get('metadata_key', '__test_data__'))
        
        if not conditions:
            logger.warning(f"No test data conditions for table {table}")
            return 0
        
        # Use parameterized query for safety
        query = f"DELETE FROM {table} WHERE {' AND '.join(conditions)} RETURNING id"
        cursor.execute(query, params)
        
        return cursor.rowcount

    def _get_table_columns(self, cursor, table: str) -> List[str]:
        """Get column names for a table."""
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = %s
        """, (table,))
        return [row['column_name'] for row in cursor.fetchall()]

    def _get_name_columns(self, cursor, table: str) -> List[str]:
        """Get name-like columns for test prefix matching."""
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = %s 
            AND column_name IN ('name', 'job_name', 'species_name', 
                            'experiment_name', 'description', 'variable')
        """, (table,))
        return [row['column_name'] for row in cursor.fetchall()]

    def mark_test_data(self, table: str, record_id: str) -> bool:
        """Mark a record as test data for later cleanup."""
        if not db.is_test_mode:
            return False
        
        from src.config import config
        metadata_key = config.testing.get('test_data_markers', {}).get('metadata_key', '__test_data__')
        
        with db.get_cursor() as cursor:
            # Check for different JSON column names based on table
            json_column = None
            if 'metadata' in self._get_table_columns(cursor, table):
                json_column = 'metadata'
            elif 'config' in self._get_table_columns(cursor, table):
                json_column = 'config'
            elif 'temporal_info' in self._get_table_columns(cursor, table):
                json_column = 'temporal_info'
            elif 'computation_metadata' in self._get_table_columns(cursor, table):
                json_column = 'computation_metadata'
            elif 'tile_stats' in self._get_table_columns(cursor, table):
                json_column = 'tile_stats'
            elif 'results' in self._get_table_columns(cursor, table):
                json_column = 'results'
            
            if json_column:
                marker_data = json.dumps({metadata_key: True})
                cursor.execute(f"""
                    UPDATE {table} 
                    SET {json_column} = COALESCE({json_column}, '{{}}') || %s::jsonb
                    WHERE id = %s
                """, (marker_data, record_id))
                return cursor.rowcount > 0
        
        return False
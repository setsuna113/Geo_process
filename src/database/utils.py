"""Database utilities for schema introspection and mapping."""

import logging
from typing import List, Dict, Optional, Any
from .connection import DatabaseManager
from ..config import config

logger = logging.getLogger(__name__)


class DatabaseSchemaUtils:
    """Utility class for database schema introspection and column mapping."""
    
    def __init__(self, db_connection: DatabaseManager, config_instance=None):
        """
        Initialize database schema utilities.
        
        Args:
            db_connection: Database manager instance
            config_instance: Configuration instance (uses global if None)
        """
        self.db = db_connection
        self.config = config_instance or config
        self._column_cache = {}  # Cache for column information
    
    def get_geometry_column(self, table_name: str) -> str:
        """
        Get geometry column name for a table using config + introspection.
        
        Args:
            table_name: Name of the database table
            
        Returns:
            Name of the geometry column
        """
        # Check config first for this table
        schema_mapping = self.config.get('database_schema_mapping', {})
        table_config = schema_mapping.get(table_name, {})
        
        # Try configured primary geometry column
        primary_geo_col = table_config.get('geometry_column')
        if primary_geo_col and self._column_exists(table_name, primary_geo_col):
            logger.debug(f"Using configured geometry column '{primary_geo_col}' for {table_name}")
            return primary_geo_col
        
        # Try PostGIS geometry_columns system table
        postgis_column = self._get_postgis_geometry_column(table_name)
        if postgis_column:
            logger.debug(f"Found PostGIS geometry column '{postgis_column}' for {table_name}")
            return postgis_column
        
        # Try configured fallback columns
        fallback_columns = table_config.get('fallback_geometry_columns', [])
        for fallback_col in fallback_columns:
            if self._column_exists(table_name, fallback_col):
                logger.debug(f"Using fallback geometry column '{fallback_col}' for {table_name}")
                return fallback_col
        
        # Try common geometry column names as last resort
        common_geo_names = ['geometry', 'geom', 'shape', 'spatial_extent', 'bounds']
        for geo_name in common_geo_names:
            if self._column_exists(table_name, geo_name):
                logger.warning(f"Using detected geometry column '{geo_name}' for {table_name}")
                return geo_name
        
        # Ultimate fallback - log warning and return a reasonable default
        logger.error(f"Could not detect geometry column for {table_name}, using 'geometry'")
        return 'geometry'
    
    def get_active_column(self, table_name: str) -> Optional[str]:
        """
        Get the 'active' status column name for a table.
        
        Args:
            table_name: Name of the database table
            
        Returns:
            Name of the active column or None if not configured/found
        """
        schema_mapping = self.config.get('database_schema_mapping', {})
        table_config = schema_mapping.get(table_name, {})
        
        active_col = table_config.get('active_column')
        if active_col and self._column_exists(table_name, active_col):
            return active_col
        
        # Try common active column names
        common_active_names = ['active', 'is_active', 'enabled']
        for active_name in common_active_names:
            if self._column_exists(table_name, active_name):
                return active_name
        
        return None
    
    def get_metadata_column(self, table_name: str) -> Optional[str]:
        """
        Get the JSON metadata column name for a table.
        
        Args:
            table_name: Name of the database table
            
        Returns:
            Name of the metadata column or None if not found
        """
        schema_mapping = self.config.get('database_schema_mapping', {})
        table_config = schema_mapping.get(table_name, {})
        
        metadata_col = table_config.get('metadata_column')
        if metadata_col and self._column_exists(table_name, metadata_col):
            return metadata_col
        
        # Try common metadata column names
        common_metadata_names = ['metadata', 'meta', 'properties', 'attributes']
        for meta_name in common_metadata_names:
            if self._column_exists(table_name, meta_name):
                return meta_name
        
        return None
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get all column names for a table.
        
        Args:
            table_name: Name of the database table
            
        Returns:
            List of column names
        """
        if table_name in self._column_cache:
            return self._column_cache[table_name]
        
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
                
                columns = [row['column_name'] for row in cursor.fetchall()]
                self._column_cache[table_name] = columns
                return columns
                
        except Exception as e:
            logger.error(f"Error getting columns for {table_name}: {e}")
            return []
    
    def get_table_schema_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive schema information for a table.
        
        Args:
            table_name: Name of the database table
            
        Returns:
            Dictionary with schema information
        """
        columns = self.get_table_columns(table_name)
        
        return {
            'table_name': table_name,
            'columns': columns,
            'geometry_column': self.get_geometry_column(table_name),
            'active_column': self.get_active_column(table_name),
            'metadata_column': self.get_metadata_column(table_name),
            'total_columns': len(columns)
        }
    
    def build_select_query(self, table_name: str, columns: List[str], 
                          include_geometry_bounds: bool = False,
                          where_conditions: Optional[List[str]] = None,
                          include_active_filter: bool = True) -> tuple:
        """
        Build a SELECT query with proper column mapping.
        
        Args:
            table_name: Name of the table
            columns: List of columns to select
            include_geometry_bounds: Whether to include ST_XMin, ST_YMin, etc.
            where_conditions: Additional WHERE conditions
            include_active_filter: Whether to filter by active=true
            
        Returns:
            Tuple of (query_string, parameters_list)
        """
        select_parts = columns.copy()
        
        # Add geometry bounds if requested
        if include_geometry_bounds:
            geo_col = self.get_geometry_column(table_name)
            select_parts.extend([
                f"ST_XMin({geo_col}) as min_x",
                f"ST_YMin({geo_col}) as min_y", 
                f"ST_XMax({geo_col}) as max_x",
                f"ST_YMax({geo_col}) as max_y"
            ])
        
        query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
        
        # Build WHERE clause
        where_parts = []
        params = []
        
        if include_active_filter:
            active_col = self.get_active_column(table_name)
            if active_col:
                where_parts.append(f"{active_col} = true")
        
        if where_conditions:
            where_parts.extend(where_conditions)
        
        if where_parts:
            query += f" WHERE {' AND '.join(where_parts)}"
        
        return query, params
    
    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND column_name = %s
                    AND table_schema = 'public'
                """, (table_name, column_name))
                
                return cursor.fetchone() is not None
                
        except Exception as e:
            logger.error(f"Error checking column {column_name} in {table_name}: {e}")
            return False
    
    def _get_postgis_geometry_column(self, table_name: str) -> Optional[str]:
        """Get geometry column from PostGIS geometry_columns table."""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT f_geometry_column 
                    FROM geometry_columns 
                    WHERE f_table_name = %s 
                    AND f_table_schema = 'public'
                    LIMIT 1
                """, (table_name,))
                
                result = cursor.fetchone()
                return result['f_geometry_column'] if result else None
                
        except Exception as e:
            logger.debug(f"PostGIS geometry_columns query failed for {table_name}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the column information cache."""
        self._column_cache.clear()
        logger.debug("Cleared database schema cache")


# Global utility instance - can be used across modules
schema_utils = None

def get_schema_utils(db_connection: DatabaseManager = None) -> DatabaseSchemaUtils:
    """
    Get or create global schema utils instance.
    
    Args:
        db_connection: Database connection (required for first call)
        
    Returns:
        DatabaseSchemaUtils instance
    """
    global schema_utils
    
    if schema_utils is None:
        if db_connection is None:
            raise ValueError("db_connection required for first call to get_schema_utils")
        schema_utils = DatabaseSchemaUtils(db_connection)
    
    return schema_utils
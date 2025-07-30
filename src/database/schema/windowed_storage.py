"""Windowed storage operations for memory-efficient raster processing."""

from typing import List, Tuple, Dict, Any, Optional
import logging
from psycopg2.extras import execute_values
from ..connection import db

logger = logging.getLogger(__name__)


class WindowedStorage:
    """Windowed storage operations for memory-efficient raster processing."""
    
    def insert_raster_chunk(self, table_name: str, chunk_data: List[Tuple],
                          batch_size: int = 1000) -> int:
        """Insert a chunk of raster data with proper indexing.
        
        Args:
            table_name: Target table name
            chunk_data: List of tuples (row_idx, col_idx, x_coord, y_coord, value)
            batch_size: Batch size for inserts
            
        Returns:
            Number of rows inserted
        """
        with db.get_cursor() as cursor:
            # Ensure table exists with proper schema
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    row_idx INTEGER,
                    col_idx INTEGER,
                    x_coord DOUBLE PRECISION,
                    y_coord DOUBLE PRECISION,
                    value DOUBLE PRECISION,
                    PRIMARY KEY (row_idx, col_idx)
                );
                
                CREATE INDEX IF NOT EXISTS {table_name}_spatial_idx 
                ON {table_name} (x_coord, y_coord);
                
                CREATE INDEX IF NOT EXISTS {table_name}_value_idx 
                ON {table_name} (value) 
                WHERE value IS NOT NULL;
            """)
            
            # Batch insert with conflict handling
            execute_values(
                cursor,
                f"""
                INSERT INTO {table_name} (row_idx, col_idx, x_coord, y_coord, value) 
                VALUES %s
                ON CONFLICT (row_idx, col_idx) 
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    x_coord = EXCLUDED.x_coord,
                    y_coord = EXCLUDED.y_coord
                """,
                chunk_data,
                page_size=batch_size
            )
            
            return len(chunk_data)
    
    def create_windowed_storage_table(self, table_name: str) -> None:
        """Create optimized table for windowed raster storage.
        
        Args:
            table_name: Name of table to create
        """
        with db.get_cursor() as cursor:
            cursor.execute(f"""
                -- Drop if exists to ensure clean state
                DROP TABLE IF EXISTS {table_name} CASCADE;
                
                -- Create table with optimized structure
                CREATE TABLE {table_name} (
                    row_idx INTEGER NOT NULL,
                    col_idx INTEGER NOT NULL,
                    x_coord DOUBLE PRECISION NOT NULL,
                    y_coord DOUBLE PRECISION NOT NULL,
                    value DOUBLE PRECISION,
                    PRIMARY KEY (row_idx, col_idx)
                );
                
                -- Spatial index for coordinate-based queries
                CREATE INDEX {table_name}_spatial_idx 
                ON {table_name} USING GIST (
                    ST_MakePoint(x_coord, y_coord)
                );
                
                -- Value index for filtering
                CREATE INDEX {table_name}_value_idx 
                ON {table_name} (value) 
                WHERE value IS NOT NULL;
                
                -- Row/col composite index for range queries
                CREATE INDEX {table_name}_rowcol_idx 
                ON {table_name} (row_idx, col_idx);
                
                -- Add table comment
                COMMENT ON TABLE {table_name} IS 
                'Windowed storage table for memory-efficient raster processing';
            """)
            
            logger.info(f"Created windowed storage table: {table_name}")
    
    def get_raster_chunk_bounds(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get spatial bounds and metadata for a windowed raster table.
        
        Args:
            table_name: Table name
            
        Returns:
            Dictionary with bounds and metadata
        """
        with db.get_cursor() as cursor:
            cursor.execute(f"""
                SELECT 
                    MIN(x_coord) as min_x,
                    MAX(x_coord) as max_x,
                    MIN(y_coord) as min_y,
                    MAX(y_coord) as max_y,
                    COUNT(*) as pixel_count,
                    COUNT(DISTINCT row_idx) as height,
                    COUNT(DISTINCT col_idx) as width,
                    AVG(value) as mean_value,
                    MIN(value) as min_value,
                    MAX(value) as max_value
                FROM {table_name}
                WHERE value IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result['pixel_count'] > 0:
                return {
                    'bounds': (result['min_x'], result['min_y'], 
                              result['max_x'], result['max_y']),
                    'shape': (result['height'], result['width']),
                    'pixel_count': result['pixel_count'],
                    'statistics': {
                        'mean': result['mean_value'],
                        'min': result['min_value'],
                        'max': result['max_value']
                    }
                }
            return None
    
    def migrate_legacy_table_to_coordinates(self, table_name: str, 
                                          bounds: Tuple[float, float, float, float],
                                          resolution: float) -> bool:
        """Migrate legacy table to include coordinate columns.
        
        Args:
            table_name: Table to migrate
            bounds: Geographic bounds (min_x, min_y, max_x, max_y)
            resolution: Pixel resolution in degrees
            
        Returns:
            True if migration successful
        """
        logger.info(f"Migrating table {table_name} to include coordinates")
        
        try:
            with db.get_cursor() as cursor:
                # Check if migration already done
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    AND column_name IN ('x_coord', 'y_coord')
                """, (table_name,))
                
                existing_cols = [row['column_name'] for row in cursor.fetchall()]
                
                if 'x_coord' in existing_cols and 'y_coord' in existing_cols:
                    logger.info(f"Table {table_name} already has coordinate columns")
                    return True
                
                # Add coordinate columns
                cursor.execute(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN IF NOT EXISTS x_coord DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS y_coord DOUBLE PRECISION
                """)
                
                # Calculate coordinates from indices
                min_x, min_y, max_x, max_y = bounds
                
                # Update coordinates based on row/col indices
                # Note: row 0 is at max_y (top), increasing rows go down
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET 
                        x_coord = {min_x} + (col_idx + 0.5) * {resolution},
                        y_coord = {max_y} - (row_idx + 0.5) * {resolution}
                    WHERE x_coord IS NULL OR y_coord IS NULL
                """)
                
                # Create spatial index
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_spatial_idx 
                    ON {table_name} USING GIST (
                        ST_MakePoint(x_coord, y_coord)
                    )
                """)
                
                # Add coordinate indexes
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_x_coord_idx ON {table_name}(x_coord);
                    CREATE INDEX IF NOT EXISTS {table_name}_y_coord_idx ON {table_name}(y_coord);
                """)
                
                db.commit()
                logger.info(f"Successfully migrated table {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to migrate table {table_name}: {e}")
            db.rollback()
            return False
    
    def ensure_table_has_coordinates(self, table_name: str) -> bool:
        """Check if table has coordinate columns.
        
        Args:
            table_name: Table to check
            
        Returns:
            True if table has x_coord and y_coord columns
        """
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as col_count
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND column_name IN ('x_coord', 'y_coord')
            """, (table_name,))
            
            result = cursor.fetchone()
            return result['col_count'] == 2
"""Windowed storage manager for memory-efficient raster processing.

This module provides chunked storage operations for large rasters,
enabling processing without loading entire datasets into memory.
"""

import rasterio
from rasterio.windows import Window
import numpy as np
from typing import Iterator, Tuple, Optional, Dict, Any, Callable
from pathlib import Path
import logging
from psycopg2.extras import execute_values

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_stage
from src.database.schema import DatabaseSchema

logger = get_logger(__name__)


class WindowedStorageManager:
    """Handles chunked storage operations for large rasters.
    
    This class enables memory-efficient processing by:
    - Reading rasters in configurable windows/chunks
    - Storing data progressively without full dataset in memory
    - Supporting both passthrough and resampled data storage
    """
    
    def __init__(self, window_size: int = 2048, overlap: int = 0):
        """Initialize windowed storage manager.
        
        Args:
            window_size: Size of windows to process (pixels)
            overlap: Overlap between windows to avoid edge artifacts
        """
        self.window_size = window_size
        self.overlap = overlap
        self.schema = DatabaseSchema()
        
    def iter_windows(self, raster_path: str) -> Iterator[Tuple[Window, Tuple[int, int]]]:
        """Iterate over windows for a raster file.
        
        Args:
            raster_path: Path to raster file
            
        Yields:
            Tuple of (Window object, (row_offset, col_offset))
        """
        with rasterio.open(raster_path) as src:
            height, width = src.shape
            
            # Calculate step size accounting for overlap
            step_size = self.window_size - self.overlap
            
            for row_off in range(0, height, step_size):
                # Calculate actual window height
                row_size = min(self.window_size, height - row_off)
                
                for col_off in range(0, width, step_size):
                    # Calculate actual window width
                    col_size = min(self.window_size, width - col_off)
                    
                    window = Window(col_off, row_off, col_size, row_size)
                    yield window, (row_off, col_off)
    
    @log_stage("windowed_passthrough_storage")
    def store_passthrough_windowed(self, 
                                 raster_path: str, 
                                 table_name: str,
                                 db_connection,
                                 bounds: Tuple[float, float, float, float],
                                 progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """Stream copy raster data using windows without loading all.
        
        Args:
            raster_path: Path to source raster
            table_name: Target database table
            db_connection: Database connection
            bounds: Geographic bounds (minx, miny, maxx, maxy)
            progress_callback: Optional progress reporting callback
            
        Returns:
            Dictionary with storage statistics
        """
        logger.info(f"Starting windowed passthrough storage for {table_name}")
        
        stats = {
            'total_windows': 0,
            'processed_windows': 0,
            'total_pixels': 0,
            'stored_pixels': 0,
            'skipped_pixels': 0
        }
        
        try:
            with rasterio.open(raster_path) as src:
                # Calculate total windows for progress
                height, width = src.shape
                step_size = self.window_size - self.overlap
                total_windows = ((height + step_size - 1) // step_size) * \
                              ((width + step_size - 1) // step_size)
                stats['total_windows'] = total_windows
                
                # Get transform for coordinate calculations
                transform = src.transform
                nodata = src.nodata
                
                # Process each window
                for window_idx, (window, (row_off, col_off)) in enumerate(self.iter_windows(raster_path)):
                    # Read window data
                    data = src.read(1, window=window)
                    stats['total_pixels'] += data.size
                    
                    # Calculate geographic bounds for this window
                    window_bounds = self._calculate_window_bounds(
                        window, transform, src.crs
                    )
                    
                    # Store this chunk
                    stored = self._store_chunk(
                        db_connection, table_name, data, 
                        row_off, col_off, transform, nodata
                    )
                    stats['stored_pixels'] += stored
                    stats['skipped_pixels'] += (data.size - stored)
                    
                    stats['processed_windows'] += 1
                    
                    # Report progress
                    if progress_callback and window_idx % 10 == 0:
                        progress = (window_idx + 1) / total_windows * 100
                        progress_callback(
                            f"Processing window {window_idx + 1}/{total_windows}", 
                            progress
                        )
                    
                    # Log performance metrics periodically
                    if window_idx % 100 == 0:
                        logger.log_performance(
                            "window_processing",
                            window_idx / total_windows,
                            items_processed=stats['stored_pixels'],
                            windows_completed=stats['processed_windows']
                        )
                
                logger.info(
                    f"Completed windowed storage: {stats['processed_windows']} windows, "
                    f"{stats['stored_pixels']:,} pixels stored"
                )
                
        except Exception as e:
            logger.error(
                "Windowed storage failed",
                exc_info=True,
                extra={
                    'context': {
                        'table': table_name,
                        'raster': raster_path,
                        'windows_processed': stats['processed_windows']
                    }
                }
            )
            raise
            
        return stats
    
    @log_stage("windowed_resampled_storage")
    def store_resampled_windowed(self,
                               table_name: str,
                               db_connection,
                               progress_callback: Optional[Callable[[str, float], None]] = None) -> None:
        """Accept resampled chunks and store progressively.
        
        This method is designed to be called iteratively as resampled
        chunks are produced, avoiding memory accumulation.
        
        Args:
            table_name: Target database table
            db_connection: Database connection
            progress_callback: Optional progress callback
        """
        # This will be called by the resampling engine
        # Implementation depends on resampling engine refactoring
        logger.info(f"Ready to receive resampled chunks for {table_name}")
    
    def _store_chunk(self, 
                    db_connection,
                    table_name: str,
                    data: np.ndarray,
                    row_offset: int,
                    col_offset: int,
                    transform,
                    nodata: Optional[float] = None) -> int:
        """Store a single chunk to database.
        
        Args:
            db_connection: Database connection
            table_name: Target table
            data: Chunk data array
            row_offset: Row offset in full raster
            col_offset: Column offset in full raster
            transform: Raster transform for coordinate calculation
            nodata: NoData value to skip
            
        Returns:
            Number of pixels stored
        """
        # Extract valid (non-nodata) values
        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = ~np.isnan(data)
            
        if not np.any(valid_mask):
            return 0
        
        rows, cols = np.where(valid_mask)
        values = data[valid_mask]
        
        # Adjust indices by offset
        global_rows = rows + row_offset
        global_cols = cols + col_offset
        
        # Calculate geographic coordinates (vectorized)
        # Transform: [a, b, c, d, e, f] where x = a*col + b*row + c, y = d*col + e*row + f
        x_coords = transform[0] * global_cols + transform[1] * global_rows + transform[2]
        y_coords = transform[3] * global_cols + transform[4] * global_rows + transform[5]
        
        # Prepare batch insert data
        data_to_insert = [
            (int(r), int(c), float(x), float(y), float(v))
            for r, c, x, y, v in zip(global_rows, global_cols, x_coords, y_coords, values)
        ]
        
        # Insert using batch operation
        try:
            with db_connection.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create table if not exists (but this should already be created by create_storage_table)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            row_idx INTEGER,
                            col_idx INTEGER,
                            x_coord DOUBLE PRECISION,
                            y_coord DOUBLE PRECISION,
                            value DOUBLE PRECISION,
                            PRIMARY KEY (row_idx, col_idx)
                        )
                    """)
                    
                    # Batch insert with conflict handling
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {table_name} (row_idx, col_idx, x_coord, y_coord, value) 
                        VALUES %s
                        ON CONFLICT (row_idx, col_idx) 
                        DO UPDATE SET value = EXCLUDED.value
                        """,
                        data_to_insert,
                        page_size=50000  # Match batch_insert_size from config
                    )
                conn.commit()
                
        except Exception as e:
            logger.error(
                f"Failed to store chunk",
                exc_info=True,
                extra={
                    'context': {
                        'table': table_name,
                        'chunk_size': len(data_to_insert),
                        'offset': (row_offset, col_offset)
                    }
                }
            )
            raise
            
        return len(data_to_insert)
    
    def _calculate_window_bounds(self, window: Window, transform, crs) -> Tuple[float, float, float, float]:
        """Calculate geographic bounds for a raster window.
        
        Args:
            window: Rasterio Window object
            transform: Raster transform
            crs: Coordinate reference system
            
        Returns:
            Bounds tuple (minx, miny, maxx, maxy)
        """
        # Get corner coordinates
        min_col, min_row = window.col_off, window.row_off
        max_col = min_col + window.width
        max_row = min_row + window.height
        
        # Transform to geographic coordinates
        minx, maxy = transform * (min_col, min_row)
        maxx, miny = transform * (max_col, max_row)
        
        return (minx, miny, maxx, maxy)
    
    def create_storage_table(self, table_name: str, db_connection) -> None:
        """Create table for windowed storage if not exists.
        
        Args:
            table_name: Name of table to create
            db_connection: Database connection
        """
        with db_connection.get_connection() as conn:
            with conn.cursor() as cur:
                # First create the table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        row_idx INTEGER,
                        col_idx INTEGER,
                        x_coord DOUBLE PRECISION,
                        y_coord DOUBLE PRECISION,
                        value DOUBLE PRECISION,
                        PRIMARY KEY (row_idx, col_idx)
                    );
                """)
                
                # Then create indexes separately for better error handling
                # GIST spatial index for coordinate-based queries (most important for merge stage)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_spatial_gist_idx 
                    ON {table_name} USING GIST (
                        ST_MakePoint(x_coord, y_coord)
                    );
                """)
                
                # Individual coordinate indexes for range queries
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_x_coord_idx 
                    ON {table_name} (x_coord);
                """)
                
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_y_coord_idx 
                    ON {table_name} (y_coord);
                """)
                
                # Composite btree index for coordinate pairs (backup for non-spatial queries)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_spatial_btree_idx 
                    ON {table_name} (x_coord, y_coord);
                """)
                
                # Value index for filtering
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_value_idx 
                    ON {table_name} (value) 
                    WHERE value IS NOT NULL;
                """)
                
                # Row/col composite index for window-based access
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_rowcol_idx 
                    ON {table_name} (row_idx, col_idx);
                """)
                
            conn.commit()
            
        logger.info(f"Created storage table with optimized indexes: {table_name}")
    
    def validate_coordinate_accuracy(self, table_name: str, db_connection,
                                   bounds: Tuple[float, float, float, float],
                                   resolution: float,
                                   sample_size: int = 100) -> Dict[str, Any]:
        """Validate that stored coordinates match expected values.
        
        Args:
            table_name: Table to validate
            db_connection: Database connection
            bounds: Expected geographic bounds
            resolution: Expected pixel resolution
            sample_size: Number of samples to check
            
        Returns:
            Validation results dictionary
        """
        min_x, min_y, max_x, max_y = bounds
        
        with db_connection.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sample some points
            cursor.execute(f"""
                SELECT row_idx, col_idx, x_coord, y_coord
                FROM {table_name}
                WHERE value IS NOT NULL
                LIMIT %s
            """, (sample_size,))
            
            errors = []
            max_error = 0.0
            
            for row in cursor.fetchall():
                row_idx, col_idx, x_coord, y_coord = row
                
                # Calculate expected coordinates
                expected_x = min_x + (col_idx + 0.5) * resolution
                expected_y = max_y - (row_idx + 0.5) * resolution
                
                # Check error
                x_error = abs(x_coord - expected_x)
                y_error = abs(y_coord - expected_y)
                
                if x_error > 1e-6 or y_error > 1e-6:
                    errors.append({
                        'row_idx': row_idx,
                        'col_idx': col_idx,
                        'x_error': x_error,
                        'y_error': y_error
                    })
                    
                max_error = max(max_error, x_error, y_error)
            
            # Check bounds
            cursor.execute(f"""
                SELECT 
                    MIN(x_coord) as min_x,
                    MAX(x_coord) as max_x,
                    MIN(y_coord) as min_y,
                    MAX(y_coord) as max_y
                FROM {table_name}
                WHERE value IS NOT NULL
            """)
            
            actual_bounds = cursor.fetchone()
            
            return {
                'valid': len(errors) == 0,
                'max_error': max_error,
                'error_count': len(errors),
                'sample_size': sample_size,
                'errors': errors[:10],  # First 10 errors
                'expected_bounds': bounds,
                'actual_bounds': actual_bounds,
                'bounds_match': (
                    abs(actual_bounds[0] - min_x) < resolution and
                    abs(actual_bounds[1] - max_x) < resolution and
                    abs(actual_bounds[2] - min_y) < resolution and
                    abs(actual_bounds[3] - max_y) < resolution
                ) if actual_bounds[0] is not None else False
            }
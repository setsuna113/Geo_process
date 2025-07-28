"""Database data reader for passthrough and resampled tables."""
import numpy as np
from typing import Optional, Tuple
import logging
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class DBDataReader:
    """Reads data from passthrough and resampled tables with proper coordinate mapping."""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        
    def read_passthrough_data(self, dataset_name: str, 
                            bounds: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Read data from passthrough table.
        
        Args:
            dataset_name: Name like 'terrestrial-richness' 
            bounds: Optional (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            2D numpy array with data
        """
        # Normalize name to table name
        table_name = f"passthrough_{dataset_name.replace('-', '_')}"
        return self._read_from_table(table_name, bounds)
    
    def read_resampled_data(self, dataset_name: str,
                          bounds: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Read data from resampled table.
        
        Args:
            dataset_name: Name like 'terrestrial-richness' 
            bounds: Optional (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            2D numpy array with data
        """
        # Normalize name to table name
        table_name = f"resampled_{dataset_name.replace('-', '_')}"
        return self._read_from_table(table_name, bounds)
    
    def read_data_chunk(self, table_name: str, 
                       row_range: Tuple[int, int],
                       col_range: Tuple[int, int]) -> np.ndarray:
        """
        Read a spatial chunk from any data table.
        
        Args:
            table_name: Full table name
            row_range: (min_row, max_row) inclusive
            col_range: (min_col, max_col) inclusive
            
        Returns:
            2D numpy array with chunk data
        """
        with self.db.get_cursor() as cursor:
            # Use parameterized query for safety
            query = """
                SELECT row_idx, col_idx, value
                FROM %(table)s
                WHERE row_idx >= %(min_row)s AND row_idx <= %(max_row)s
                  AND col_idx >= %(min_col)s AND col_idx <= %(max_col)s
                  AND value IS NOT NULL
                ORDER BY row_idx, col_idx
            """
            
            cursor.execute(query, {
                'table': table_name,
                'min_row': row_range[0],
                'max_row': row_range[1],
                'min_col': col_range[0],
                'max_col': col_range[1]
            })
            
            # Create chunk array
            height = row_range[1] - row_range[0] + 1
            width = col_range[1] - col_range[0] + 1
            chunk = np.full((height, width), np.nan, dtype=np.float32)
            
            # Fill with data
            for row in cursor:
                r = row['row_idx'] - row_range[0]
                c = row['col_idx'] - col_range[0]
                chunk[r, c] = row['value']
                
        return chunk
    
    def _read_from_table(self, table_name: str,
                        bounds: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Internal method to read from any table.
        
        Args:
            table_name: Full table name
            bounds: Optional geographic bounds (not implemented yet)
            
        Returns:
            2D numpy array with data
        """
        with self.db.get_cursor() as cursor:
            # Get dimensions
            cursor.execute(f"""
                SELECT MIN(row_idx), MAX(row_idx), 
                       MIN(col_idx), MAX(col_idx)
                FROM {table_name}
            """)
            result = cursor.fetchone()
            
            if not result or result[0] is None:
                logger.warning(f"No data found in table {table_name}")
                return np.array([])
            
            min_row, max_row, min_col, max_col = result
            
            # Create empty array
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            data = np.full((height, width), np.nan, dtype=np.float32)
            
            # Fill with data
            cursor.execute(f"""
                SELECT row_idx, col_idx, value
                FROM {table_name}
                WHERE value IS NOT NULL
            """)
            
            for row in cursor:
                r = row['row_idx'] - min_row
                c = row['col_idx'] - min_col
                data[r, c] = row['value']
                
        logger.info(f"Loaded {table_name}: shape={data.shape}, "
                   f"valid_cells={np.sum(~np.isnan(data))}")
        
        return data
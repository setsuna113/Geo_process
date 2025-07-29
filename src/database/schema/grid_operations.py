# src/database/schema/grid_operations.py
"""Grid and cell operations for spatial grid systems."""

import logging
import json
from typing import Dict, Any, List, Optional
from ..connection import db
from ..exceptions import handle_database_error, safe_fetch_id, DatabaseNotFoundError
from src.config import config

logger = logging.getLogger(__name__)


class GridOperations:
    """Handles grid definition and cell operations."""
    
    def __init__(self, db_manager):
        """Initialize with database manager."""
        self.db = db_manager
    
    @handle_database_error("store_grid_definition")
    def store_grid_definition(self, name: str, grid_type: str, resolution: int,
                             bounds: Optional[str] = None, 
                             metadata: Optional[Dict] = None) -> str:
        """Store grid definition metadata."""
        # Get CRS from config with fallback
        grid_config = config.get(f'grids.{grid_type}')
        if not grid_config:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        crs = grid_config.get('crs', 'EPSG:4326')  # Default fallback
        
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO grids (name, grid_type, resolution, crs, bounds, metadata)
                VALUES (%(name)s, %(grid_type)s, %(resolution)s, %(crs)s, 
                        ST_GeomFromText(%(bounds)s, 4326), %(metadata)s)
                RETURNING id
            """, {
                'name': name,
                'grid_type': grid_type,
                'resolution': resolution,
                'crs': crs,
                'bounds': bounds,
                'metadata': json.dumps(metadata or {})
            })
            grid_id = safe_fetch_id(cursor, "store_grid_definition")
            logger.info(f"✅ Created grid '{name}' with ID: {grid_id}")
            return grid_id
    
    @handle_database_error("store_grid_cells_batch")
    def store_grid_cells_batch(self, grid_id: str, cells_data: List[Dict]) -> int:
        """Bulk insert grid cells."""
        if not cells_data:
            return 0
            
        with self.db.get_cursor() as cursor:
            # Prepare data for bulk insert
            cell_records = []
            for cell in cells_data:
                cell_records.append((
                    grid_id,
                    cell['cell_id'],
                    cell['geometry_wkt'],  # WKT format
                    cell.get('area_km2'),
                    cell.get('centroid_wkt')
                ))
                
            cursor.executemany("""
                INSERT INTO grid_cells (grid_id, cell_id, geometry, area_km2, centroid)
                VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, ST_GeomFromText(%s, 4326))
                ON CONFLICT (grid_id, cell_id) DO NOTHING
            """, cell_records)
            
            inserted_count = cursor.rowcount
            logger.info(f"✅ Inserted {inserted_count} grid cells for grid {grid_id}")
            return inserted_count
    
    @handle_database_error("get_grid_by_name")
    def get_grid_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get grid definition by name."""
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM grids WHERE name = %s", (name,))
            return cursor.fetchone()
    
    @handle_database_error("get_grid_cells")
    def get_grid_cells(self, grid_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get grid cells for a grid."""
        with self.db.get_cursor() as cursor:
            query = """
                SELECT cell_id, ST_AsText(geometry) as geometry_wkt, 
                       area_km2, ST_AsText(centroid) as centroid_wkt
                FROM grid_cells 
                WHERE grid_id = %s
                ORDER BY cell_id
            """
            params = [grid_id]
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    @handle_database_error("delete_grid")
    def delete_grid(self, name: str) -> bool:
        """Delete grid and all related data."""
        with self.db.get_cursor() as cursor:
            # First get grid ID
            cursor.execute("SELECT id FROM grids WHERE name = %s", (name,))
            grid = cursor.fetchone()
            if not grid:
                logger.warning(f"Grid {name} not found for deletion")
                return False
            
            grid_id = grid['id']
            
            # Delete in order: cells, then grid
            cursor.execute("DELETE FROM grid_cells WHERE grid_id = %s", (grid_id,))
            cursor.execute("DELETE FROM grids WHERE id = %s", (grid_id,))
            
            logger.info(f"✅ Deleted grid {name} and all associated data")
            return True
    
    def validate_grid_config(self, grid_type: str, resolution: int) -> bool:
        """Validate grid configuration parameters."""
        # Check if grid type is supported
        grid_config = config.get(f'grids.{grid_type}')
        if not grid_config:
            logger.error(f"Invalid grid type: {grid_type}")
            return False
        
        # Check if resolution is within allowed range
        allowed_resolutions = grid_config.get('allowed_resolutions', [])
        if allowed_resolutions and resolution not in allowed_resolutions:
            logger.error(f"Invalid resolution: {resolution}")  
            return False
        
        # Check for reasonable resolution limits
        if resolution <= 0:
            logger.error(f"Invalid resolution: {resolution}")
            return False
        elif resolution < 100 or resolution > 1000000:
            logger.warning(f"Resolution {resolution}m may be impractical")
            
        return True
    
    def get_grid_status(self, grid_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get grid processing status."""
        with self.db.get_cursor() as cursor:
            if grid_name:
                cursor.execute("""
                    SELECT g.name, g.grid_type, g.resolution,
                           COUNT(gc.cell_id) as cell_count,
                           g.created_at
                    FROM grids g
                    LEFT JOIN grid_cells gc ON g.id = gc.grid_id
                    WHERE g.name = %s
                    GROUP BY g.id, g.name, g.grid_type, g.resolution, g.created_at
                """, (grid_name,))
            else:
                cursor.execute("""
                    SELECT g.name, g.grid_type, g.resolution,
                           COUNT(gc.cell_id) as cell_count,
                           g.created_at
                    FROM grids g
                    LEFT JOIN grid_cells gc ON g.id = gc.grid_id
                    GROUP BY g.id, g.name, g.grid_type, g.resolution, g.created_at
                    ORDER BY g.created_at DESC
                """)
            return cursor.fetchall()
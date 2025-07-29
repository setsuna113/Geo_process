"""Raster processing and caching operations for database schema."""

from typing import Dict, Any, List, Optional
import json
import logging
from ..connection import db

logger = logging.getLogger(__name__)


class RasterCache:
    """Raster processing and caching database operations."""
    
    def store_resampled_dataset(self, name: str, source_path: str, target_resolution: float,
                               target_crs: str, bounds: List[float], shape: tuple,
                               data_type: str, resampling_method: str, band_name: str,
                               data_table_name: str, metadata: Dict) -> int:
        """Store resampled dataset metadata."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO resampled_datasets 
                (name, source_path, target_resolution, target_crs, bounds, 
                 shape_height, shape_width, data_type, resampling_method, 
                 band_name, data_table_name, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                name, source_path, target_resolution, target_crs, bounds,
                shape[0], shape[1], data_type, resampling_method, 
                band_name, data_table_name, json.dumps(metadata)
            ))
            dataset_id = cursor.fetchone()['id']
            logger.info(f"✅ Stored resampled dataset '{name}': {dataset_id}")
            return dataset_id
    
    def get_resampled_datasets(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve resampled datasets with optional filters."""
        with db.get_cursor() as cursor:
            query = """
                SELECT id, name, source_path, target_resolution, target_crs, 
                       bounds, shape_height, shape_width, data_type, 
                       resampling_method, band_name, data_table_name, 
                       metadata, created_at
                FROM resampled_datasets
            """
            params = []
            
            if filters:
                conditions = []
                if 'name' in filters:
                    conditions.append("name = %s")
                    params.append(filters['name'])
                if 'data_type' in filters:
                    conditions.append("data_type = %s")
                    params.append(filters['data_type'])
                if 'target_resolution' in filters:
                    conditions.append("target_resolution = %s")
                    params.append(filters['target_resolution'])
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def create_resampled_data_table(self, table_name: str):
        """Create table for storing resampled dataset values."""
        with db.get_cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    row_idx INTEGER NOT NULL,
                    col_idx INTEGER NOT NULL,
                    value FLOAT,
                    PRIMARY KEY (row_idx, col_idx)
                )
            """)
            logger.info(f"✅ Created resampled data table: {table_name}")
    
    def drop_resampled_data_table(self, table_name: str):
        """Drop resampled data table."""
        with db.get_cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            logger.info(f"Dropped resampled data table: {table_name}")
    
    def store_raster_source(self, raster_data: Dict[str, Any]) -> str:
        """Store raster source metadata."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO raster_sources 
                (name, file_path, data_type, pixel_size_degrees, spatial_extent, 
                 nodata_value, band_count, file_size_mb, checksum, last_modified,
                 source_dataset, variable_name, units, description, temporal_info, metadata)
                VALUES (%(name)s, %(file_path)s, %(data_type)s, %(pixel_size_degrees)s, 
                        ST_GeomFromText(%(spatial_extent_wkt)s, 4326), %(nodata_value)s,
                        %(band_count)s, %(file_size_mb)s, %(checksum)s, %(last_modified)s,
                        %(source_dataset)s, %(variable_name)s, %(units)s, %(description)s,
                        %(temporal_info)s, %(metadata)s)
                RETURNING id
            """, {
                **raster_data,
                'temporal_info': json.dumps(raster_data.get('temporal_info', {})),
                'metadata': json.dumps(raster_data.get('metadata', {}))
            })
            raster_id = cursor.fetchone()['id']
            logger.info(f"✅ Stored raster source '{raster_data['name']}': {raster_id}")
            return raster_id

    def get_raster_sources(self, active_only: bool = True, 
                          processing_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get raster sources with optional filtering."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM raster_sources WHERE 1=1"
            params = []
            
            if active_only:
                query += " AND active = true"
            
            if processing_status:
                query += " AND processing_status = %s"
                params.append(processing_status)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()

    def update_raster_processing_status(self, raster_id: str, status: str, 
                                       metadata: Optional[Dict] = None):
        """Update raster processing status."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                UPDATE raster_sources 
                SET processing_status = %s, updated_at = CURRENT_TIMESTAMP,
                    metadata = COALESCE(metadata, '{}') || %s
                WHERE id = %s
            """, (status, json.dumps(metadata or {}), raster_id))

    def store_raster_tiles_batch(self, raster_id: str, tiles_data: List[Dict]) -> int:
        """Bulk store raster tiles."""
        with db.get_cursor() as cursor:
            tile_records = []
            for tile in tiles_data:
                tile_records.append((
                    raster_id,
                    tile['tile_x'],
                    tile['tile_y'],
                    tile['tile_size_pixels'],
                    tile['tile_bounds_wkt'],
                    tile.get('file_byte_offset'),
                    tile.get('file_byte_length'),
                    json.dumps(tile.get('tile_stats', {}))
                ))
            
            cursor.executemany("""
                INSERT INTO raster_tiles 
                (raster_source_id, tile_x, tile_y, tile_size_pixels, tile_bounds,
                 file_byte_offset, file_byte_length, tile_stats)
                VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s)
                ON CONFLICT (raster_source_id, tile_x, tile_y)
                DO UPDATE SET
                    tile_size_pixels = EXCLUDED.tile_size_pixels,
                    tile_bounds = EXCLUDED.tile_bounds,
                    file_byte_offset = EXCLUDED.file_byte_offset,
                    file_byte_length = EXCLUDED.file_byte_length,
                    tile_stats = EXCLUDED.tile_stats
            """, tile_records)
            
            return cursor.rowcount

    def get_raster_tiles_for_bounds(self, raster_id: str, bounds_wkt: str) -> List[Dict[str, Any]]:
        """Get raster tiles that intersect with given bounds."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM raster_tiles 
                WHERE raster_source_id = %s 
                AND ST_Intersects(tile_bounds, ST_GeomFromText(%s, 4326))
                ORDER BY tile_y, tile_x
            """, (raster_id, bounds_wkt))
            return cursor.fetchall()

    def store_resampling_cache_batch(self, cache_data: List[Dict]) -> int:
        """Bulk store resampling cache entries."""
        with db.get_cursor() as cursor:
            cache_records = []
            for cache in cache_data:
                cache_records.append((
                    cache['source_raster_id'],
                    cache['target_grid_id'],
                    cache['cell_id'],
                    cache['method'],
                    cache['band_number'],
                    cache['value'],
                    cache.get('confidence_score', 1.0),
                    cache.get('source_tiles_used', []),
                    json.dumps(cache.get('computation_metadata', {}))
                ))
            
            cursor.executemany("""
                INSERT INTO resampling_cache 
                (source_raster_id, target_grid_id, cell_id, method, band_number,
                 value, confidence_score, source_tiles_used, computation_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_raster_id, target_grid_id, cell_id, method, band_number)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence_score = EXCLUDED.confidence_score,
                    source_tiles_used = EXCLUDED.source_tiles_used,
                    computation_metadata = EXCLUDED.computation_metadata,
                    last_accessed = CURRENT_TIMESTAMP,
                    access_count = resampling_cache.access_count + 1
            """, cache_records)
            
            return cursor.rowcount

    def get_cached_resampling_values(self, raster_id: str, grid_id: str, 
                                   cell_ids: List[str], method: str, 
                                   band_number: int) -> Dict[str, float]:
        """Get cached resampling values for specific cells."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                UPDATE resampling_cache 
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE source_raster_id = %s AND target_grid_id = %s 
                AND cell_id = ANY(%s) AND method = %s AND band_number = %s
                RETURNING cell_id, value
            """, (raster_id, grid_id, cell_ids, method, band_number))
            
            results = cursor.fetchall()
            return {row['cell_id']: row['value'] for row in results}

    def add_processing_task(self, queue_type: str, raster_id: Optional[str] = None,
                           grid_id: Optional[str] = None, tile_id: Optional[str] = None,
                           parameters: Optional[Dict] = None, priority: int = 0) -> str:
        """Add a task to the processing queue."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO processing_queue 
                (queue_type, raster_source_id, grid_id, tile_id, parameters, priority)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (queue_type, raster_id, grid_id, tile_id, 
                  json.dumps(parameters or {}), priority))
            task_id = cursor.fetchone()['id']
            logger.info(f"✅ Added {queue_type} task: {task_id}")
            return task_id

    def get_next_processing_task(self, queue_type: str, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get next processing task with worker assignment."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                UPDATE processing_queue 
                SET status = 'processing', worker_id = %s, started_at = CURRENT_TIMESTAMP
                WHERE id = (
                    SELECT id FROM processing_queue 
                    WHERE queue_type = %s AND status = 'pending'
                    AND retry_count < max_retries
                    ORDER BY priority DESC, created_at ASC 
                    LIMIT 1 FOR UPDATE SKIP LOCKED
                )
                RETURNING *
            """, (worker_id, queue_type))
            return cursor.fetchone()

    def complete_processing_task(self, task_id: str, success: bool = True, 
                               error_message: Optional[str] = None):
        """Mark processing task as completed or failed."""
        with db.get_cursor() as cursor:
            if success:
                cursor.execute("""
                    UPDATE processing_queue 
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (task_id,))
            else:
                cursor.execute("""
                    UPDATE processing_queue 
                    SET status = 'failed', error_message = %s, retry_count = retry_count + 1,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (error_message, task_id))
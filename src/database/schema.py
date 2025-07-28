"""Database schema operations and data access layer."""

from pathlib import Path
from .connection import DatabaseManager, db
from src.config import config
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import json
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class DatabaseSchema:
    """Database schema operations and data access interface."""
    
    def __init__(self):
        self.schema_file = Path(__file__).parent / "schema.sql"
    
    # Schema Management
    def create_schema(self) -> bool:
        """Create database schema from SQL file."""
        try:
            db.execute_sql_file(self.schema_file)
            logger.info("✅ Database schema created successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            return False
    
    def drop_schema(self, confirm: bool = False) -> bool:
        """Drop all tables (use with extreme caution!)."""
        if not confirm:
            raise ValueError("Must set confirm=True to drop schema")
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute("""
                    DROP VIEW IF EXISTS experiment_summary CASCADE;
                    DROP VIEW IF EXISTS grid_processing_status CASCADE;
                    DROP VIEW IF EXISTS species_richness_summary CASCADE;
                    DROP FUNCTION IF EXISTS update_grid_cell_count() CASCADE;
                    DROP TABLE IF EXISTS processing_jobs CASCADE;
                    DROP TABLE IF EXISTS experiments CASCADE;
                    DROP TABLE IF EXISTS climate_data CASCADE;
                    DROP TABLE IF EXISTS features CASCADE;
                    DROP TABLE IF EXISTS species_grid_intersections CASCADE;
                    DROP TABLE IF EXISTS species_ranges CASCADE;
                    DROP TABLE IF EXISTS grid_cells CASCADE;
                    DROP TABLE IF EXISTS grids CASCADE;
                    -- Drop resampled dataset tables
                    DROP TABLE IF EXISTS resampled_datasets CASCADE;
                """)
            logger.warning("⚠️ Database schema dropped")
            return True
        except Exception as e:
            logger.error(f"❌ Schema drop failed: {e}")
            return False
    
    # Grid Operations (for grid_systems/ modules)
    def store_grid_definition(self, name: str, grid_type: str, resolution: int,
                             bounds: Optional[str] = None, 
                             metadata: Optional[Dict] = None) -> str:
        """Store grid definition metadata."""
        # Get CRS from config with fallback
        grid_config = config.get(f'grids.{grid_type}')
        if not grid_config:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        crs = grid_config.get('crs', 'EPSG:4326')  # Default fallback
        
        with db.get_cursor() as cursor:
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
            grid_id = cursor.fetchone()['id']
            logger.info(f"✅ Created grid '{name}' with ID: {grid_id}")
            return grid_id
    
    def store_grid_cells_batch(self, grid_id: str, cells_data: List[Dict]) -> int:
        """Bulk insert grid cells."""
        with db.get_cursor() as cursor:
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

            # Bulk insert with conflict resolution
            cursor.executemany("""
                INSERT INTO grid_cells (grid_id, cell_id, geometry, area_km2, centroid)
                VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, ST_GeomFromText(%s, 4326))
                ON CONFLICT (grid_id, cell_id) DO NOTHING
            """, cell_records)
            
            inserted_count = cursor.rowcount
            logger.info(f"✅ Inserted {inserted_count} grid cells for grid {grid_id}")
            return inserted_count
    
    def get_grid_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get grid by name."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM grids WHERE name = %s", (name,))
            return cursor.fetchone()
    
    def get_grid_cells(self, grid_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get grid cells for a grid."""
        with db.get_cursor() as cursor:
            query = """
                SELECT cell_id, ST_AsText(geometry) as geometry_wkt, 
                       area_km2, ST_AsText(centroid) as centroid_wkt
                FROM grid_cells 
                WHERE grid_id = %s
                ORDER BY cell_id
            """
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (grid_id,))
            return cursor.fetchall()
    
    def delete_grid(self, name: str) -> bool:
        """Delete grid and all related data."""
        with db.get_cursor() as cursor:
            cursor.execute("DELETE FROM grids WHERE name = %s", (name,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"✅ Deleted grid: {name}")
            else:
                logger.warning(f"⚠️ Grid not found: {name}")
            return deleted
    
    # Species Operations (for species/ and processors/ modules)
    def store_species_range(self, species_data: Dict[str, Any]) -> str:
        """Store species range data from .gpkg file."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO species_ranges 
                (species_name, scientific_name, genus, family, order_name, class_name, 
                 phylum, kingdom, category, range_type, geometry, source_file, 
                 source_dataset, confidence, area_km2, metadata)
                VALUES (%(species_name)s, %(scientific_name)s, %(genus)s, %(family)s,
                        %(order_name)s, %(class_name)s, %(phylum)s, %(kingdom)s,
                        %(category)s, %(range_type)s, ST_GeomFromText(%(geometry_wkt)s, 4326),
                        %(source_file)s, %(source_dataset)s, %(confidence)s, 
                        %(area_km2)s, %(metadata)s)
                RETURNING id
            """, {
                'species_name': species_data['species_name'],
                'scientific_name': species_data.get('scientific_name', ''),
                'genus': species_data.get('genus', ''),
                'family': species_data.get('family', ''),
                'order_name': species_data.get('order_name', ''),
                'class_name': species_data.get('class_name', ''),
                'phylum': species_data.get('phylum', ''),
                'kingdom': species_data.get('kingdom', ''),
                'category': species_data.get('category', 'unknown'),
                'range_type': species_data.get('range_type', 'distribution'),
                'geometry_wkt': species_data['geometry_wkt'],
                'source_file': species_data['source_file'],
                'source_dataset': species_data.get('source_dataset', ''),
                'confidence': species_data.get('confidence', 1.0),
                'area_km2': species_data.get('area_km2'),
                'metadata': json.dumps(species_data.get('metadata', {}))
            })
            range_id = cursor.fetchone()['id']
            logger.debug(f"✅ Stored species range: {species_data['species_name']} ({range_id})")
            return range_id
    
    def store_species_intersections_batch(self, intersections: List[Dict]) -> int:
        """Bulk store species-grid intersections."""
        with db.get_cursor() as cursor:
            intersection_records = []
            for inter in intersections:
                intersection_records.append((
                    inter['grid_id'],
                    inter['cell_id'],
                    inter['species_range_id'],
                    inter['species_name'],
                    inter['category'],
                    inter['range_type'],
                    inter.get('intersection_area_km2'),
                    inter.get('coverage_percent'),
                    inter.get('presence_score', 1.0),
                    json.dumps(inter.get('computation_metadata', {}))
                ))
            
            cursor.executemany("""
                INSERT INTO species_grid_intersections 
                (grid_id, cell_id, species_range_id, species_name, category, range_type,
                 intersection_area_km2, coverage_percent, presence_score, computation_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, species_range_id) 
                DO UPDATE SET
                    intersection_area_km2 = EXCLUDED.intersection_area_km2,
                    coverage_percent = EXCLUDED.coverage_percent,
                    presence_score = EXCLUDED.presence_score,
                    computation_metadata = EXCLUDED.computation_metadata,
                    computed_at = CURRENT_TIMESTAMP
            """, intersection_records)
            
            inserted_count = cursor.rowcount
            logger.info(f"✅ Stored {inserted_count} species-grid intersections")
            return inserted_count
    
    def get_species_ranges(self, category: Optional[str] = None, 
                          source_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get species ranges with optional filtering."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM species_ranges WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = %s"
                params.append(category)
            
            if source_file:
                query += " AND source_file = %s"
                params.append(source_file)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    # Feature Operations (for features/ modules)
    def store_feature(self, grid_id: str, cell_id: str, feature_type: str,
                     feature_name: str, feature_value: float,
                     metadata: Optional[Dict] = None) -> str:
        """Store computed feature."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO features 
                (grid_id, cell_id, feature_type, feature_name, feature_value, computation_metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, feature_type, feature_name)
                DO UPDATE SET
                    feature_value = EXCLUDED.feature_value,
                    computation_metadata = EXCLUDED.computation_metadata,
                    computed_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (grid_id, cell_id, feature_type, feature_name, feature_value,
                  json.dumps(metadata or {})))
            return cursor.fetchone()['id']
    
    def store_features_batch(self, features: List[Dict]) -> int:
        """Bulk store features."""
        with db.get_cursor() as cursor:
            feature_records = []
            for feat in features:
                feature_records.append((
                    feat['grid_id'],
                    feat['cell_id'],
                    feat['feature_type'],
                    feat['feature_name'],
                    feat['feature_value'],
                    json.dumps(feat.get('computation_metadata', {}))
                ))
            
            cursor.executemany("""
                INSERT INTO features 
                (grid_id, cell_id, feature_type, feature_name, feature_value, computation_metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, feature_type, feature_name)
                DO UPDATE SET
                    feature_value = EXCLUDED.feature_value,
                    computation_metadata = EXCLUDED.computation_metadata,
                    computed_at = CURRENT_TIMESTAMP
            """, feature_records)
            
            return cursor.rowcount
    
    def store_climate_data_batch(self, climate_data: List[Dict]) -> int:
        """Bulk store climate data."""
        with db.get_cursor() as cursor:
            climate_records = []
            for data in climate_data:
                climate_records.append((
                    data['grid_id'],
                    data['cell_id'],
                    data['variable'],
                    data['value'],
                    data.get('source'),
                    data.get('resolution')
                ))
            
            cursor.executemany("""
                INSERT INTO climate_data 
                (grid_id, cell_id, variable, value, source, resolution)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, variable, source, resolution)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    extracted_at = CURRENT_TIMESTAMP
            """, climate_records)
            
            return cursor.rowcount
    
    def get_features(self, grid_id: str, feature_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get features for a grid."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM features WHERE grid_id = %s"
            params = [grid_id]
            
            if feature_type:
                query += " AND feature_type = %s"
                params.append(feature_type)
            
            query += " ORDER BY cell_id, feature_type, feature_name"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    # Resampled Dataset Operations (for resampling pipeline)
    def store_resampled_dataset(self, name: str, source_path: str, target_resolution: float,
                               target_crs: str, bounds: List[float], shape: Tuple[int, int],
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
    
    def get_passthrough_datasets(self, target_resolution: float, tolerance: float = 0.001) -> List[Dict]:
        """
        Query datasets marked as passthrough for given resolution.
        
        Args:
            target_resolution: Target resolution to match
            tolerance: Tolerance for resolution comparison
            
        Returns:
            List of passthrough dataset info
        """
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, name, source_path, target_resolution, target_crs, 
                       bounds, shape_height, shape_width, data_type, 
                       resampling_method, band_name, data_table_name, 
                       metadata, created_at
                FROM resampled_datasets
                WHERE metadata->>'passthrough' = 'true'
                  AND ABS(target_resolution - %s) <= %s
                ORDER BY created_at DESC
            """, (target_resolution, tolerance))
            
            results = cursor.fetchall()
            logger.info(f"Found {len(results)} passthrough datasets matching resolution {target_resolution}±{tolerance}")
            return results
    
    def get_dataset_by_type(self, passthrough_only: bool = False, resampled_only: bool = False) -> List[Dict]:
        """
        Get datasets filtered by type (passthrough vs resampled).
        
        Args:
            passthrough_only: Return only passthrough datasets
            resampled_only: Return only resampled datasets
            
        Returns:
            List of filtered dataset info
        """
        if passthrough_only and resampled_only:
            raise ValueError("Cannot specify both passthrough_only and resampled_only")
        
        with db.get_cursor() as cursor:
            query = """
                SELECT id, name, source_path, target_resolution, target_crs, 
                       bounds, shape_height, shape_width, data_type, 
                       resampling_method, band_name, data_table_name, 
                       metadata, created_at
                FROM resampled_datasets
            """
            
            if passthrough_only:
                query += " WHERE metadata->>'passthrough' = 'true'"
            elif resampled_only:
                query += " WHERE COALESCE(metadata->>'passthrough', 'false') = 'false'"
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query)
            
            results = cursor.fetchall()
            dataset_type = "passthrough" if passthrough_only else "resampled" if resampled_only else "all"
            logger.info(f"Found {len(results)} {dataset_type} datasets")
            return results
    
    def update_dataset_metadata(self, dataset_name: str, additional_metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            additional_metadata: Additional metadata to merge
            
        Returns:
            True if updated successfully
        """
        try:
            with db.get_cursor() as cursor:
                # Get existing metadata
                cursor.execute("""
                    SELECT metadata FROM resampled_datasets WHERE name = %s
                """, (dataset_name,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Dataset {dataset_name} not found for metadata update")
                    return False
                
                existing_metadata = result['metadata'] or {}
                
                # Merge metadata
                updated_metadata = {**existing_metadata, **additional_metadata}
                
                # Update in database
                cursor.execute("""
                    UPDATE resampled_datasets 
                    SET metadata = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE name = %s
                """, (json.dumps(updated_metadata), dataset_name))
                
                logger.info(f"✅ Updated metadata for dataset: {dataset_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update metadata for {dataset_name}: {e}")
            return False
    
    # Experiment and Job Tracking (for pipeline/ modules)
    def create_experiment(self, name: str, description: str, config: Dict) -> str:
        """Create new experiment, handling duplicates gracefully."""
        with db.get_cursor() as cursor:
            try:
                cursor.execute("""
                    INSERT INTO experiments (name, description, config, created_by)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (name, description, json.dumps(config), config.get('created_by', 'system')))
                experiment_id = cursor.fetchone()['id']
                logger.info(f"✅ Created experiment '{name}': {experiment_id}")
                return experiment_id
            except Exception as e:
                if 'duplicate key' in str(e).lower() or 'unique constraint' in str(e).lower():
                    # Handle duplicate experiment name by appending timestamp
                    import time
                    timestamp = int(time.time())
                    new_name = f"{name}_{timestamp}"
                    logger.warning(f"Experiment '{name}' exists, creating as '{new_name}'")
                    cursor.execute("""
                        INSERT INTO experiments (name, description, config, created_by)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (new_name, description, json.dumps(config), config.get('created_by', 'system')))
                    experiment_id = cursor.fetchone()['id']
                    logger.info(f"✅ Created experiment '{new_name}': {experiment_id}")
                    return experiment_id
                else:
                    raise
    
    def update_experiment_status(self, experiment_id: str, status: str, 
                                results: Optional[Dict] = None, 
                                error_message: Optional[str] = None):
        """Update experiment status."""
        with db.get_cursor() as cursor:
            completed_at = datetime.now() if status == 'completed' else None
            cursor.execute("""
                UPDATE experiments 
                SET status = %s, completed_at = %s, results = %s, error_message = %s
                WHERE id = %s
            """, (status, completed_at, json.dumps(results or {}), error_message, experiment_id))
    
    def create_processing_job(self, job_type: str, job_name: str, parameters: Dict,
                             parent_experiment_id: Optional[str] = None) -> str:
        """Create processing job."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO processing_jobs (job_type, job_name, parameters, parent_experiment_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (job_type, job_name, json.dumps(parameters), parent_experiment_id))
            job_id = cursor.fetchone()['id']
            logger.info(f"✅ Created job '{job_name}' ({job_type}): {job_id}")
            return job_id
    
    def update_job_progress(self, job_id: str, progress_percent: float, 
                           status: Optional[str] = None, log_message: Optional[str] = None):
        """Update job progress."""
        with db.get_cursor() as cursor:
            updates = ["progress_percent = %s"]
            params: List[Any] = [progress_percent]
            
            if status:
                updates.append("status = %s")
                params.append(status)
                if status == 'running' and progress_percent == 0:
                    updates.append("started_at = CURRENT_TIMESTAMP")
                elif status in ['completed', 'failed']:
                    updates.append("completed_at = CURRENT_TIMESTAMP")
            
            if log_message:
                updates.append("log_messages = array_append(log_messages, %s)")
                params.append(log_message)
            
            params.append(job_id)
            
            cursor.execute(f"""
                UPDATE processing_jobs 
                SET {', '.join(updates)}
                WHERE id = %s
            """, params)
    
    # Utility and Query Methods
    def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information."""
        with db.get_cursor() as cursor:
            # Table info
            cursor.execute("""
                SELECT 
                    t.table_name,
                    (SELECT COUNT(*) FROM information_schema.columns 
                     WHERE table_name = t.table_name AND table_schema = 'public') as column_count,
                    pg_size_pretty(pg_total_relation_size(quote_ident(t.table_name))) as size
                FROM information_schema.tables t
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY t.table_name;
            """)
            tables = cursor.fetchall()
            
            # View info
            cursor.execute("""
                SELECT table_name as view_name
                FROM information_schema.views
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            views = cursor.fetchall()
            
            # Row counts
            table_counts = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table['table_name']}")
                table_counts[table['table_name']] = cursor.fetchone()['count']
            
            return {
                'tables': tables,
                'views': views,
                'table_counts': table_counts,
                'summary': {
                    'table_count': len(tables),
                    'view_count': len(views),
                    'total_rows': sum(table_counts.values())
                }
            }
    
    def get_grid_status(self, grid_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get grid processing status."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM grid_processing_status"
            params = []
            
            if grid_name:
                query += " WHERE grid_name = %s"
                params.append(grid_name)
            
            query += " ORDER BY grid_name"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_species_richness(self, grid_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get species richness summary."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM species_richness_summary WHERE grid_id = %s"
            params = [grid_id]
            
            if category:
                query += " AND category = %s"
                params.append(category)
            
            query += " ORDER BY cell_id, category"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def validate_grid_config(self, grid_type: str, resolution: int) -> bool:
        """Validate grid configuration against defaults."""
        grid_config = config.get(f'grids.{grid_type}')
        if not grid_config:
            return False
        
        valid_resolutions = grid_config.get('resolutions', [])
        return resolution in valid_resolutions

    # Raster Operations (for raster/ modules)
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
                    SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                        retry_count = retry_count + 1, error_message = %s
                    WHERE id = %s
                """, (error_message, task_id))

    def get_raster_processing_status(self, raster_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get raster processing status overview."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM raster_processing_status"
            params = []
            
            if raster_id:
                query += " WHERE raster_id = %s"
                params.append(raster_id)
            
            query += " ORDER BY raster_name"
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_cache_efficiency_summary(self, raster_id: Optional[str] = None, 
                                   grid_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cache efficiency statistics."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM cache_efficiency_summary WHERE 1=1"
            params = []
            
            if raster_id:
                query += " AND source_raster_id = %s"
                params.append(raster_id)
            
            if grid_id:
                query += " AND target_grid_id = %s"
                params.append(grid_id)
            
            query += " ORDER BY raster_name, grid_name, method"
            cursor.execute(query, params)
            return cursor.fetchall()

    def cleanup_old_cache(self, days_old: int = 30, min_access_count: int = 1) -> int:
        """Clean up old and rarely accessed cache entries."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT cleanup_resampling_cache(%s, %s)", 
                          (days_old, min_access_count))
            return cursor.fetchone()['cleanup_resampling_cache']

    def get_processing_queue_summary(self) -> List[Dict[str, Any]]:
        """Get processing queue statistics."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM processing_queue_summary ORDER BY queue_type, status")
            return cursor.fetchall()

    def create_all_tables(self):
        """Create all database tables."""
        return self.create_schema()

    def drop_all_tables(self):
        """Drop all database tables."""
        return self.drop_schema(confirm=True)
    
    # Test Mode Operations
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
        import json
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
    
    # Checkpoint Operations
    # LEGACY CHECKPOINT METHODS - Consider using the unified checkpoint system (src.checkpoints) instead
    # These methods are kept for backward compatibility and low-level database operations
    
    def create_checkpoint(self, checkpoint_id: str, level: str, parent_id: Optional[str],
                         processor_name: str, data_summary: Dict[str, Any],
                         file_path: str, file_size_bytes: int,
                         compression_type: Optional[str] = None) -> str:
        """Create a checkpoint record."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pipeline_checkpoints 
                (checkpoint_id, level, parent_id, processor_name, data_summary,
                 file_path, file_size_bytes, compression_type, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'created')
                RETURNING id
            """, (
                checkpoint_id, level, parent_id, processor_name, 
                json.dumps(data_summary), file_path, file_size_bytes, 
                compression_type
            ))
            return cursor.fetchone()['id']
    
    def update_checkpoint_status(self, checkpoint_id: str, status: str,
                               validation_checksum: Optional[str] = None,
                               validation_result: Optional[Dict] = None,
                               error_message: Optional[str] = None):
        """Update checkpoint status and validation info."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                UPDATE pipeline_checkpoints 
                SET status = %s, validation_checksum = %s, validation_result = %s,
                    error_message = %s, updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = %s
            """, (
                status, validation_checksum, json.dumps(validation_result or {}),
                error_message, checkpoint_id
            ))
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint by ID."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM pipeline_checkpoints 
                WHERE checkpoint_id = %s
            """, (checkpoint_id,))
            return cursor.fetchone()
    
    def get_latest_checkpoint(self, processor_name: str, level: str) -> Optional[Dict[str, Any]]:
        """Get latest checkpoint for a processor at a specific level."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM pipeline_checkpoints 
                WHERE processor_name = %s AND level = %s AND status = 'valid'
                ORDER BY created_at DESC
                LIMIT 1
            """, (processor_name, level))
            return cursor.fetchone()
    
    def list_checkpoints(self, processor_name: Optional[str] = None,
                        level: Optional[str] = None,
                        parent_id: Optional[str] = None,
                        status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List checkpoints with optional filtering."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM pipeline_checkpoints WHERE 1=1"
            params = []
            
            if processor_name:
                query += " AND processor_name = %s"
                params.append(processor_name)
            if level:
                query += " AND level = %s"
                params.append(level)
            if parent_id:
                query += " AND parent_id = %s"
                params.append(parent_id)
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def cleanup_old_checkpoints(self, days_old: int, keep_minimum: Dict[str, int]) -> int:
        """Clean up old checkpoints while preserving minimum counts per level."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                WITH ranked_checkpoints AS (
                    SELECT id, level, created_at,
                           ROW_NUMBER() OVER (PARTITION BY level ORDER BY created_at DESC) as rn
                    FROM pipeline_checkpoints
                    WHERE status = 'valid'
                )
                DELETE FROM pipeline_checkpoints
                WHERE id IN (
                    SELECT id FROM ranked_checkpoints
                    WHERE created_at < NOW() - INTERVAL %s
                    AND (
                        (level = 'pipeline' AND rn > %s) OR
                        (level = 'phase' AND rn > %s) OR
                        (level = 'step' AND rn > %s) OR
                        (level = 'substep' AND rn > %s)
                    )
                )
            """, (
                f'{days_old} days',
                keep_minimum.get('pipeline', 5),
                keep_minimum.get('phase', 3),
                keep_minimum.get('step', 2),
                keep_minimum.get('substep', 1)
            ))
            return cursor.rowcount
    
    # Processing Steps Operations
    def create_processing_step(self, step_name: str, processor_name: str,
                             parent_job_id: str, total_items: int,
                             parameters: Optional[Dict] = None) -> str:
        """Create a processing step record."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO processing_steps 
                (step_name, processor_name, parent_job_id, total_items,
                 parameters, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                RETURNING id
            """, (
                step_name, processor_name, parent_job_id, total_items,
                json.dumps(parameters or {})
            ))
            return cursor.fetchone()['id']
    
    def update_processing_step(self, step_id: str, processed_items: int,
                             failed_items: int = 0, status: Optional[str] = None,
                             error_messages: Optional[List[str]] = None,
                             checkpoint_id: Optional[str] = None):
        """Update processing step progress."""
        with db.get_cursor() as cursor:
            updates = ["processed_items = %s", "failed_items = %s"]
            params = [processed_items, failed_items]
            
            if status:
                updates.append("status = %s")
                params.append(status)
                if status == 'running' and processed_items == 0:
                    updates.append("started_at = CURRENT_TIMESTAMP")
                elif status in ['completed', 'failed']:
                    updates.append("completed_at = CURRENT_TIMESTAMP")
            
            if error_messages:
                updates.append("error_messages = array_cat(error_messages, %s)")
                params.append(error_messages)
            
            if checkpoint_id:
                updates.append("last_checkpoint_id = %s")
                params.append(checkpoint_id)
            
            params.append(step_id)
            
            cursor.execute(f"""
                UPDATE processing_steps 
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, params)
    
    def get_processing_steps(self, parent_job_id: str) -> List[Dict[str, Any]]:
        """Get all processing steps for a job."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM processing_steps 
                WHERE parent_job_id = %s
                ORDER BY created_at
            """, (parent_job_id,))
            return cursor.fetchall()
    
    # File Processing Status Operations
    def create_file_processing_status(self, file_path: str, file_type: str,
                                    file_size_bytes: int, processor_name: str,
                                    parent_job_id: Optional[str] = None) -> str:
        """Create file processing status record."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO file_processing_status 
                (file_path, file_type, file_size_bytes, processor_name,
                 parent_job_id, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                RETURNING id
            """, (
                file_path, file_type, file_size_bytes, processor_name,
                parent_job_id
            ))
            return cursor.fetchone()['id']
    
    def update_file_processing_status(self, file_id: str, status: str,
                                    bytes_processed: Optional[int] = None,
                                    chunks_completed: Optional[int] = None,
                                    total_chunks: Optional[int] = None,
                                    checkpoint_id: Optional[str] = None,
                                    error_message: Optional[str] = None):
        """Update file processing status."""
        with db.get_cursor() as cursor:
            updates = ["status = %s"]
            params = [status]
            
            if bytes_processed is not None:
                updates.append("bytes_processed = %s")
                params.append(bytes_processed)
            
            if chunks_completed is not None:
                updates.append("chunks_completed = %s")
                params.append(chunks_completed)
            
            if total_chunks is not None:
                updates.append("total_chunks = %s")
                params.append(total_chunks)
            
            if checkpoint_id:
                updates.append("last_checkpoint_id = %s")
                params.append(checkpoint_id)
            
            if error_message:
                updates.append("error_message = %s")
                params.append(error_message)
            
            if status == 'processing' and chunks_completed == 0:
                updates.append("started_at = CURRENT_TIMESTAMP")
            elif status in ['completed', 'failed']:
                updates.append("completed_at = CURRENT_TIMESTAMP")
            
            params.append(file_id)
            
            cursor.execute(f"""
                UPDATE file_processing_status 
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, params)
    
    def get_file_processing_status(self, file_path: Optional[str] = None,
                                 parent_job_id: Optional[str] = None,
                                 status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get file processing status with optional filtering."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM file_processing_status WHERE 1=1"
            params = []
            
            if file_path:
                query += " AND file_path = %s"
                params.append(file_path)
            
            if parent_job_id:
                query += " AND parent_job_id = %s"
                params.append(parent_job_id)
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_resumable_files(self, processor_name: str) -> List[Dict[str, Any]]:
        """Get files that can be resumed from checkpoints."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM file_processing_status 
                WHERE processor_name = %s 
                AND status IN ('processing', 'failed')
                AND last_checkpoint_id IS NOT NULL
                ORDER BY updated_at DESC
            """, (processor_name,))
            return cursor.fetchall()

# Global schema instance
schema = DatabaseSchema()
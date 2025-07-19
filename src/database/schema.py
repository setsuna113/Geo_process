"""Database schema operations and data access layer."""

from pathlib import Path
from .connection import db
from ..config import config
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
                    -- Drop raster views
                    DROP VIEW IF EXISTS processing_queue_summary CASCADE;
                    DROP VIEW IF EXISTS cache_efficiency_summary CASCADE;
                    DROP VIEW IF EXISTS raster_processing_status CASCADE;
                    
                    -- Drop existing views
                    DROP VIEW IF EXISTS experiment_summary CASCADE;
                    DROP VIEW IF EXISTS grid_processing_status CASCADE;
                    DROP VIEW IF EXISTS species_richness_summary CASCADE;
                    DROP VIEW IF EXISTS export_summary CASCADE;
                    
                    -- Drop raster functions
                    DROP FUNCTION IF EXISTS get_next_processing_task(VARCHAR, VARCHAR) CASCADE;
                    DROP FUNCTION IF EXISTS update_tile_statistics(UUID, JSONB) CASCADE;
                    DROP FUNCTION IF EXISTS cleanup_resampling_cache(INTEGER, INTEGER) CASCADE;
                    DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
                    
                    -- Drop existing functions
                    DROP FUNCTION IF EXISTS update_grid_cell_count() CASCADE;
                    DROP FUNCTION IF EXISTS update_grid_processing_status(UUID, processing_status_enum, JSONB) CASCADE;
                    DROP FUNCTION IF EXISTS cleanup_expired_exports() CASCADE;
                    
                    -- Drop raster tables
                    DROP TABLE IF EXISTS processing_queue CASCADE;
                    DROP TABLE IF EXISTS resampling_cache CASCADE;
                    DROP TABLE IF EXISTS raster_tiles CASCADE;
                    DROP TABLE IF EXISTS raster_sources CASCADE;
                    
                    -- Drop existing tables
                    DROP TABLE IF EXISTS processing_jobs CASCADE;
                    DROP TABLE IF EXISTS experiments CASCADE;
                    DROP TABLE IF EXISTS climate_data CASCADE;
                    DROP TABLE IF EXISTS features CASCADE;
                    DROP TABLE IF EXISTS species_grid_intersections CASCADE;
                    DROP TABLE IF EXISTS species_ranges CASCADE;
                    DROP TABLE IF EXISTS grid_cells CASCADE;
                    DROP TABLE IF EXISTS grids CASCADE;
                    DROP TABLE IF EXISTS export_metadata CASCADE;
                    
                    -- Drop raster enums
                    DROP TYPE IF EXISTS tile_status_enum CASCADE;
                    DROP TYPE IF EXISTS raster_status_enum CASCADE;
                    
                    -- Drop existing enums
                    DROP TYPE IF EXISTS processing_status_enum CASCADE;
                """)
            logger.warning("⚠️ Database schema dropped")
            return True
        except Exception as e:
            logger.error(f"❌ Schema drop failed: {e}")
            return False
    
    # Grid Operations (for grid_systems/ modules)
    def store_grid_definition(self, name: str, grid_type: str, resolution: int,
                             bounds: Optional[str] = None, 
                             metadata: Optional[Dict] = None,
                             processing_status: str = 'pending') -> str:
        """Store grid definition metadata."""
        # Get CRS from config
        grid_config = config.get(f'grids.{grid_type}')
        if not grid_config:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        crs = grid_config['crs']
        
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO grids (name, grid_type, resolution, crs, bounds, metadata, processing_status)
                VALUES (%(name)s, %(grid_type)s, %(resolution)s, %(crs)s, 
                        ST_GeomFromText(%(bounds)s, 4326), %(metadata)s, %(processing_status)s)
                RETURNING id
            """, {
                'name': name,
                'grid_type': grid_type,
                'resolution': resolution,
                'crs': crs,
                'bounds': bounds,
                'metadata': json.dumps(metadata or {}),
                'processing_status': processing_status
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
            
            # Bulk insert
            cursor.executemany("""
                INSERT INTO grid_cells (grid_id, cell_id, geometry, area_km2, centroid)
                VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, ST_GeomFromText(%s, 4326))
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
    
    # Experiment and Job Tracking (for pipeline/ modules)
    def create_experiment(self, name: str, description: str, config: Dict) -> str:
        """Create new experiment."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO experiments (name, description, config, created_by)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (name, description, json.dumps(config), config.get('created_by', 'system')))
            experiment_id = cursor.fetchone()['id']
            logger.info(f"✅ Created experiment '{name}': {experiment_id}")
            return experiment_id
    
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
                  AND t.table_name NOT IN (\'spatial_ref_sys\', \'geography_columns\', \'geometry_columns\')
                ORDER BY t.table_name;
            """)
            tables = cursor.fetchall()
            
            # View info
            cursor.execute("""
                SELECT table_name as view_name
                FROM information_schema.views
                WHERE table_schema = 'public'
                  AND table_name NOT IN (\'geography_columns\', \'geometry_columns\')
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


    def get_grid_status_fast(self, grid_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Optimized grid status query - avoids expensive view aggregations."""
        with db.get_cursor() as cursor:
            if grid_name:
                # Single grid - use efficient targeted query
                cursor.execute("""
                    SELECT 
                        g.id as grid_id,
                        g.name as grid_name,
                        g.grid_type,
                        g.resolution,
                        g.total_cells,
                        COALESCE((SELECT COUNT(*) FROM grid_cells WHERE grid_id = g.id), 0) as cells_generated,
                        COALESCE((SELECT COUNT(DISTINCT cell_id) FROM species_grid_intersections WHERE grid_id = g.id), 0) as cells_with_species,
                        COALESCE((SELECT COUNT(DISTINCT cell_id) FROM features WHERE grid_id = g.id), 0) as cells_with_features,
                        COALESCE((SELECT COUNT(DISTINCT cell_id) FROM climate_data WHERE grid_id = g.id), 0) as cells_with_climate,
                        ROUND(
                            (((SELECT COUNT(*) FROM grid_cells WHERE grid_id = g.id)::FLOAT / NULLIF(g.total_cells, 0)) * 100)::NUMERIC, 2
                        ) as generation_progress_percent
                    FROM grids g
                    WHERE g.name = %s
                """, (grid_name,))
            else:
                # Multiple grids - return basic info only (fast)
                cursor.execute("""
                    SELECT 
                        id as grid_id,
                        name as grid_name,
                        grid_type,
                        resolution,
                        total_cells,
                        -1 as cells_generated,
                        -1 as cells_with_species,
                        -1 as cells_with_features,
                        -1 as cells_with_climate,
                        -1.0 as generation_progress_percent
                    FROM grids
                    ORDER BY name
                    LIMIT 100
                """)
            return cursor.fetchall()

    # Enhanced Grid Operations with Processing Status
    def update_grid_processing_status(self, grid_id: str, new_status: str, 
                                    metadata: Optional[Dict] = None) -> bool:
        """Update grid processing status with validation."""
        with db.get_cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT update_grid_processing_status(%s, %s, %s)
                """, (grid_id, new_status, json.dumps(metadata) if metadata else None))
                return cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"❌ Failed to update grid status: {e}")
                return False
    
    def get_grids_by_bounds(self, bounds_wkt: str, grid_type: Optional[str] = None,
                           processing_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get grids that intersect with given bounds."""
        with db.get_cursor() as cursor:
            query = """
                SELECT g.*, ST_AsText(g.bounds) as bounds_wkt,
                       ST_Area(ST_Intersection(g.bounds, ST_GeomFromText(%s, 4326))::geography) / 1000000.0 as intersection_area_km2
                FROM grids g 
                WHERE g.bounds IS NOT NULL 
                  AND ST_Intersects(g.bounds, ST_GeomFromText(%s, 4326))
            """
            params = [bounds_wkt, bounds_wkt]
            
            if grid_type:
                query += " AND g.grid_type = %s"
                params.append(grid_type)
            
            if processing_status:
                query += " AND g.processing_status = %s"
                params.append(processing_status)
            
            query += " ORDER BY intersection_area_km2 DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_grid_cells_in_bounds(self, grid_id: str, bounds_wkt: str) -> List[Dict[str, Any]]:
        """Get grid cells that intersect with given bounds."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT gc.cell_id, ST_AsText(gc.geometry) as geometry_wkt,
                       gc.area_km2, ST_AsText(gc.centroid) as centroid_wkt,
                       ST_Area(ST_Intersection(gc.geometry, ST_GeomFromText(%s, 4326))::geography) / 1000000.0 as intersection_area_km2
                FROM grid_cells gc
                WHERE gc.grid_id = %s 
                  AND ST_Intersects(gc.geometry, ST_GeomFromText(%s, 4326))
                ORDER BY intersection_area_km2 DESC
            """, (bounds_wkt, grid_id, bounds_wkt))
            return cursor.fetchall()
    
    # Export Metadata Operations
    def store_export_metadata(self, export_data: Dict[str, Any]) -> str:
        """Store export file metadata."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO export_metadata 
                (grid_id, export_type, feature_types, file_path, file_name, 
                 file_size_bytes, format_version, compression, spatial_extent,
                 temporal_range, metadata, checksum, expires_at, created_by)
                VALUES (%(grid_id)s, %(export_type)s, %(feature_types)s, %(file_path)s,
                        %(file_name)s, %(file_size_bytes)s, %(format_version)s, 
                        %(compression)s, ST_GeomFromText(%(spatial_extent)s, 4326),
                        %(temporal_range)s, %(metadata)s, %(checksum)s, 
                        %(expires_at)s, %(created_by)s)
                RETURNING id
            """, {
                'grid_id': export_data['grid_id'],
                'export_type': export_data['export_type'],
                'feature_types': export_data.get('feature_types', []),
                'file_path': export_data['file_path'],
                'file_name': export_data['file_name'],
                'file_size_bytes': export_data.get('file_size_bytes'),
                'format_version': export_data.get('format_version'),
                'compression': export_data.get('compression', 'none'),
                'spatial_extent': export_data.get('spatial_extent_wkt'),
                'temporal_range': json.dumps(export_data.get('temporal_range')) if export_data.get('temporal_range') else None,
                'metadata': json.dumps(export_data.get('metadata', {})),
                'checksum': export_data.get('checksum'),
                'expires_at': export_data.get('expires_at'),
                'created_by': export_data.get('created_by', 'system')
            })
            export_id = cursor.fetchone()['id']
            logger.info(f"✅ Stored export metadata: {export_data['file_name']} ({export_id})")
            return export_id
    
    def get_export_metadata(self, grid_id: Optional[str] = None, 
                           export_type: Optional[str] = None,
                           active_only: bool = True) -> List[Dict[str, Any]]:
        """Get export metadata with optional filtering."""
        with db.get_cursor() as cursor:
            query = """
                SELECT em.*, ST_AsText(em.spatial_extent) as spatial_extent_wkt,
                       g.name as grid_name
                FROM export_metadata em
                JOIN grids g ON em.grid_id = g.id
                WHERE 1=1
            """
            params = []
            
            if grid_id:
                query += " AND em.grid_id = %s"
                params.append(grid_id)
            
            if export_type:
                query += " AND em.export_type = %s"
                params.append(export_type)
            
            if active_only:
                query += " AND (em.expires_at IS NULL OR em.expires_at > NOW())"
            
            query += " ORDER BY em.created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def cleanup_expired_exports(self) -> int:
        """Remove expired export metadata."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT cleanup_expired_exports()")
            deleted_count = cursor.fetchone()[0]
            if deleted_count > 0:
                logger.info(f"✅ Cleaned up {deleted_count} expired export records")
            return deleted_count
    
    def get_export_summary(self, grid_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get export summary statistics."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM export_summary"
            params = []
            
            if grid_id:
                query += " WHERE grid_id = %s"
                params.append(grid_id)
            
            query += " ORDER BY grid_name, export_type"
            cursor.execute(query, params)
            return cursor.fetchall()

    # Enhanced utility methods for bounds queries
    def get_grids_summary_with_bounds(self) -> List[Dict[str, Any]]:
        """Get grid summary including bounds information."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    id as grid_id,
                    name as grid_name,
                    grid_type,
                    resolution,
                    processing_status,
                    total_cells,
                    ST_AsText(bounds) as bounds_wkt,
                    CASE 
                        WHEN bounds IS NOT NULL THEN ST_Area(bounds::geography) / 1000000.0
                        ELSE NULL 
                    END as bounds_area_km2,
                    created_at,
                    metadata
                FROM grids
                ORDER BY created_at DESC
            """)
            return cursor.fetchall()
    
    def run_migration(self, migration_file: str) -> bool:
        """Run a database migration file."""
        try:
            migration_path = Path(__file__).parent / "migrations" / migration_file
            if not migration_path.exists():
                logger.error(f"❌ Migration file not found: {migration_file}")
                return False
            
            db.execute_sql_file(migration_path)
            logger.info(f"✅ Migration applied successfully: {migration_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False

    # ==============================================================================
    # RASTER DATA OPERATIONS
    # ==============================================================================
    
    def store_raster_source(self, raster_data: Dict[str, Any]) -> str:
        """Store raster source metadata."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO raster_sources 
                (name, file_path, data_type, pixel_size_degrees, spatial_extent,
                 nodata_value, band_count, file_size_mb, checksum, last_modified,
                 source_dataset, variable_name, units, description, temporal_info, metadata)
                VALUES (%(name)s, %(file_path)s, %(data_type)s, %(pixel_size_degrees)s,
                        ST_GeomFromText(%(spatial_extent)s, 4326), %(nodata_value)s,
                        %(band_count)s, %(file_size_mb)s, %(checksum)s, %(last_modified)s,
                        %(source_dataset)s, %(variable_name)s, %(units)s, %(description)s,
                        %(temporal_info)s, %(metadata)s)
                RETURNING id
            """, {
                'name': raster_data['name'],
                'file_path': raster_data['file_path'],
                'data_type': raster_data['data_type'],
                'pixel_size_degrees': raster_data.get('pixel_size_degrees', 0.016666666666667),
                'spatial_extent': raster_data['spatial_extent_wkt'],
                'nodata_value': raster_data.get('nodata_value'),
                'band_count': raster_data.get('band_count', 1),
                'file_size_mb': raster_data.get('file_size_mb'),
                'checksum': raster_data.get('checksum'),
                'last_modified': raster_data.get('last_modified'),
                'source_dataset': raster_data.get('source_dataset'),
                'variable_name': raster_data.get('variable_name'),
                'units': raster_data.get('units'),
                'description': raster_data.get('description'),
                'temporal_info': json.dumps(raster_data.get('temporal_info', {})),
                'metadata': json.dumps(raster_data.get('metadata', {}))
            })
            raster_id = cursor.fetchone()['id']
            logger.info(f"✅ Stored raster source: {raster_data['name']} ({raster_id})")
            return raster_id

    def get_raster_sources(self, active_only: bool = True, 
                          processing_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get raster sources with optional filtering."""
        with db.get_cursor() as cursor:
            query = """
                SELECT *, ST_AsText(spatial_extent) as spatial_extent_wkt
                FROM raster_sources
                WHERE 1=1
            """
            params = []
            
            if active_only:
                query += " AND active = TRUE"
            
            if processing_status:
                query += " AND processing_status = %s"
                params.append(processing_status)
            
            query += " ORDER BY name"
            cursor.execute(query, params)
            return cursor.fetchall()

    def update_raster_processing_status(self, raster_id: str, status: str,
                                       metadata: Optional[Dict] = None) -> bool:
        """Update raster processing status."""
        with db.get_cursor() as cursor:
            try:
                cursor.execute("""
                    UPDATE raster_sources
                    SET processing_status = %s,
                        metadata = CASE 
                            WHEN %s IS NOT NULL THEN metadata || %s::jsonb
                            ELSE metadata 
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, json.dumps(metadata) if metadata else None,
                      json.dumps(metadata) if metadata else None, raster_id))
                return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"❌ Failed to update raster status: {e}")
                return False

    def store_raster_tiles_batch(self, raster_id: str, tiles_data: List[Dict]) -> int:
        """Bulk store raster tiles."""
        with db.get_cursor() as cursor:
            tile_records = []
            for tile in tiles_data:
                tile_records.append((
                    raster_id,
                    tile['tile_x'],
                    tile['tile_y'],
                    tile.get('tile_size_pixels', 1000),
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
                    tile_bounds = EXCLUDED.tile_bounds,
                    file_byte_offset = EXCLUDED.file_byte_offset,
                    file_byte_length = EXCLUDED.file_byte_length,
                    tile_stats = EXCLUDED.tile_stats
            """, tile_records)
            
            inserted_count = cursor.rowcount
            logger.info(f"✅ Stored {inserted_count} raster tiles for raster {raster_id}")
            return inserted_count

    def get_raster_tiles_for_bounds(self, raster_id: str, bounds_wkt: str) -> List[Dict[str, Any]]:
        """Get raster tiles that intersect with given bounds."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT rt.*, ST_AsText(rt.tile_bounds) as tile_bounds_wkt
                FROM raster_tiles rt
                WHERE rt.raster_source_id = %s
                AND ST_Intersects(rt.tile_bounds, ST_GeomFromText(%s, 4326))
                ORDER BY rt.tile_x, rt.tile_y
            """, (raster_id, bounds_wkt))
            return cursor.fetchall()

    def store_resampling_cache_batch(self, cache_data: List[Dict]) -> int:
        """Bulk store resampling cache entries."""
        with db.get_cursor() as cursor:
            cache_records = []
            for entry in cache_data:
                cache_records.append((
                    entry['source_raster_id'],
                    entry['target_grid_id'],
                    entry['cell_id'],
                    entry.get('method', 'bilinear'),
                    entry.get('band_number', 1),
                    entry['value'],
                    entry.get('confidence_score', 1.0),
                    entry.get('source_tiles_used', []),
                    json.dumps(entry.get('computation_metadata', {}))
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
                                   cell_ids: List[str], method: str = 'bilinear',
                                   band_number: int = 1) -> Dict[str, float]:
        """Get cached resampling values for given cells."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                UPDATE resampling_cache 
                SET last_accessed = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE source_raster_id = %s AND target_grid_id = %s 
                AND cell_id = ANY(%s) AND method = %s AND band_number = %s
                RETURNING cell_id, value
            """, (raster_id, grid_id, cell_ids, method, band_number))
            
            return {row['cell_id']: row['value'] for row in cursor.fetchall()}

    def add_processing_task(self, queue_type: str, raster_id: Optional[str] = None,
                           grid_id: Optional[str] = None, tile_id: Optional[str] = None,
                           parameters: Optional[Dict] = None, priority: int = 0) -> str:
        """Add task to processing queue."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO processing_queue 
                (queue_type, raster_source_id, grid_id, tile_id, parameters, priority)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (queue_type, raster_id, grid_id, tile_id, 
                  json.dumps(parameters or {}), priority))
            task_id = cursor.fetchone()['id']
            logger.info(f"✅ Added processing task: {queue_type} ({task_id})")
            return task_id

    def get_next_processing_task(self, queue_type: str, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get next processing task for worker."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT get_next_processing_task(%s, %s)", (queue_type, worker_id))
            result = cursor.fetchone()
            task_id = result['get_next_processing_task'] if result else None
            
            if task_id:
                cursor.execute("""
                    SELECT pq.*, rs.name as raster_name, g.name as grid_name
                    FROM processing_queue pq
                    LEFT JOIN raster_sources rs ON pq.raster_source_id = rs.id
                    LEFT JOIN grids g ON pq.grid_id = g.id
                    WHERE pq.id = %s
                """, (task_id,))
                return cursor.fetchone()
            return None

    def complete_processing_task(self, task_id: str, success: bool = True,
                                error_message: Optional[str] = None,
                                checkpoint_data: Optional[Dict] = None) -> bool:
        """Mark processing task as completed or failed."""
        with db.get_cursor() as cursor:
            if success:
                cursor.execute("""
                    UPDATE processing_queue 
                    SET status = 'completed',
                        completed_at = CURRENT_TIMESTAMP,
                        checkpoint_data = %s
                    WHERE id = %s
                """, (json.dumps(checkpoint_data or {}), task_id))
            else:
                cursor.execute("""
                    UPDATE processing_queue 
                    SET status = 'failed',
                        error_message = %s,
                        retry_count = retry_count + 1,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (error_message, task_id))
            
            return cursor.rowcount > 0

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
        """Clean up old and rarely used cache entries."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT cleanup_resampling_cache(%s, %s)", (days_old, min_access_count))
            result = cursor.fetchone()
            deleted_count = result['cleanup_resampling_cache'] if result else 0
            if deleted_count > 0:
                logger.info(f"✅ Cleaned up {deleted_count} old cache entries")
            return deleted_count

    def get_processing_queue_summary(self) -> List[Dict[str, Any]]:
        """Get processing queue statistics."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM processing_queue_summary")
            return cursor.fetchall()

# Global schema instance
schema = DatabaseSchema()


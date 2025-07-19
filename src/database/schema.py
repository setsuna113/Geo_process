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
        # Get CRS from config
        grid_config = config.get(f'grids.{grid_type}')
        if not grid_config:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        crs = grid_config['crs']
        
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

# Global schema instance
schema = DatabaseSchema()
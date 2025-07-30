"""Schema management operations for database schema."""

from typing import Dict, Any, List
import logging
from pathlib import Path
from ..connection import db

logger = logging.getLogger(__name__)


class SchemaManagement:
    """Database schema management operations."""
    
    def __init__(self):
        # Schema file location
        self.schema_file = Path(__file__).parent.parent / "schema.sql"
    
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
    
    def create_all_tables(self) -> bool:
        """Create all database tables (alias for create_schema)."""
        return self.create_schema()
    
    def drop_all_tables(self, confirm: bool = False) -> bool:
        """Drop all database tables (alias for drop_schema)."""
        return self.drop_schema(confirm=confirm)
#!/usr/bin/env python3
"""Add monitoring and logging tables to database.

This script creates the new monitoring schema tables for unified logging and progress tracking.
It can be run multiple times safely as it uses IF NOT EXISTS clauses.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import these later to avoid connection at import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringMigration:
    """Handle migration of monitoring schema."""
    
    def __init__(self, db_manager = None):
        """Initialize migration handler."""
        # Import here to avoid connection at module load
        from src.config.config import Config
        from src.database.connection import DatabaseManager
        from src.database.schema import DatabaseSchema
        
        self.config = Config()
        self.db = db_manager or DatabaseManager()
        self.schema = DatabaseSchema(self.db)
        self.monitoring_schema_file = project_root / "src/database/monitoring_schema.sql"
        
    def check_existing_tables(self) -> dict:
        """Check which monitoring tables already exist."""
        tables_to_check = [
            'pipeline_logs',
            'pipeline_events', 
            'pipeline_progress',
            'pipeline_metrics'
        ]
        
        existing_tables = {}
        
        with self.db.get_cursor() as cursor:
            for table in tables_to_check:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table,))
                exists = cursor.fetchone()['exists']
                existing_tables[table] = exists
                
        return existing_tables
    
    def check_existing_views(self) -> dict:
        """Check which monitoring views already exist."""
        views_to_check = [
            'recent_experiment_errors',
            'experiment_progress_summary',
            'active_pipeline_monitoring'
        ]
        
        existing_views = {}
        
        with self.db.get_cursor() as cursor:
            for view in views_to_check:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.views
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    );
                """, (view,))
                exists = cursor.fetchone()['exists']
                existing_views[view] = exists
                
        return existing_views
    
    def run_migration(self, force: bool = False) -> bool:
        """Run the monitoring schema migration.
        
        Args:
            force: If True, drop existing tables first (dangerous!)
            
        Returns:
            True if migration successful
        """
        try:
            # Check what already exists
            existing_tables = self.check_existing_tables()
            existing_views = self.check_existing_views()
            
            logger.info("Checking existing schema...")
            for table, exists in existing_tables.items():
                status = "✓ exists" if exists else "✗ missing"
                logger.info(f"  Table {table}: {status}")
                
            for view, exists in existing_views.items():
                status = "✓ exists" if exists else "✗ missing"
                logger.info(f"  View {view}: {status}")
            
            # If tables exist and not forcing, ask for confirmation
            if any(existing_tables.values()) and not force:
                logger.warning("Some monitoring tables already exist.")
                response = input("Continue with migration? Existing tables will be preserved. [y/N]: ")
                if response.lower() != 'y':
                    logger.info("Migration cancelled.")
                    return False
            
            # If forcing, drop existing tables first
            if force:
                logger.warning("Force mode: dropping existing monitoring tables...")
                self._drop_monitoring_tables()
            
            # Run the migration
            logger.info("Running monitoring schema migration...")
            
            if not self.monitoring_schema_file.exists():
                logger.error(f"Schema file not found: {self.monitoring_schema_file}")
                return False
                
            self.db.execute_sql_file(self.monitoring_schema_file)
            
            logger.info("✅ Monitoring schema migration completed successfully!")
            
            # Verify creation
            new_tables = self.check_existing_tables()
            new_views = self.check_existing_views()
            
            logger.info("\nVerifying migration results:")
            for table, exists in new_tables.items():
                status = "✓ created" if exists else "✗ failed"
                logger.info(f"  Table {table}: {status}")
                
            for view, exists in new_views.items():
                status = "✓ created" if exists else "✗ failed"
                logger.info(f"  View {view}: {status}")
            
            return all(new_tables.values())
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
    
    def _drop_monitoring_tables(self):
        """Drop monitoring tables (use with caution!)."""
        with self.db.get_cursor() as cursor:
            # Drop in reverse dependency order
            cursor.execute("""
                -- Drop views first
                DROP VIEW IF EXISTS active_pipeline_monitoring CASCADE;
                DROP VIEW IF EXISTS experiment_progress_summary CASCADE;
                DROP VIEW IF EXISTS recent_experiment_errors CASCADE;
                
                -- Drop functions
                DROP FUNCTION IF EXISTS get_experiment_logs CASCADE;
                DROP FUNCTION IF EXISTS get_progress_tree CASCADE;
                DROP FUNCTION IF EXISTS get_error_summary CASCADE;
                DROP FUNCTION IF EXISTS update_parent_progress CASCADE;
                DROP FUNCTION IF EXISTS cleanup_old_logs CASCADE;
                
                -- Drop triggers
                DROP TRIGGER IF EXISTS update_parent_progress_trigger ON pipeline_progress;
                
                -- Drop tables
                DROP TABLE IF EXISTS pipeline_metrics CASCADE;
                DROP TABLE IF EXISTS pipeline_progress CASCADE;
                DROP TABLE IF EXISTS pipeline_events CASCADE;
                DROP TABLE IF EXISTS pipeline_logs CASCADE;
            """)
            logger.info("Dropped existing monitoring tables and views")
    
    def create_test_data(self, experiment_id: str):
        """Create test data for monitoring tables.
        
        Args:
            experiment_id: UUID of an existing experiment
        """
        logger.info(f"Creating test data for experiment {experiment_id}")
        
        with self.db.get_cursor() as cursor:
            # Create some progress nodes
            cursor.execute("""
                INSERT INTO pipeline_progress 
                (experiment_id, node_id, node_level, node_name, status, progress_percent)
                VALUES 
                (%s, 'pipeline/test', 'pipeline', 'Test Pipeline', 'running', 25),
                (%s, 'pipeline/test/load', 'phase', 'Data Loading', 'completed', 100),
                (%s, 'pipeline/test/process', 'phase', 'Processing', 'running', 50),
                (%s, 'pipeline/test/process/step1', 'step', 'Step 1', 'completed', 100),
                (%s, 'pipeline/test/process/step2', 'step', 'Step 2', 'running', 0)
                ON CONFLICT (experiment_id, node_id) DO NOTHING;
            """, (experiment_id, experiment_id, experiment_id, experiment_id, experiment_id))
            
            # Create some logs
            cursor.execute("""
                INSERT INTO pipeline_logs
                (experiment_id, node_id, level, logger_name, message, context)
                VALUES
                (%s, 'pipeline/test/load', 'INFO', 'test.loader', 'Started data loading', '{"stage": "load"}'::jsonb),
                (%s, 'pipeline/test/load', 'INFO', 'test.loader', 'Completed data loading', '{"stage": "load"}'::jsonb),
                (%s, 'pipeline/test/process', 'WARNING', 'test.processor', 'Low memory warning', '{"stage": "process"}'::jsonb),
                (%s, 'pipeline/test/process', 'ERROR', 'test.processor', 'Failed to process item', '{"stage": "process"}'::jsonb);
            """, (experiment_id, experiment_id, experiment_id, experiment_id))
            
            # Create some events
            cursor.execute("""
                INSERT INTO pipeline_events
                (experiment_id, event_type, source, title, severity)
                VALUES
                (%s, 'stage_start', 'orchestrator', 'Started data loading phase', 'info'),
                (%s, 'stage_complete', 'orchestrator', 'Completed data loading phase', 'info'),
                (%s, 'warning', 'memory_monitor', 'Memory usage above 80%%', 'warning');
            """, (experiment_id, experiment_id, experiment_id))
            
            logger.info("✅ Test data created successfully")
    
    def show_sample_queries(self):
        """Display sample queries for the new monitoring tables."""
        print("\n" + "="*60)
        print("SAMPLE MONITORING QUERIES")
        print("="*60)
        
        queries = [
            ("Get recent errors for an experiment:", """
SELECT * FROM recent_experiment_errors 
WHERE experiment_name = 'your_experiment' 
LIMIT 10;"""),
            
            ("Get progress tree for an experiment:", """
SELECT * FROM get_progress_tree('experiment_uuid'::uuid);"""),
            
            ("Get error summary:", """
SELECT * FROM get_error_summary('experiment_uuid'::uuid);"""),
            
            ("Query logs with filters:", """
SELECT * FROM get_experiment_logs(
    'experiment_uuid'::uuid,
    p_level := 'ERROR',
    p_search := 'memory',
    p_limit := 50
);"""),
            
            ("View active pipeline monitoring:", """
SELECT * FROM active_pipeline_monitoring;"""),
            
            ("Get experiment progress summary:", """
SELECT * FROM experiment_progress_summary 
WHERE experiment_name = 'your_experiment';""")
        ]
        
        for title, query in queries:
            print(f"\n{title}")
            print("-" * len(title))
            print(query)
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate database schema for unified monitoring and logging"
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Drop existing monitoring tables before creating (DANGEROUS!)'
    )
    parser.add_argument(
        '--test-data',
        metavar='EXPERIMENT_ID',
        help='Create test data for the given experiment ID'
    )
    parser.add_argument(
        '--show-queries',
        action='store_true',
        help='Show sample queries for monitoring tables'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check what tables exist, do not migrate'
    )
    
    args = parser.parse_args()
    
    try:
        migration = MonitoringMigration()
        
        if args.check_only:
            migration.check_existing_tables()
            migration.check_existing_views()
            return
        
        # Run migration
        success = migration.run_migration(force=args.force)
        
        if success:
            # Create test data if requested
            if args.test_data:
                migration.create_test_data(args.test_data)
            
            # Show sample queries if requested
            if args.show_queries:
                migration.show_sample_queries()
                
            logger.info("\nMigration completed! You can now use the unified monitoring system.")
            logger.info("Run with --show-queries to see example usage.")
        else:
            logger.error("Migration failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Migration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
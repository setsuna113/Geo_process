#!/usr/bin/env python3
"""
Migration script to add coordinate columns to existing storage tables.

This script migrates legacy tables that only have row_idx/col_idx to include
x_coord/y_coord columns for better alignment handling and SQL-based processing.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.connection import DatabaseManager
from src.database.schema import DatabaseSchema
from src.config import config
from src.infrastructure.logging import setup_simple_logging, get_logger

logger = get_logger(__name__)


class CoordinateStorageMigrator:
    """Handles migration of storage tables to include coordinate columns."""
    
    def __init__(self, db: DatabaseManager, dry_run: bool = False):
        """Initialize migrator.
        
        Args:
            db: Database connection manager
            dry_run: If True, only show what would be done
        """
        self.db = db
        self.schema = DatabaseSchema(db)
        self.dry_run = dry_run
        
    def find_tables_needing_migration(self) -> List[str]:
        """Find tables that need coordinate columns added.
        
        Returns:
            List of table names needing migration
        """
        tables_to_migrate = []
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Find all passthrough_ and resampled_ tables (but not resampled_datasets metadata table)
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND (table_name LIKE 'passthrough_%' 
                     OR (table_name LIKE 'resampled_%' AND table_name != 'resampled_datasets')
                     OR table_name LIKE 'windowed_%')
            """)
            
            for row in cursor.fetchall():
                table_name = row[0]
                
                # Check if table has coordinate columns
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    AND column_name IN ('x_coord', 'y_coord')
                """, (table_name,))
                
                if cursor.fetchone()[0] < 2:
                    tables_to_migrate.append(table_name)
                    
        return tables_to_migrate
    
    def get_table_metadata(self, table_name: str) -> Optional[Tuple[Tuple, float]]:
        """Get bounds and resolution for a table from resampled_datasets.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Tuple of (bounds, resolution) or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Try to find in resampled_datasets table
            cursor.execute("""
                SELECT bounds, target_resolution
                FROM resampled_datasets
                WHERE data_table_name = %s
            """, (table_name,))
            
            row = cursor.fetchone()
            if row:
                bounds = row[0]
                resolution = row[1]
                return bounds, resolution
                
            # Try passthrough naming
            dataset_name = table_name.replace('passthrough_', '').replace('_', '-')
            cursor.execute("""
                SELECT bounds, target_resolution, metadata
                FROM resampled_datasets
                WHERE name = %s
            """, (dataset_name,))
            
            row = cursor.fetchone()
            if row:
                bounds = row[0]
                resolution = row[1]
                return bounds, resolution
                
        return None
    
    def migrate_table(self, table_name: str, bounds: Tuple, resolution: float) -> bool:
        """Migrate a single table to include coordinate columns.
        
        Args:
            table_name: Name of table to migrate
            bounds: Geographic bounds (min_x, min_y, max_x, max_y)
            resolution: Pixel resolution
            
        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would migrate table {table_name}")
            logger.info(f"  Bounds: {bounds}")
            logger.info(f"  Resolution: {resolution}")
            return True
            
        try:
            logger.info(f"Migrating table {table_name}...")
            
            # Use schema method to do the migration
            # Need to access monolithic schema for this method
            if hasattr(self.schema, '_get_monolithic_schema'):
                monolithic = self.schema._get_monolithic_schema()
                success = monolithic.migrate_legacy_table_to_coordinates(
                    table_name, bounds, resolution
                )
            else:
                # Should not happen, but handle gracefully
                logger.error("Cannot access migration method")
                return False
            
            if success:
                logger.info(f"✓ Successfully migrated {table_name}")
                
                # Verify migration
                if self.verify_migration(table_name):
                    logger.info(f"✓ Verified coordinate columns in {table_name}")
                else:
                    logger.error(f"✗ Verification failed for {table_name}")
                    return False
            else:
                logger.error(f"✗ Failed to migrate {table_name}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error migrating {table_name}: {e}")
            return False
    
    def verify_migration(self, table_name: str) -> bool:
        """Verify that migration was successful.
        
        Args:
            table_name: Table to verify
            
        Returns:
            True if verification passes
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check columns exist
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s 
                AND column_name IN ('x_coord', 'y_coord')
            """, (table_name,))
            
            if len(cursor.fetchall()) != 2:
                return False
                
            # Check some values are populated
            cursor.execute(f"""
                SELECT COUNT(*) as total,
                       COUNT(x_coord) as with_x,
                       COUNT(y_coord) as with_y
                FROM {table_name}
                WHERE value IS NOT NULL AND value != 0
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row and row[0] > 0:  # total
                return row[1] == row[0] and row[2] == row[0]  # with_x == total and with_y == total
                
        return True
    
    def create_rollback_script(self, tables: List[str]) -> str:
        """Create SQL script to rollback migration.
        
        Args:
            tables: List of migrated tables
            
        Returns:
            SQL rollback script
        """
        rollback_sql = "-- Rollback script for coordinate migration\n\n"
        
        for table in tables:
            rollback_sql += f"""
-- Rollback {table}
ALTER TABLE {table} DROP COLUMN IF EXISTS x_coord;
ALTER TABLE {table} DROP COLUMN IF EXISTS y_coord;

"""
        
        return rollback_sql
    
    def run_migration(self) -> bool:
        """Run the full migration process.
        
        Returns:
            True if all migrations successful
        """
        logger.info("Starting coordinate storage migration...")
        
        # Find tables needing migration
        tables = self.find_tables_needing_migration()
        
        if not tables:
            logger.info("No tables need migration!")
            return True
            
        logger.info(f"Found {len(tables)} tables needing migration:")
        for table in tables:
            logger.info(f"  - {table}")
            
        # Migrate each table
        migrated_tables = []
        failed_tables = []
        
        for table in tables:
            # Get metadata
            metadata = self.get_table_metadata(table)
            if not metadata:
                logger.warning(f"Could not find metadata for {table}, skipping")
                continue
                
            bounds, resolution = metadata
            
            # Migrate
            if self.migrate_table(table, bounds, resolution):
                migrated_tables.append(table)
            else:
                failed_tables.append(table)
                
        # Summary
        logger.info("\n=== Migration Summary ===")
        logger.info(f"Total tables: {len(tables)}")
        logger.info(f"Migrated successfully: {len(migrated_tables)}")
        logger.info(f"Failed: {len(failed_tables)}")
        
        if failed_tables:
            logger.error("Failed tables:")
            for table in failed_tables:
                logger.error(f"  - {table}")
                
        # Create rollback script
        if migrated_tables and not self.dry_run:
            rollback_path = Path("rollback_coordinate_migration.sql")
            rollback_sql = self.create_rollback_script(migrated_tables)
            rollback_path.write_text(rollback_sql)
            logger.info(f"\nRollback script saved to: {rollback_path}")
            
        return len(failed_tables) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate storage tables to include coordinate columns"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Specific tables to migrate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_simple_logging()
    
    # Initialize database
    db = DatabaseManager()
    
    try:
        # Run migration
        migrator = CoordinateStorageMigrator(db, dry_run=args.dry_run)
        
        if args.tables:
            # Migrate specific tables
            for table in args.tables:
                metadata = migrator.get_table_metadata(table)
                if metadata:
                    bounds, resolution = metadata
                    migrator.migrate_table(table, bounds, resolution)
                else:
                    logger.error(f"Could not find metadata for {table}")
        else:
            # Run full migration
            success = migrator.run_migration()
            sys.exit(0 if success else 1)
            
    finally:
        pass  # DatabaseManager handles cleanup automatically


if __name__ == "__main__":
    main()
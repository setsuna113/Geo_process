"""Database setup and initialization script."""

import sys
import logging
from typing import Dict, Any
from .schema import schema
from .connection import db
from ..config import config

logger = logging.getLogger(__name__)

def setup_database(reset: bool = False) -> bool:
    """Initialize database schema and verify setup."""
    try:
        logger.info("🚀 Starting database setup...")
        
        # Test connection first
        logger.info("Testing database connection...")
        if not db.test_connection():
            raise Exception("Database connection failed")
        
        # Drop existing schema if reset requested
        if reset:
            logger.warning("⚠️ Resetting database schema...")
            if not schema.drop_schema(confirm=True):
                raise Exception("Schema reset failed")
        
        # Create schema
        logger.info("Creating database schema...")
        if not schema.create_schema():
            raise Exception("Schema creation failed")
        
        # Verify schema
        logger.info("Verifying schema creation...")
        info = schema.get_schema_info()
        
        # Log schema summary
        logger.info("✅ Database schema created successfully!")
        logger.info(f"   📊 Tables: {info['summary']['table_count']}")
        logger.info(f"   👁️ Views: {info['summary']['view_count']}")
        
        # Log individual tables
        for table in info['tables']:
            count = info['table_counts'][table['table_name']]
            logger.info(f"     • {table['table_name']}: {table['column_count']} columns, {count} rows ({table['size']})")
        
        # Log views
        for view in info['views']:
            logger.info(f"     • {view['view_name']} (view)")
        
        # Validate grid configuration compatibility
        logger.info("Validating grid configurations...")
        grid_configs = config.get('grids', {})
        for grid_type, grid_config in grid_configs.items():
            for resolution in grid_config.get('resolutions', []):
                if schema.validate_grid_config(grid_type, resolution):
                    logger.info(f"   ✅ {grid_type} resolution {resolution}")
                else:
                    logger.warning(f"   ⚠️ Invalid {grid_type} resolution {resolution}")
        
        # Test basic operations
        logger.info("Testing basic database operations...")
        
        # Test experiment creation
        try:
            exp_id = schema.create_experiment(
                name="setup_test",
                description="Database setup verification",
                config={"test": True}
            )
            schema.update_experiment_status(exp_id, "completed", {"setup": "success"})
            logger.info("   ✅ Experiment tracking working")
        except Exception as e:
            logger.warning(f"   ⚠️ Experiment tracking test failed: {e}")
        
        # Connection info
        conn_info = db.get_connection_info()
        logger.info(f"✅ Connected to {conn_info.get('database')} as {conn_info.get('user')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def reset_database() -> bool:
    """Reset database schema (development only)."""
    logger.warning("⚠️ RESETTING DATABASE SCHEMA")
    logger.warning("   This will delete all data!")
    
    return setup_database(reset=True)

def check_database_status() -> Dict[str, Any]:
    """Check database status and readiness."""
    status = {
        'status': 'unknown',
        'connection': False,
        'schema': False,
        'ready': False,
        'info': None  # ← Initialize here
    }
    
    try:
        # Test connection
        connection_ok = db.test_connection()
        status['connection'] = connection_ok
        
        if not connection_ok:
            status['status'] = 'connection_failed'
            return status
        
        # Check schema
        info = None  # ← Initialize here too
        try:
            info = schema.get_schema_info()
            schema_ok = info['summary']['table_count'] >= 7
            status['schema'] = schema_ok
        except Exception as e:
            schema_ok = False
            status['schema'] = False
            logger.error(f"Schema check failed: {e}")
        
        # Update status
        status.update({
            'status': 'ready' if (connection_ok and schema_ok) else 'schema_incomplete',
            'ready': connection_ok and schema_ok,
            'info': info if schema_ok else None  # ← Now info is always defined
        })
        
    except Exception as e:
        status['status'] = 'error'
        logger.error(f"Database status check failed: {e}")
    
    return status

def main():
    """Main setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database setup utility")
    parser.add_argument("--reset", action="store_true", help="Reset database schema")
    parser.add_argument("--status", action="store_true", help="Check database status")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.status:
        # Check status
        status = check_database_status()
        print(f"Database Status: {status['status'].upper()}")
        print(f"Connection: {'✅' if status['connection'] else '❌'}")
        print(f"Schema: {'✅' if status['schema'] else '❌'}")
        print(f"Ready: {'✅' if status['ready'] else '❌'}")
        
        if status.get('info'):
            info = status['info']
            print(f"Tables: {info['summary']['table_count']}")
            print(f"Total rows: {info['summary']['total_rows']}")
        
        return 0 if status['ready'] else 1
    
    elif args.reset:
        # Reset database
        success = reset_database()
    else:
        # Regular setup
        success = setup_database()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Standalone migration script for monitoring schema."""

import psycopg2
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default connection parameters
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 51051,
    'database': 'geo_cluster_db',
    'user': 'jason',
    'password': '123456'
}

def get_connection(config):
    """Create database connection."""
    return psycopg2.connect(**config)

def run_migration(config, schema_file):
    """Run the monitoring schema migration."""
    conn = None
    try:
        logger.info(f"Connecting to database at {config['host']}:{config['port']}")
        conn = get_connection(config)
        cursor = conn.cursor()
        
        # Check if PostGIS is available
        try:
            cursor.execute("SELECT PostGIS_Version();")
            postgis_version = cursor.fetchone()[0]
            logger.info(f"PostGIS version: {postgis_version}")
        except:
            logger.warning("PostGIS not found, creating extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            conn.commit()
        
        # Read and execute schema file
        logger.info(f"Reading schema from {schema_file}")
        with open(schema_file, 'r') as f:
            sql = f.read()
        
        logger.info("Executing migration...")
        cursor.execute(sql)
        conn.commit()
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('pipeline_logs', 'pipeline_events', 'pipeline_progress', 'pipeline_metrics')
            ORDER BY table_name;
        """)
        
        created_tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Created tables: {', '.join(created_tables)}")
        
        logger.info("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Run monitoring schema migration")
    parser.add_argument('--host', default=DEFAULT_DB_CONFIG['host'], help='Database host')
    parser.add_argument('--port', type=int, default=DEFAULT_DB_CONFIG['port'], help='Database port')
    parser.add_argument('--database', default=DEFAULT_DB_CONFIG['database'], help='Database name')
    parser.add_argument('--user', default=DEFAULT_DB_CONFIG['user'], help='Database user')
    parser.add_argument('--password', default=DEFAULT_DB_CONFIG['password'], help='Database password')
    
    args = parser.parse_args()
    
    config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    schema_file = Path(__file__).parent.parent / "src/database/monitoring_schema.sql"
    
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return 1
    
    success = run_migration(config, schema_file)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""Test monitoring integration."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.infrastructure.logging import get_logger
from src.database.connection import DatabaseManager
from src.config import config

# Test structured logging
logger = get_logger(__name__)

logger.info("Testing monitoring integration", extra={
    'experiment_id': 'test_monitoring',
    'stage': 'test',
    'metrics': {'test_value': 42}
})

# Check if log was written to database
if config.get('monitoring.enable_database_logging', False):
    print(f"Database logging is enabled: {config.get('monitoring.enable_database_logging')}")
    
    db = DatabaseManager()
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM pipeline_logs 
                WHERE message LIKE '%Testing monitoring%'
            """)
            count = cur.fetchone()[0]
            print(f"Logs written to database: {count}")
else:
    print("Database logging is disabled")

print("Test complete!")
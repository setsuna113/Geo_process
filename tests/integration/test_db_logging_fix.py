#!/usr/bin/env python3
"""Test database logging fix."""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.infrastructure.logging import get_logger, setup_logging
from src.database.connection import DatabaseManager
from src.config import config

# Setup logging with database
db = DatabaseManager()
setup_logging(
    config=config,
    db_manager=db,
    console=True,
    database=True,
    experiment_id='test_db_logging_fix',
    log_level='INFO'
)

# Test logging
logger = get_logger(__name__)

logger.info("Test message 1: Testing database logging fix")
logger.info("Test message 2: With extra context", extra={
    'experiment_id': 'test_db_logging_fix',
    'stage': 'test',
    'node_id': 'test_node',
    'metrics': {'test_value': 42}
})
logger.warning("Test warning message")
logger.error("Test error message", extra={'error_code': 'TEST001'})

# Force flush
import logging
for handler in logging.getLogger().handlers:
    if hasattr(handler, 'flush'):
        handler.flush()

# Wait for async writes
time.sleep(3)

# Check database
print("\nChecking database for logs...")
with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT level, message, context
            FROM pipeline_logs 
            WHERE message LIKE 'Test message%' 
               OR message LIKE 'Test warning%'
               OR message LIKE 'Test error%'
            ORDER BY created_at DESC
            LIMIT 10
        """)
        logs = cur.fetchall()
        
        print(f"Logs found in database: {len(logs)}")
        for log in logs:
            print(f"  [{log['level']}] {log['message']}")
            if log['context'] and log['context'] != '{}':
                print(f"    Context: {log['context']}")

print("\nDatabase logging test complete!")
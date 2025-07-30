#!/usr/bin/env python3
"""Direct test of monitoring with setup."""

import sys
from pathlib import Path

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
    log_level='INFO'
)

# Now test logging
logger = get_logger(__name__)

logger.info("Test message 1: Basic info")
logger.info("Test message 2: With context", extra={
    'experiment_id': 'test_monitoring_direct',
    'stage': 'test',
    'metrics': {'test_value': 42}
})
logger.warning("Test warning message")
logger.error("Test error message", extra={'error_code': 'TEST001'})

# Force flush of any pending logs
import logging
for handler in logging.getLogger().handlers:
    if hasattr(handler, 'flush'):
        handler.flush()

# Give time for async writes
import time
time.sleep(2)

# Check database
with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT level, message, context
            FROM pipeline_logs 
            WHERE created_at > NOW() - INTERVAL '1 minute'
            ORDER BY created_at DESC
            LIMIT 10
        """)
        logs = cur.fetchall()
        
        print(f"\nLogs found in database: {len(logs)}")
        for level, message, context in logs:
            print(f"  [{level}] {message}")
            if context:
                print(f"    Context: {context}")

print("\nTest complete!")
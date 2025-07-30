#!/usr/bin/env python3
"""Direct test of DatabaseLogHandler."""

import sys
from pathlib import Path
import logging
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.infrastructure.logging.handlers.database_handler import DatabaseLogHandler
from src.database.connection import DatabaseManager
from src.config import config

# Create handler
db = DatabaseManager()
handler = DatabaseLogHandler(
    db_manager=db,
    batch_size=5,  # Small batch for testing
    flush_interval=2.0
)
handler.setLevel(logging.INFO)

# Create logger
logger = logging.getLogger('test_handler')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Test logging
print("Sending test messages...")
for i in range(10):
    logger.info(f"Test message {i+1}")
    
print("Waiting for flush...")
time.sleep(3)

# Force close to flush
print("Closing handler...")
handler.close()

# Check database
print("\nChecking database...")
with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) as count 
            FROM pipeline_logs 
            WHERE message LIKE 'Test message%'
        """)
        result = cur.fetchone()
        print(f"Messages in database: {result[0] if result else 0}")
        
        # Get details
        cur.execute("""
            SELECT level, message 
            FROM pipeline_logs 
            WHERE message LIKE 'Test message%'
            ORDER BY created_at DESC
            LIMIT 5
        """)
        logs = cur.fetchall()
        for log in logs:
            print(f"  [{log[0]}] {log[1]}")

print("\nTest complete!")
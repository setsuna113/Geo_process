"""
Database module for geoprocessing operations.

This module provides database connectivity and schema management.
It handles only database operations and data persistence.
"""

# Note: Database connections should be created explicitly when needed,
# not as a side effect of importing. Use get_db() and get_schema() functions.

__all__ = [
    'DatabaseSchema',
    'get_db',
    'get_schema',
]

# Re-export lazy initialization functions
from .connection import get_db
from .schema import get_schema

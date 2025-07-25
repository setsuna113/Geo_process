"""
Database module for geoprocessing operations.

This module provides database connectivity and schema management.
It handles only database operations and data persistence.
"""

# from .connection import db
from .schema import DatabaseSchema

__all__ = [
    'db',
    'DatabaseSchema',
]

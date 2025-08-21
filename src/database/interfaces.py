"""Database interfaces to break circular dependencies."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pathlib import Path


class DatabaseInterface(ABC):
    """Abstract interface for database operations."""
    
    @abstractmethod
    def get_cursor(self):
        """Get a database cursor."""
        pass
    
    @abstractmethod
    def execute_sql_file(self, file_path: Path) -> bool:
        """Execute SQL from a file."""
        pass
    
    @abstractmethod
    def get_connection(self):
        """Get the database connection."""
        pass


class SchemaInterface(ABC):
    """Abstract interface for database schema operations."""
    
    @abstractmethod
    def create_schema(self) -> bool:
        """Create database schema."""
        pass
    
    @abstractmethod
    def create_all_tables(self) -> bool:
        """Create all database tables."""
        pass
    
    @abstractmethod
    def drop_schema(self, confirm: bool = False) -> bool:
        """Drop database schema."""
        pass
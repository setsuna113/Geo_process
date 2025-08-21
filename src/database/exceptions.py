"""Database-specific exceptions for better error handling."""

import psycopg2
from typing import Optional, Any


class DatabaseError(Exception):
    """Base database error."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class DatabaseNotFoundError(DatabaseError):
    """Raised when expected database result is not found."""
    pass


class DatabaseDuplicateError(DatabaseError):
    """Raised when attempting to insert duplicate data."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraints are violated."""
    pass


def handle_database_error(operation_name: str):
    """Decorator to handle database errors consistently."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except psycopg2.IntegrityError as e:
                error_msg = str(e).lower()
                if 'duplicate key' in error_msg or 'unique constraint' in error_msg:
                    raise DatabaseDuplicateError(
                        f"{operation_name} failed: Duplicate entry detected", e
                    )
                else:
                    raise DatabaseIntegrityError(
                        f"{operation_name} failed: Integrity constraint violation", e
                    )
            except psycopg2.OperationalError as e:
                raise DatabaseConnectionError(
                    f"{operation_name} failed: Database connection error", e
                )
            except psycopg2.Error as e:
                raise DatabaseError(
                    f"{operation_name} failed: Database error", e
                )
            except Exception as e:
                raise DatabaseError(
                    f"{operation_name} failed: Unexpected error", e
                )
        return wrapper
    return decorator


def safe_fetch_one(cursor, operation_name: str) -> Any:
    """Safely fetch one result with null checking."""
    result = cursor.fetchone()
    if not result:
        raise DatabaseNotFoundError(f"{operation_name}: Expected result not found")
    return result


def safe_fetch_id(cursor, operation_name: str) -> str:
    """Safely fetch ID from database result."""
    result = safe_fetch_one(cursor, operation_name)
    if 'id' not in result:
        raise DatabaseError(f"{operation_name}: Result missing 'id' field")
    return result['id']
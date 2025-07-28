"""File handling utilities for safe writes and cleanup."""
from pathlib import Path
import tempfile
import atexit
import logging
from contextlib import contextmanager
from typing import Callable, Set

logger = logging.getLogger(__name__)

# Track temp files for cleanup
_temp_files: Set[Path] = set()

def _cleanup_temp_files():
    """Clean up any remaining temp files on exit."""
    for temp_file in list(_temp_files):
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to clean up {temp_file}: {e}")
        finally:
            _temp_files.discard(temp_file)

# Register cleanup on exit
atexit.register(_cleanup_temp_files)

@contextmanager
def temp_file_context(suffix: str = '.tmp', dir: Path = None):
    """Context manager for temporary files with guaranteed cleanup."""
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=dir)
    temp_path = Path(temp_path)
    _temp_files.add(temp_path)
    
    try:
        yield temp_path
    finally:
        try:
            import os
            os.close(temp_fd)  # Close file descriptor
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {temp_path}: {e}")
        finally:
            _temp_files.discard(temp_path)

def safe_write_file(file_path: Path, write_func: Callable):
    """Safely write file with overwrite handling and atomic replace."""
    file_path = Path(file_path)
    
    # Use temp file in same directory for atomic rename
    with temp_file_context(suffix='.tmp', dir=file_path.parent) as temp_path:
        # Write to temp file
        write_func(temp_path)
        
        # Atomic replace
        if file_path.exists():
            logger.warning(f"File exists, replacing: {file_path}")
        
        temp_path.replace(file_path)

def register_temp_file(temp_path: Path):
    """Register a temporary file for cleanup on exit."""
    _temp_files.add(Path(temp_path))

def cleanup_temp_file(temp_path: Path, ignore_errors: bool = True):
    """
    Clean up a specific temporary file with proper error handling.
    
    Args:
        temp_path: Path to temporary file
        ignore_errors: If True, log warnings on errors; if False, raise exceptions
    """
    temp_path = Path(temp_path)
    try:
        if temp_path.exists():
            temp_path.unlink()
            logger.debug(f"Cleaned up temp file: {temp_path}")
        _temp_files.discard(temp_path)
    except Exception as e:
        if ignore_errors:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}, will retry on exit")
        else:
            raise
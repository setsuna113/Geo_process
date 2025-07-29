# src/base/checkpoint.py
"""Core abstractions and interfaces for the unified checkpoint system.

This module defines the abstract base classes that all checkpoint implementations
must follow, providing a consistent interface across storage backends and validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import hashlib
import json
import time
import logging

from src.abstractions.types.checkpoint_types import (
    CheckpointData, CheckpointFilter, CheckpointLevel, CheckpointStatus,
    StorageConfig, StorageBackend,
    CheckpointError, CheckpointNotFoundError, CheckpointCorruptedError,
    CheckpointValidationError, StorageBackendError
)

logger = logging.getLogger(__name__)


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends.
    
    All storage implementations (file, database, memory) must implement this interface
    to ensure consistent behavior across the checkpoint system.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize storage backend with configuration."""
        self.config = config
        self.backend_type = config.backend
        
        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            raise StorageBackendError(f"Invalid storage config: {validation_errors}")
    
    @abstractmethod
    def save(self, checkpoint: CheckpointData) -> str:
        """Save checkpoint data to storage.
        
        Args:
            checkpoint: CheckpointData to save
            
        Returns:
            Storage location identifier (path, key, etc.)
            
        Raises:
            StorageBackendError: If save operation fails
        """
        pass
    
    @abstractmethod
    def load(self, checkpoint_id: str) -> CheckpointData:
        """Load checkpoint data from storage.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            CheckpointData instance
            
        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointCorruptedError: If checkpoint data is corrupted
            StorageBackendError: If load operation fails
        """
        pass
    
    @abstractmethod
    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists in storage.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from storage.
        
        Args:
            checkpoint_id: Unique identifier for the checkpoint
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            StorageBackendError: If delete operation fails
        """
        pass
    
    @abstractmethod
    def list(self, filter_criteria: Optional[CheckpointFilter] = None) -> List[CheckpointData]:
        """List checkpoints in storage with optional filtering.
        
        Args:
            filter_criteria: Optional filter to apply
            
        Returns:
            List of CheckpointData instances matching the filter
            
        Raises:
            StorageBackendError: If list operation fails
        """
        pass
    
    @abstractmethod
    def cleanup(self, max_age_days: int = 7, max_count: Optional[int] = None) -> int:
        """Clean up old checkpoints based on retention policy.
        
        Args:
            max_age_days: Remove checkpoints older than this many days
            max_count: Keep only the most recent N checkpoints
            
        Returns:
            Number of checkpoints removed
            
        Raises:
            StorageBackendError: If cleanup operation fails
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage backend statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        return {
            'backend_type': self.backend_type.value,
            'config': self.config.__dict__,
        }


class CheckpointValidator(ABC):
    """Abstract base class for checkpoint validation.
    
    Validators ensure checkpoint data integrity and can be customized
    for different types of checkpoint data or validation requirements.
    """
    
    @abstractmethod
    def validate(self, checkpoint: CheckpointData) -> bool:
        """Validate checkpoint data integrity.
        
        Args:
            checkpoint: CheckpointData to validate
            
        Returns:
            True if checkpoint is valid, False otherwise
            
        Raises:
            CheckpointValidationError: If validation fails with specific error
        """
        pass
    
    @abstractmethod
    def calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for checkpoint data.
        
        Args:
            data: Data dictionary to calculate checksum for
            
        Returns:
            Checksum string
        """
        pass
    
    @abstractmethod
    def verify_integrity(self, checkpoint: CheckpointData) -> bool:
        """Verify checkpoint data integrity against stored checksum.
        
        Args:
            checkpoint: CheckpointData to verify
            
        Returns:
            True if integrity check passes, False otherwise
        """
        pass


class DefaultCheckpointValidator(CheckpointValidator):
    """Default implementation of checkpoint validation using SHA256."""
    
    def validate(self, checkpoint: CheckpointData) -> bool:
        """Validate checkpoint data structure and integrity."""
        try:
            # Basic structure validation
            if not checkpoint.checkpoint_id:
                raise CheckpointValidationError("Missing checkpoint_id")
            
            if not checkpoint.level:
                raise CheckpointValidationError("Missing checkpoint level")
            
            if not isinstance(checkpoint.data, dict):
                raise CheckpointValidationError("Checkpoint data must be a dictionary")
            
            # Integrity validation if checksum exists
            if checkpoint.metadata.checksum:
                if not self.verify_integrity(checkpoint):
                    raise CheckpointValidationError("Checksum verification failed")
            
            return True
            
        except CheckpointValidationError:
            raise
        except Exception as e:
            raise CheckpointValidationError(f"Validation failed: {e}")
    
    def calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate SHA256 checksum of data."""
        # Convert data to JSON string for consistent hashing
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    def verify_integrity(self, checkpoint: CheckpointData) -> bool:
        """Verify checkpoint data against stored checksum."""
        if not checkpoint.metadata.checksum:
            return True  # No checksum to verify against
        
        calculated_checksum = self.calculate_checksum(checkpoint.data)
        return calculated_checksum == checkpoint.metadata.checksum


class CheckpointCompressor(ABC):
    """Abstract base class for checkpoint data compression."""
    
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress checkpoint data.
        
        Args:
            data: Raw data bytes to compress
            
        Returns:
            Compressed data bytes
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes) -> bytes:
        """Decompress checkpoint data.
        
        Args:
            compressed_data: Compressed data bytes
            
        Returns:
            Decompressed data bytes
        """
        pass
    
    @abstractmethod
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio.
        
        Args:
            original_size: Size of original data
            compressed_size: Size of compressed data
            
        Returns:
            Compression ratio (0.0 to 1.0, lower is better compression)
        """
        pass


class NoopCompressor(CheckpointCompressor):
    """No-operation compressor that doesn't actually compress data."""
    
    def compress(self, data: bytes) -> bytes:
        """Return data unchanged."""
        return data
    
    def decompress(self, compressed_data: bytes) -> bytes:
        """Return data unchanged."""
        return compressed_data
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Return 1.0 (no compression)."""
        return 1.0


def generate_checkpoint_id(prefix: str = "chk", level: Optional[CheckpointLevel] = None) -> str:
    """Generate a unique checkpoint ID.
    
    Args:
        prefix: Prefix for the ID
        level: Optional checkpoint level to include in ID
        
    Returns:
        Unique checkpoint ID string
    """
    import uuid
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = str(uuid.uuid4())[:8]
    
    if level:
        return f"{prefix}_{level.value}_{timestamp}_{unique_suffix}"
    return f"{prefix}_{timestamp}_{unique_suffix}"


def parse_checkpoint_id(checkpoint_id: str) -> Dict[str, Any]:
    """Parse information from a checkpoint ID.
    
    Args:
        checkpoint_id: Checkpoint ID to parse
        
    Returns:
        Dictionary with parsed information (prefix, level, timestamp, etc.)
    """
    parts = checkpoint_id.split('_')
    result = {'original_id': checkpoint_id}
    
    if len(parts) >= 1:
        result['prefix'] = parts[0]
    
    # Try to identify level
    for level in CheckpointLevel:
        if level.value in parts:
            result['level'] = level.value
            break
    
    # Try to extract timestamp (look for YYYYMMDD and HHMMSS patterns)
    timestamp_parts = []
    for part in parts:
        if len(part) == 8 and part.isdigit():  # YYYYMMDD
            timestamp_parts.append(part)
        elif len(part) == 6 and part.isdigit():  # HHMMSS  
            timestamp_parts.append(part)
    
    if len(timestamp_parts) >= 2:
        try:
            from datetime import datetime
            timestamp_str = f"{timestamp_parts[0]}_{timestamp_parts[1]}"
            result['timestamp'] = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    
    return result


class CheckpointMetrics:
    """Utility class for collecting checkpoint operation metrics."""
    
    def __init__(self):
        self.metrics = {
            'saves': {'count': 0, 'total_time_ms': 0.0, 'total_size_bytes': 0},
            'loads': {'count': 0, 'total_time_ms': 0.0},
            'validations': {'count': 0, 'failures': 0},
            'compressions': {'count': 0, 'total_ratio': 0.0},
        }
    
    def record_save(self, duration_ms: float, size_bytes: int):
        """Record a save operation."""
        self.metrics['saves']['count'] += 1
        self.metrics['saves']['total_time_ms'] += duration_ms
        self.metrics['saves']['total_size_bytes'] += size_bytes
    
    def record_load(self, duration_ms: float):
        """Record a load operation."""
        self.metrics['loads']['count'] += 1
        self.metrics['loads']['total_time_ms'] += duration_ms
    
    def record_validation(self, success: bool):
        """Record a validation operation."""
        self.metrics['validations']['count'] += 1
        if not success:
            self.metrics['validations']['failures'] += 1
    
    def record_compression(self, ratio: float):
        """Record a compression operation."""
        self.metrics['compressions']['count'] += 1
        self.metrics['compressions']['total_ratio'] += ratio
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        saves = self.metrics['saves']
        loads = self.metrics['loads']
        validations = self.metrics['validations']
        compressions = self.metrics['compressions']
        
        return {
            'saves': {
                'count': saves['count'],
                'avg_time_ms': saves['total_time_ms'] / max(saves['count'], 1),
                'avg_size_bytes': saves['total_size_bytes'] / max(saves['count'], 1),
            },
            'loads': {
                'count': loads['count'],
                'avg_time_ms': loads['total_time_ms'] / max(loads['count'], 1),
            },
            'validations': {
                'count': validations['count'],
                'failure_rate': validations['failures'] / max(validations['count'], 1),
            },
            'compressions': {
                'count': compressions['count'],
                'avg_ratio': compressions['total_ratio'] / max(compressions['count'], 1),
            },
        }


# Context manager for measuring checkpoint operations
class checkpoint_operation_timer:
    """Context manager for timing checkpoint operations."""
    
    def __init__(self, operation_name: str, metrics: Optional[CheckpointMetrics] = None):
        self.operation_name = operation_name
        self.metrics = metrics
        self.start_time = None
        self.duration_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.time() - self.start_time) * 1000
            logger.debug(f"{self.operation_name} took {self.duration_ms:.2f}ms")
    
    def get_duration_ms(self) -> float:
        """Get operation duration in milliseconds."""
        return self.duration_ms
# src/base/checkpoint_types.py
"""Core data structures and enums for the unified checkpoint system.

This module defines the fundamental types that all checkpoint operations
use across the biodiversity pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid


class CheckpointLevel(Enum):
    """Hierarchical levels for checkpoint organization."""
    PIPELINE = "pipeline"    # Top-level pipeline execution
    STAGE = "stage"         # Pipeline stages (data_load, resample, etc.)
    STEP = "step"           # Individual processing steps within stages  
    SUBSTEP = "substep"     # Fine-grained operations within steps


class CheckpointStatus(Enum):
    """Status of a checkpoint."""
    CREATED = "created"         # Just created, not yet validated
    VALID = "valid"            # Validated and ready for use
    CORRUPTED = "corrupted"    # Failed validation checks
    ARCHIVED = "archived"      # Old checkpoint, kept for history
    DELETED = "deleted"        # Marked for deletion


class StorageBackend(Enum):
    """Available storage backends for checkpoints."""
    FILE = "file"              # File system storage (JSON/binary)
    DATABASE = "database"      # PostgreSQL storage
    MEMORY = "memory"          # In-memory storage (testing)


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint."""
    process_name: str                    # Name of the process that created this
    process_version: str = "1.0.0"      # Version for compatibility checking
    tags: Dict[str, str] = field(default_factory=dict)  # Custom tags
    description: Optional[str] = None     # Human-readable description
    size_bytes: int = 0                  # Size of checkpoint data
    compression: Optional[str] = None     # Compression algorithm used
    checksum: Optional[str] = None       # Data integrity checksum
    
    # Performance metrics
    save_duration_ms: float = 0.0        # Time taken to save
    load_duration_ms: float = 0.0        # Time taken to load
    
    # Hierarchy information
    parent_checkpoint_id: Optional[str] = None
    child_checkpoint_ids: List[str] = field(default_factory=list)


@dataclass
class CheckpointData:
    """Universal checkpoint data structure.
    
    This is the core data structure that all checkpoint operations use,
    regardless of storage backend or checkpoint type.
    """
    # Core identification
    checkpoint_id: str
    level: CheckpointLevel
    status: CheckpointStatus = CheckpointStatus.CREATED
    
    # Data payload
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: CheckpointMetadata = field(default_factory=lambda: CheckpointMetadata(process_name="unknown"))
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: Optional[datetime] = None
    
    # Hierarchy
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.checkpoint_id:
            self.checkpoint_id = self.generate_id()
        
        # Update metadata with checkpoint info
        if self.metadata.parent_checkpoint_id is None:
            self.metadata.parent_checkpoint_id = self.parent_id
    
    @staticmethod
    def generate_id(prefix: str = "chk", level: Optional[CheckpointLevel] = None) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:8]
        
        if level:
            return f"{prefix}_{level.value}_{timestamp}_{unique_suffix}"
        return f"{prefix}_{timestamp}_{unique_suffix}"
    
    def mark_accessed(self) -> None:
        """Mark checkpoint as accessed."""
        self.accessed_at = datetime.now()
    
    def mark_updated(self) -> None:
        """Mark checkpoint as updated."""
        self.updated_at = datetime.now()
    
    def is_valid(self) -> bool:
        """Check if checkpoint is in valid state."""
        return self.status == CheckpointStatus.VALID
    
    def is_stale(self, max_age_hours: float = 24.0) -> bool:
        """Check if checkpoint is stale based on age."""
        if not self.created_at:
            return True
        
        age_hours = (datetime.now() - self.created_at).total_seconds() / 3600
        return age_hours > max_age_hours
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'level': self.level.value,
            'status': self.status.value,
            'data': self.data,
            'metadata': {
                'process_name': self.metadata.process_name,
                'process_version': self.metadata.process_version,
                'tags': self.metadata.tags,
                'description': self.metadata.description,
                'size_bytes': self.metadata.size_bytes,
                'compression': self.metadata.compression,
                'checksum': self.metadata.checksum,
                'save_duration_ms': self.metadata.save_duration_ms,
                'load_duration_ms': self.metadata.load_duration_ms,
                'parent_checkpoint_id': self.metadata.parent_checkpoint_id,
                'child_checkpoint_ids': self.metadata.child_checkpoint_ids,
            },
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat() if self.accessed_at else None,
            'parent_id': self.parent_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Create CheckpointData from dictionary."""
        metadata = CheckpointMetadata(
            process_name=data['metadata'].get('process_name', 'unknown'),
            process_version=data['metadata'].get('process_version', '1.0.0'),
            tags=data['metadata'].get('tags', {}),
            description=data['metadata'].get('description'),
            size_bytes=data['metadata'].get('size_bytes', 0),
            compression=data['metadata'].get('compression'),
            checksum=data['metadata'].get('checksum'),
            save_duration_ms=data['metadata'].get('save_duration_ms', 0.0),
            load_duration_ms=data['metadata'].get('load_duration_ms', 0.0),
            parent_checkpoint_id=data['metadata'].get('parent_checkpoint_id'),
            child_checkpoint_ids=data['metadata'].get('child_checkpoint_ids', []),
        )
        
        return cls(
            checkpoint_id=data['checkpoint_id'],
            level=CheckpointLevel(data['level']),
            status=CheckpointStatus(data['status']),
            data=data['data'],
            metadata=metadata,
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            accessed_at=datetime.fromisoformat(data['accessed_at']) if data['accessed_at'] else None,
            parent_id=data.get('parent_id'),
        )


@dataclass
class StorageConfig:
    """Configuration for checkpoint storage backends."""
    backend: StorageBackend
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Common configuration options
    base_path: Optional[Path] = None      # For file storage
    database_url: Optional[str] = None    # For database storage
    compression_enabled: bool = True      # Enable compression
    encryption_enabled: bool = False      # Enable encryption
    max_file_size_mb: float = 100.0      # Maximum checkpoint file size
    retention_days: int = 7               # How long to keep checkpoints
    
    def validate(self) -> List[str]:
        """Validate storage configuration."""
        errors = []
        
        if self.backend == StorageBackend.FILE:
            if not self.base_path:
                errors.append("File storage requires base_path")
        elif self.backend == StorageBackend.DATABASE:
            if not self.database_url:
                errors.append("Database storage requires database_url")
        
        if self.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")
        
        if self.retention_days < 0:
            errors.append("retention_days must be non-negative")
        
        return errors


@dataclass
class CheckpointFilter:
    """Filter criteria for querying checkpoints."""
    process_name: Optional[str] = None
    level: Optional[CheckpointLevel] = None
    status: Optional[CheckpointStatus] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None
    limit: Optional[int] = None
    
    def matches(self, checkpoint: CheckpointData) -> bool:
        """Check if checkpoint matches this filter."""
        if self.process_name and checkpoint.metadata.process_name != self.process_name:
            return False
        
        if self.level and checkpoint.level != self.level:
            return False
        
        if self.status and checkpoint.status != self.status:
            return False
        
        if self.created_after and checkpoint.created_at < self.created_after:
            return False
        
        if self.created_before and checkpoint.created_at > self.created_before:
            return False
        
        if self.parent_id and checkpoint.parent_id != self.parent_id:
            return False
        
        # Check tags
        for key, value in self.tags.items():
            if checkpoint.metadata.tags.get(key) != value:
                return False
        
        return True


# Exception classes for checkpoint operations
class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint is corrupted."""
    pass


class CheckpointValidationError(CheckpointError):
    """Raised when checkpoint validation fails."""
    pass


class StorageBackendError(CheckpointError):
    """Raised when storage backend operations fail."""
    pass
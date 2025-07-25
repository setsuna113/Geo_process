# src/core/checkpoint_manager.py
"""Central checkpoint management for the biodiversity pipeline."""

import json
import pickle
import logging
import threading
import time
import shutil
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import gzip

logger = logging.getLogger(__name__)


class CheckpointCorruptedError(Exception):
    """Raised when a checkpoint is corrupted or invalid."""
    pass


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    timestamp: float
    level: str  # 'pipeline', 'phase', 'step', 'substep'
    parent_id: Optional[str] = None
    status: str = "created"
    data_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    validation_checksum: Optional[str] = None
    compressed: bool = False
    compression_type: Optional[str] = None
    size_bytes: int = 0
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def file_size_bytes(self) -> int:
        """Alias for size_bytes for backward compatibility."""
        return self.size_bytes


class CheckpointManager:
    """
    Unified checkpointing system with hierarchical support.
    
    Features:
    - JSON metadata + binary data storage
    - Corruption detection
    - Automatic cleanup and retention
    - Compression support
    """
    
    _instance: Optional['CheckpointManager'] = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize checkpoint manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # Get checkpoint directory from config
            from src.config import config
            checkpoint_path = config.get('paths.checkpoint_dir', 'checkpoints')
            self._checkpoint_dir = Path(checkpoint_path)
            self._checkpoint_dir.mkdir(exist_ok=True)
            
            # Checkpoint registry
            self._checkpoints: Dict[str, CheckpointMetadata] = {}
            self._checkpoint_lock = threading.RLock()
            
            # Retention policy
            self._retention_config = {
                'max_checkpoints': 100,
                'max_age_days': 7,
                'min_keep_per_level': {
                    'pipeline': 5,
                    'phase': 3,
                    'step': 2,
                    'substep': 1
                }
            }
            
            # Cleanup thread
            self._cleanup_thread: Optional[threading.Thread] = None
            self._stop_cleanup = threading.Event()
            
            # Load existing checkpoints
            self._load_checkpoint_registry()
            
            # Start cleanup thread
            self._start_cleanup_thread()
            
            logger.info(f"Checkpoint manager initialized with directory: {self._checkpoint_dir}")
    
    def set_checkpoint_directory(self, directory: Union[str, Path]) -> None:
        """Set checkpoint directory."""
        self._checkpoint_dir = Path(directory)
        self._checkpoint_dir.mkdir(exist_ok=True)
        self._load_checkpoint_registry()
    
    def configure_retention(self, **config) -> None:
        """Configure retention policy."""
        self._retention_config.update(config)
    
    def save_checkpoint(self,
                       checkpoint_id: str,
                       data: Dict[str, Any],
                       level: str = "step",
                       parent_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       compress: bool = True) -> str:
        """
        Save a checkpoint.
        
        Args:
            checkpoint_id: Unique checkpoint identifier
            data: Data to checkpoint
            level: Checkpoint level in hierarchy
            parent_id: Parent checkpoint ID
            metadata: Additional metadata
            compress: Whether to compress data
            
        Returns:
            Checkpoint file path
        """
        with self._checkpoint_lock:
            # Create checkpoint directory
            checkpoint_path = self._checkpoint_dir / checkpoint_id
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save metadata
            checkpoint_meta = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=time.time(),
                level=level,
                parent_id=parent_id,
                metadata=metadata or {},
                compressed=compress,
                compression_type='gzip' if compress else None
            )
            
            # Save data files
            data_files = []
            total_size = 0
            
            for key, value in data.items():
                file_path = checkpoint_path / f"{key}.pkl"
                if compress:
                    file_path = file_path.with_suffix('.pkl.gz')
                
                # Save data
                try:
                    if compress:
                        with gzip.open(file_path, 'wb') as f:
                            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        with open(file_path, 'wb') as f:
                            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    data_files.append(str(file_path.relative_to(self._checkpoint_dir)))
                    total_size += file_path.stat().st_size
                    
                except Exception as e:
                    logger.error(f"Failed to save checkpoint data {key}: {e}")
                    # Clean up partial checkpoint
                    shutil.rmtree(checkpoint_path, ignore_errors=True)
                    raise
            
            checkpoint_meta.data_files = data_files
            checkpoint_meta.size_bytes = total_size
            checkpoint_meta.file_path = str(checkpoint_path)
            
            # Calculate checksum
            checkpoint_meta.checksum = self._calculate_checksum(checkpoint_path)
            
            # Save metadata
            meta_path = checkpoint_path / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(checkpoint_meta.to_dict(), f, indent=2)
            
            # Register checkpoint
            self._checkpoints[checkpoint_id] = checkpoint_meta
            self._save_checkpoint_registry()
            
            logger.info(f"Saved checkpoint {checkpoint_id} ({total_size / 1024 / 1024:.1f} MB)")
            
            return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to load
            
        Returns:
            Checkpoint data
            
        Raises:
            ValueError: If checkpoint not found or corrupted
        """
        with self._checkpoint_lock:
            if checkpoint_id not in self._checkpoints:
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            
            checkpoint_meta = self._checkpoints[checkpoint_id]
            checkpoint_path = self._checkpoint_dir / checkpoint_id
            
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint directory missing: {checkpoint_path}")
            
            # Verify checksum
            current_checksum = self._calculate_checksum(checkpoint_path)
            if current_checksum != checkpoint_meta.checksum:
                raise ValueError(f"Checkpoint corrupted: {checkpoint_id}")
            
            # Load data
            data = {}
            
            for data_file in checkpoint_meta.data_files:
                file_path = self._checkpoint_dir / data_file
                key = file_path.stem.replace('.pkl', '')
                
                try:
                    if checkpoint_meta.compressed:
                        with gzip.open(file_path, 'rb') as f:
                            data[key] = pickle.load(f)
                    else:
                        with open(file_path, 'rb') as f:
                            data[key] = pickle.load(f)
                            
                except Exception as e:
                    logger.error(f"Failed to load checkpoint data {key}: {e}")
                    raise
            
            logger.info(f"Loaded checkpoint {checkpoint_id}")
            
            return data
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        with self._checkpoint_lock:
            if checkpoint_id not in self._checkpoints:
                return False
            
            checkpoint_path = self._checkpoint_dir / checkpoint_id
            
            # Remove from registry
            del self._checkpoints[checkpoint_id]
            
            # Delete files
            try:
                shutil.rmtree(checkpoint_path)
                self._save_checkpoint_registry()
                logger.info(f"Deleted checkpoint {checkpoint_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
                return False
    
    def list_checkpoints(self, 
                        level: Optional[str] = None,
                        parent_id: Optional[str] = None,
                        processor_name: Optional[str] = None,
                        status: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        List checkpoints with optional filtering.
        
        Args:
            level: Filter by level
            parent_id: Filter by parent
            processor_name: Filter by processor name (stored in metadata)
            status: Filter by status
            
        Returns:
            List of checkpoint metadata
        """
        with self._checkpoint_lock:
            checkpoints = list(self._checkpoints.values())
            
            if level:
                checkpoints = [cp for cp in checkpoints if cp.level == level]
            
            if parent_id:
                checkpoints = [cp for cp in checkpoints if cp.parent_id == parent_id]
            
            if processor_name:
                checkpoints = [cp for cp in checkpoints if cp.metadata.get('processor_name') == processor_name]
                
            if status:
                checkpoints = [cp for cp in checkpoints if cp.status == status]
            
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
            
            return checkpoints
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get checkpoint metadata."""
        return self._checkpoints.get(checkpoint_id)
    
    def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_id: Checkpoint to validate
            
        Returns:
            True if valid
        """
        with self._checkpoint_lock:
            if checkpoint_id not in self._checkpoints:
                return False
            
            checkpoint_meta = self._checkpoints[checkpoint_id]
            checkpoint_path = self._checkpoint_dir / checkpoint_id
            
            # Check directory exists
            if not checkpoint_path.exists():
                checkpoint_meta.status = "corrupted"
                return False
            
            # Check all data files exist
            for data_file in checkpoint_meta.data_files:
                if not (self._checkpoint_dir / data_file).exists():
                    checkpoint_meta.status = "corrupted"
                    return False
            
            # Verify checksum
            current_checksum = self._calculate_checksum(checkpoint_path)
            is_valid = current_checksum == checkpoint_meta.checksum
            
            # Update status based on validation result
            if is_valid:
                checkpoint_meta.status = "valid"
                checkpoint_meta.validation_checksum = current_checksum
            else:
                checkpoint_meta.status = "corrupted"
            
            # Save updated metadata
            self._save_checkpoint_registry()
            
            return is_valid
    
    def cleanup_old_checkpoints(self) -> int:
        """
        Clean up old checkpoints based on retention policy.
        
        Returns:
            Number of checkpoints deleted
        """
        with self._checkpoint_lock:
            current_time = time.time()
            max_age_seconds = self._retention_config['max_age_days'] * 24 * 3600
            
            # Group by level
            by_level: Dict[str, List[CheckpointMetadata]] = {}
            for cp in self._checkpoints.values():
                by_level.setdefault(cp.level, []).append(cp)
            
            deleted_count = 0
            
            # Apply retention policy
            for level, checkpoints in by_level.items():
                # Sort by timestamp (newest first)
                checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
                
                min_keep = self._retention_config['min_keep_per_level'].get(level, 1)
                
                for i, cp in enumerate(checkpoints):
                    # Keep minimum number
                    if i < min_keep:
                        continue
                    
                    # Check age
                    age_seconds = current_time - cp.timestamp
                    if age_seconds > max_age_seconds:
                        if self.delete_checkpoint(cp.checkpoint_id):
                            deleted_count += 1
            
            # Check total limit
            if len(self._checkpoints) > self._retention_config['max_checkpoints']:
                # Delete oldest checkpoints
                all_checkpoints = sorted(
                    self._checkpoints.values(),
                    key=lambda x: x.timestamp
                )
                
                excess = len(all_checkpoints) - self._retention_config['max_checkpoints']
                for cp in all_checkpoints[:excess]:
                    if self.delete_checkpoint(cp.checkpoint_id):
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old checkpoints")
            
            return deleted_count
    
    def _calculate_checksum(self, checkpoint_path: Path) -> str:
        """Calculate checksum for checkpoint data files only (excluding metadata)."""
        hasher = hashlib.sha256()
        
        # Hash all files in checkpoint except metadata.json
        for file_path in sorted(checkpoint_path.rglob('*')):
            if file_path.is_file() and file_path.name != 'metadata.json':
                hasher.update(file_path.name.encode())
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _load_checkpoint_registry(self) -> None:
        """Load checkpoint registry from disk."""
        registry_path = self._checkpoint_dir / "registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                self._checkpoints = {
                    cp_id: CheckpointMetadata(**cp_data)
                    for cp_id, cp_data in registry_data.items()
                }
                
                logger.info(f"Loaded {len(self._checkpoints)} checkpoints from registry")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint registry: {e}")
                self._checkpoints = {}
        else:
            self._checkpoints = {}
    
    def _save_checkpoint_registry(self) -> None:
        """Save checkpoint registry to disk."""
        registry_path = self._checkpoint_dir / "registry.json"
        
        try:
            registry_data = {
                cp_id: cp.to_dict()
                for cp_id, cp in self._checkpoints.items()
            }
            
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint registry: {e}")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        def cleanup_worker():
            while not self._stop_cleanup.is_set():
                try:
                    # Run cleanup every hour
                    if self._stop_cleanup.wait(3600):
                        break
                    
                    self.cleanup_old_checkpoints()
                    
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def stop(self) -> None:
        """Stop checkpoint manager."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager
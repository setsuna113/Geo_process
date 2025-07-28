# src/base/checkpointable.py
"""Mixin classes and interfaces for processes that support checkpointing.

This module provides the core abstractions that any process can use to add
checkpoint functionality without needing to implement the details directly.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta

from src.foundations.types.checkpoint_types import (
    CheckpointData, CheckpointLevel, CheckpointStatus, CheckpointMetadata,
    CheckpointFilter, CheckpointError
)
from .checkpoint import CheckpointValidator, DefaultCheckpointValidator

logger = logging.getLogger(__name__)


@dataclass
class CheckpointPolicy:
    """Configuration for checkpoint timing and retention rules."""
    
    # Timing policies
    interval_items: int = 100           # Checkpoint every N items processed
    interval_seconds: float = 300.0     # Checkpoint every N seconds
    interval_memory_mb: float = 500.0   # Checkpoint when memory usage increases by N MB
    
    # Retention policies
    max_checkpoints: int = 5            # Keep only N most recent checkpoints
    max_age_hours: float = 24.0         # Remove checkpoints older than N hours
    keep_failed_checkpoints: bool = True # Whether to keep checkpoints from failed runs
    
    # Validation policies
    validate_on_save: bool = True       # Validate checkpoints when saving
    verify_on_load: bool = True         # Verify checksums when loading
    auto_cleanup: bool = True           # Automatically clean up old checkpoints
    
    # Performance policies
    compress_large_checkpoints: bool = True  # Compress checkpoints > threshold
    compression_threshold_mb: float = 10.0   # Size threshold for compression
    async_save: bool = False            # Save checkpoints asynchronously
    
    def should_checkpoint_by_items(self, items_processed: int, last_checkpoint_items: int) -> bool:
        """Check if should checkpoint based on items processed."""
        return (items_processed - last_checkpoint_items) >= self.interval_items
    
    def should_checkpoint_by_time(self, last_checkpoint_time: float) -> bool:
        """Check if should checkpoint based on time elapsed."""
        return (time.time() - last_checkpoint_time) >= self.interval_seconds
    
    def should_checkpoint_by_memory(self, memory_increase_mb: float) -> bool:
        """Check if should checkpoint based on memory usage."""
        return memory_increase_mb >= self.interval_memory_mb


class ResumableProcess(ABC):
    """Interface for processes that can be resumed from checkpoints."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current process state for checkpointing.
        
        Returns:
            Dictionary containing all state needed to resume the process
        """
        pass
    
    @abstractmethod
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore process state from checkpoint data.
        
        Args:
            state: State dictionary from checkpoint
        """
        pass
    
    @abstractmethod
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary with progress metrics (items_processed, percentage, etc.)
        """
        pass
    
    def can_resume_from(self, checkpoint: CheckpointData) -> bool:
        """Check if this process can resume from the given checkpoint.
        
        Args:
            checkpoint: CheckpointData to check compatibility with
            
        Returns:
            True if can resume from this checkpoint, False otherwise
        """
        # Default implementation - check process name match
        return checkpoint.metadata.process_name == self.__class__.__name__


class CheckpointableProcess:
    """Mixin class for any process that needs checkpointing capability.
    
    This mixin provides a unified interface for checkpoint operations that can be
    added to any process class without requiring inheritance from a specific base class.
    """
    
    def __init__(self, 
                 checkpoint_policy: Optional[CheckpointPolicy] = None,
                 checkpoint_validator: Optional[CheckpointValidator] = None,
                 process_id: Optional[str] = None,
                 **kwargs):
        """Initialize checkpointing capabilities.
        
        Args:
            checkpoint_policy: Policy for checkpoint timing and retention
            checkpoint_validator: Validator for checkpoint data integrity
            process_id: Unique identifier for this process instance
            **kwargs: Additional arguments passed to super().__init__()
        """
        # Call super().__init__() if this is part of multiple inheritance
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        
        # Checkpoint configuration
        self._checkpoint_policy = checkpoint_policy or CheckpointPolicy()
        self._checkpoint_validator = checkpoint_validator or DefaultCheckpointValidator()
        self._process_id = process_id or self._generate_process_id()
        
        # Checkpoint state tracking
        self._last_checkpoint_time = time.time()
        self._last_checkpoint_items = 0
        self._last_checkpoint_memory_mb = 0.0
        self._checkpoint_count = 0
        self._current_checkpoint_id: Optional[str] = None
        
        # Checkpoint storage (will be set by checkpoint manager)
        self._checkpoint_manager = None
        
        # Progress tracking
        self._items_processed = 0
        self._total_items: Optional[int] = None
        self._start_time = time.time()
        
        logger.debug(f"Initialized checkpointable process: {self._process_id}")
    
    def set_checkpoint_manager(self, checkpoint_manager):
        """Set the checkpoint manager for this process.
        
        Args:
            checkpoint_manager: CheckpointManager instance to use
        """
        self._checkpoint_manager = checkpoint_manager
    
    def save_checkpoint(self, 
                       level: CheckpointLevel = CheckpointLevel.STEP,
                       data: Optional[Dict[str, Any]] = None,
                       description: Optional[str] = None,
                       tags: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Save a checkpoint for this process.
        
        Args:
            level: Checkpoint level (PIPELINE, STAGE, STEP, SUBSTEP)
            data: Custom data to include in checkpoint (merged with process state)
            description: Human-readable description of checkpoint
            tags: Additional tags for checkpoint metadata
            
        Returns:
            Checkpoint ID if successful, None if failed
        """
        if not self._checkpoint_manager:
            logger.warning(f"No checkpoint manager set for process {self._process_id}")
            return None
        
        try:
            # Get process state if this is a ResumableProcess
            checkpoint_data = data or {}
            if isinstance(self, ResumableProcess):
                checkpoint_data.update(self.get_state())
            
            # Add progress information
            checkpoint_data.update({
                'items_processed': self._items_processed,
                'total_items': self._total_items,
                'start_time': self._start_time,
                'checkpoint_time': time.time(),
                'progress_percentage': self.get_progress_percentage(),
            })
            
            # Create metadata
            metadata = CheckpointMetadata(
                process_name=self.__class__.__name__,
                description=description or f"Checkpoint at {self._items_processed} items",
                tags=tags or {},
            )
            
            # Create checkpoint
            checkpoint = CheckpointData(
                checkpoint_id="",  # Will be generated
                level=level,
                data=checkpoint_data,
                metadata=metadata,
            )
            
            # Validate if required
            if self._checkpoint_policy.validate_on_save:
                if not self._checkpoint_validator.validate(checkpoint):
                    logger.error(f"Checkpoint validation failed for {self._process_id}")
                    return None
            
            # Calculate checksum
            checkpoint.metadata.checksum = self._checkpoint_validator.calculate_checksum(checkpoint_data)
            
            # Save through manager
            checkpoint_id = self._checkpoint_manager.save(checkpoint)
            
            # Update tracking
            self._last_checkpoint_time = time.time()
            self._last_checkpoint_items = self._items_processed
            self._checkpoint_count += 1
            self._current_checkpoint_id = checkpoint_id
            
            logger.info(f"Saved checkpoint {checkpoint_id} for {self._process_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {self._process_id}: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> Optional[CheckpointData]:
        """Load a checkpoint for this process.
        
        Args:
            checkpoint_id: Specific checkpoint ID to load, or None for latest
            
        Returns:
            CheckpointData if successful, None if failed
        """
        if not self._checkpoint_manager:
            logger.warning(f"No checkpoint manager set for process {self._process_id}")
            return None
        
        try:
            # Get checkpoint
            if checkpoint_id:
                checkpoint = self._checkpoint_manager.load(checkpoint_id)
            else:
                checkpoint = self._checkpoint_manager.load_latest_for_process(self._process_id)
            
            if not checkpoint:
                logger.info(f"No checkpoint found for {self._process_id}")
                return None
            
            # Verify if required
            if self._checkpoint_policy.verify_on_load:
                if not self._checkpoint_validator.verify_integrity(checkpoint):
                    logger.error(f"Checkpoint integrity check failed: {checkpoint.checkpoint_id}")
                    return None
            
            # Restore state if this is a ResumableProcess
            if isinstance(self, ResumableProcess):
                self.restore_state(checkpoint.data)
            
            # Restore progress tracking
            self._items_processed = checkpoint.data.get('items_processed', 0)
            self._total_items = checkpoint.data.get('total_items')
            self._start_time = checkpoint.data.get('start_time', time.time())
            self._last_checkpoint_time = checkpoint.data.get('checkpoint_time', time.time())
            self._last_checkpoint_items = self._items_processed
            self._current_checkpoint_id = checkpoint.checkpoint_id
            
            logger.info(f"Loaded checkpoint {checkpoint.checkpoint_id} for {self._process_id}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {self._process_id}: {e}")
            return None
    
    def should_checkpoint(self, force_check: bool = False) -> bool:
        """Check if a checkpoint should be saved based on current policy.
        
        Args:
            force_check: Force policy evaluation even if recently checked
            
        Returns:
            True if checkpoint should be saved, False otherwise
        """
        policy = self._checkpoint_policy
        
        # Check by items processed
        if policy.should_checkpoint_by_items(self._items_processed, self._last_checkpoint_items):
            return True
        
        # Check by time elapsed
        if policy.should_checkpoint_by_time(self._last_checkpoint_time):
            return True
        
        # Check by memory usage (requires implementation in subclass)
        if hasattr(self, 'get_memory_usage_mb'):
            current_memory = self.get_memory_usage_mb()
            memory_increase = current_memory - self._last_checkpoint_memory_mb
            if policy.should_checkpoint_by_memory(memory_increase):
                self._last_checkpoint_memory_mb = current_memory
                return True
        
        return False
    
    def auto_checkpoint(self, force: bool = False) -> Optional[str]:
        """Automatically save checkpoint if policy conditions are met.
        
        Args:
            force: Force checkpoint regardless of policy
            
        Returns:
            Checkpoint ID if checkpoint was saved, None otherwise
        """
        if force or self.should_checkpoint():
            return self.save_checkpoint(
                level=CheckpointLevel.STEP,
                description=f"Auto checkpoint at {self._items_processed} items"
            )
        return None
    
    def update_progress(self, items_processed: Optional[int] = None, total_items: Optional[int] = None):
        """Update progress tracking.
        
        Args:
            items_processed: Current number of items processed
            total_items: Total number of items to process
        """
        if items_processed is not None:
            self._items_processed = items_processed
        
        if total_items is not None:
            self._total_items = total_items
        
        # Auto-checkpoint if conditions are met
        self.auto_checkpoint()
    
    def get_progress_percentage(self) -> float:
        """Get current progress as percentage.
        
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if self._total_items and self._total_items > 0:
            return min(100.0, (self._items_processed / self._total_items) * 100.0)
        return 0.0
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about checkpoint state.
        
        Returns:
            Dictionary with checkpoint status information
        """
        return {
            'process_id': self._process_id,
            'current_checkpoint_id': self._current_checkpoint_id,
            'checkpoint_count': self._checkpoint_count,
            'last_checkpoint_time': self._last_checkpoint_time,
            'last_checkpoint_items': self._last_checkpoint_items,
            'items_processed': self._items_processed,
            'total_items': self._total_items,
            'progress_percentage': self.get_progress_percentage(),
            'policy': {
                'interval_items': self._checkpoint_policy.interval_items,
                'interval_seconds': self._checkpoint_policy.interval_seconds,
                'max_checkpoints': self._checkpoint_policy.max_checkpoints,
            }
        }
    
    def cleanup_checkpoints(self) -> int:
        """Clean up old checkpoints according to retention policy.
        
        Returns:
            Number of checkpoints removed
        """
        if not self._checkpoint_manager or not self._checkpoint_policy.auto_cleanup:
            return 0
        
        try:
            # Create filter for this process
            filter_criteria = CheckpointFilter(
                process_name=self.__class__.__name__,
                created_before=datetime.now() - timedelta(hours=self._checkpoint_policy.max_age_hours)
            )
            
            # Get checkpoints to clean up
            old_checkpoints = self._checkpoint_manager.list(filter_criteria)
            
            # Sort by creation time (oldest first)
            old_checkpoints.sort(key=lambda cp: cp.created_at)
            
            # Keep the most recent N checkpoints
            max_keep = self._checkpoint_policy.max_checkpoints
            checkpoints_to_remove = old_checkpoints[:-max_keep] if len(old_checkpoints) > max_keep else []
            
            # Remove old checkpoints
            removed_count = 0
            for checkpoint in checkpoints_to_remove:
                if self._checkpoint_manager.delete(checkpoint.checkpoint_id):
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old checkpoints for {self._process_id}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints for {self._process_id}: {e}")
            return 0
    
    def _generate_process_id(self) -> str:
        """Generate a unique process ID."""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.__class__.__name__}_{timestamp}_{str(uuid.uuid4())[:8]}"


class SimpleCheckpointableProcess(CheckpointableProcess, ResumableProcess):
    """Simple implementation of a checkpointable and resumable process.
    
    This is a concrete implementation that can be used directly or as an example
    for implementing custom checkpointable processes.
    """
    
    def __init__(self, **kwargs):
        """Initialize simple checkpointable process."""
        super().__init__(**kwargs)
        self._custom_state: Dict[str, Any] = {}
    
    def get_state(self) -> Dict[str, Any]:
        """Get current process state."""
        return {
            'custom_state': self._custom_state.copy(),
            'class_name': self.__class__.__name__,
            'process_id': self._process_id,
        }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore process state from checkpoint."""
        self._custom_state = state.get('custom_state', {})
        logger.debug(f"Restored state for {self._process_id}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        return {
            'items_processed': self._items_processed,
            'total_items': self._total_items,
            'progress_percentage': self.get_progress_percentage(),
            'elapsed_time_seconds': time.time() - self._start_time,
            'custom_state_keys': list(self._custom_state.keys()),
        }
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a custom state value."""
        self._custom_state[key] = value
    
    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get a custom state value."""
        return self._custom_state.get(key, default)


# Utility functions for checkpoint integration

def make_checkpointable(checkpoint_policy: Optional[CheckpointPolicy] = None):
    """Class decorator to add checkpoint capabilities to an existing class.
    
    This creates a new class that inherits from both the original class and
    CheckpointableProcess, avoiding __bases__ modification issues.
    
    Args:
        checkpoint_policy: Optional checkpoint policy to use
        
    Returns:
        Decorator function that creates a checkpointable version of the class
    """
    def decorator(cls):
        # Create a new class that inherits from both the original and CheckpointableProcess
        class CheckpointableVersion(cls, CheckpointableProcess):
            def __init__(self, *args, **kwargs):
                # Extract checkpoint-specific kwargs
                checkpoint_kwargs = {
                    'checkpoint_policy': kwargs.pop('checkpoint_policy', checkpoint_policy),
                    'checkpoint_validator': kwargs.pop('checkpoint_validator', None),
                    'process_id': kwargs.pop('process_id', None),
                }
                
                # Initialize both parent classes
                cls.__init__(self, *args, **kwargs)
                CheckpointableProcess.__init__(self, **checkpoint_kwargs)
        
        # Preserve the original class name and module
        CheckpointableVersion.__name__ = cls.__name__
        CheckpointableVersion.__qualname__ = cls.__qualname__
        CheckpointableVersion.__module__ = cls.__module__
        
        return CheckpointableVersion
    
    return decorator


def checkpoint_on_interval(items: int = 100, seconds: float = 300.0):
    """Method decorator to automatically checkpoint at specified intervals.
    
    Args:
        items: Checkpoint every N items processed
        seconds: Checkpoint every N seconds
        
    Returns:
        Decorated method that auto-checkpoints
    """
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            # Check if this is a checkpointable process
            if not isinstance(self, CheckpointableProcess):
                return method(self, *args, **kwargs)
            
            # Update checkpoint policy temporarily
            original_policy = self._checkpoint_policy
            self._checkpoint_policy.interval_items = items
            self._checkpoint_policy.interval_seconds = seconds
            
            try:
                # Execute method
                result = method(self, *args, **kwargs)
                
                # Auto-checkpoint if needed
                self.auto_checkpoint()
                
                return result
            finally:
                # Restore original policy
                self._checkpoint_policy = original_policy
        
        return wrapper
    return decorator
# src/infrastructure/processors/base_processor.py
"""Enhanced base processor implementation with proper architecture."""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from src.foundations.interfaces.processor import IProcessor, ICheckpointableProcessor, IMemoryAwareProcessor

logger = logging.getLogger(__name__)

class EnhancedBaseProcessor(ICheckpointableProcessor, IMemoryAwareProcessor):
    """
    Enhanced base processor implementation following the new architecture.
    
    This provides a clean implementation of the processor interfaces without
    violating dependency inversion principles.
    """
    
    def __init__(self, 
                 batch_size: int = 1000,
                 max_workers: Optional[int] = None,
                 memory_limit_mb: Optional[int] = 1024,
                 checkpoint_interval: int = 100,
                 **kwargs):
        """
        Initialize enhanced base processor.
        
        Args:
            batch_size: Size of batches for processing
            max_workers: Maximum parallel workers
            memory_limit_mb: Memory limit in MB
            checkpoint_interval: Items between checkpoints
            **kwargs: Additional processor-specific parameters
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or 4
        self.memory_limit_mb = memory_limit_mb
        self.checkpoint_interval = checkpoint_interval
        
        # Processing state
        self._items_processed = 0
        self._checkpoint_data = {}
        self._config_requirements = {}
        
        # Memory tracking
        self._memory_usage = 0
    
    # === IProcessor Implementation ===
    
    def process_single(self, item: Any) -> Any:
        """Process a single item - must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement process_single")
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input - can be overridden by subclasses."""
        if item is None:
            return False, "Input cannot be None"
        return True, None
    
    def get_config_requirements(self) -> Dict[str, Any]:
        """Get configuration requirements."""
        return {
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'memory_limit_mb': self.memory_limit_mb,
            'checkpoint_interval': self.checkpoint_interval
        }
    
    # === ICheckpointableProcessor Implementation ===
    
    def should_checkpoint(self) -> bool:
        """Determine if checkpoint is needed."""
        return (self._items_processed % self.checkpoint_interval) == 0
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data to checkpoint."""
        return {
            'items_processed': self._items_processed,
            'memory_usage': self._memory_usage,
            'config': self.get_config_requirements()
        }
    
    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Restore from checkpoint data."""
        self._items_processed = checkpoint_data.get('items_processed', 0)
        self._memory_usage = checkpoint_data.get('memory_usage', 0)
        logger.info(f"Restored from checkpoint: {self._items_processed} items processed")
    
    # === IMemoryAwareProcessor Implementation ===
    
    def estimate_memory_usage(self, input_size: int) -> int:
        """Estimate memory needs."""
        # Simple estimation - 2x input size plus overhead
        overhead_mb = 100  # Base overhead
        data_mb = (input_size * 2) // (1024 * 1024)
        return overhead_mb + data_mb
    
    def get_memory_usage(self) -> int:
        """Get current memory usage."""
        return self._memory_usage
    
    # === Helper Methods ===
    
    def process_batch(self, items):
        """Process a batch of items."""
        results = []
        for item in items:
            valid, error = self.validate_input(item)
            if valid:
                try:
                    result = self.process_single(item)
                    results.append(result)
                    self._items_processed += 1
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    results.append(None)
            else:
                logger.warning(f"Invalid input: {error}")
                results.append(None)
        
        return results
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .processor import IProcessor


class ICheckpointableProcessor(IProcessor):
    """Processor that can be checkpointed."""
    
    @abstractmethod
    def should_checkpoint(self) -> bool:
        """Determine if checkpoint is needed."""
        pass
    
    @abstractmethod
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data to checkpoint."""
        pass


class IMemoryAwareProcessor(IProcessor):
    """Processor that tracks memory usage."""
    
    @abstractmethod
    def estimate_memory_usage(self, input_size: int) -> int:
        """Estimate memory needs."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """Get current memory usage."""
        pass
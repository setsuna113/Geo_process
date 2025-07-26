# src/foundations/interfaces/processor.py
"""Pure processor interfaces - NO IMPLEMENTATIONS!"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Dict, List, Union
import numpy as np

class IProcessor(ABC):
    """Pure processor interface - NO IMPLEMENTATIONS!"""
    
    @abstractmethod
    def process_single(self, item: Any) -> Any:
        """Process a single item."""
        pass
    
    @abstractmethod
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input before processing."""
        pass
    
    @abstractmethod
    def get_config_requirements(self) -> Dict[str, Any]:
        """Declare configuration requirements."""
        pass

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
    
    @abstractmethod
    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Restore from checkpoint data."""
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

class ITileProcessor(IProcessor):
    """Processor that works with spatial tiles."""
    
    @abstractmethod
    def process_tile(self, tile_data: Any, tile_info: Dict[str, Any]) -> Any:
        """Process a single tile."""
        pass
    
    @abstractmethod
    def validate_tile(self, tile_data: Any, tile_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate tile data."""
        pass
    
    @abstractmethod
    def get_tile_requirements(self) -> Dict[str, Any]:
        """Get tile processing requirements."""
        pass
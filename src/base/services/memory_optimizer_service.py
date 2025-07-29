"""Memory optimization and chunk sizing service."""

import math
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ...config.processing_config import ProcessingConfig, ChunkInfo

logger = logging.getLogger(__name__)


@dataclass
class OptimalChunkConfig:
    """Configuration for optimal chunk sizing."""
    chunk_size: int
    num_chunks: int
    memory_per_chunk_mb: float
    total_memory_estimate_mb: float
    safety_factor: float


class MemoryOptimizerService:
    """Service for memory optimization and chunk size calculation."""
    
    def __init__(self):
        self._config = {
            'target_memory_usage_percent': 70.0,
            'safety_factor': 0.8,
            'min_chunk_size': 1000,
            'max_chunk_size': 1000000,
            'memory_overhead_factor': 1.5  # Account for processing overhead
        }
    
    def calculate_optimal_chunk_size(self, total_items: int, 
                                   memory_per_item_bytes: float,
                                   available_memory_mb: Optional[float] = None,
                                   target_memory_mb: Optional[float] = None) -> OptimalChunkConfig:
        """Calculate optimal chunk size for processing."""
        import psutil
        
        # Determine target memory
        if target_memory_mb is None:
            if available_memory_mb is None:
                memory = psutil.virtual_memory()
                available_memory_mb = memory.available / (1024**2)
            
            target_memory_mb = available_memory_mb * (self._config['target_memory_usage_percent'] / 100.0)
        
        # Apply safety factor
        safe_memory_mb = target_memory_mb * self._config['safety_factor']
        
        # Calculate memory per item in MB
        memory_per_item_mb = (memory_per_item_bytes / (1024**2)) * self._config['memory_overhead_factor']
        
        # Calculate optimal chunk size
        if memory_per_item_mb <= 0:
            chunk_size = min(total_items, self._config['max_chunk_size'])
        else:
            chunk_size = int(safe_memory_mb / memory_per_item_mb)
            chunk_size = max(self._config['min_chunk_size'], 
                           min(chunk_size, self._config['max_chunk_size']))
        
        # Ensure chunk size doesn't exceed total items
        chunk_size = min(chunk_size, total_items)
        
        # Calculate number of chunks
        num_chunks = math.ceil(total_items / chunk_size)
        
        # Calculate actual memory usage
        memory_per_chunk_mb = chunk_size * memory_per_item_mb
        total_memory_estimate_mb = memory_per_chunk_mb  # Only one chunk in memory at a time
        
        config = OptimalChunkConfig(
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            memory_per_chunk_mb=memory_per_chunk_mb,
            total_memory_estimate_mb=total_memory_estimate_mb,
            safety_factor=self._config['safety_factor']
        )
        
        logger.debug(
            f"Optimal chunking: {total_items} items → "
            f"{num_chunks} chunks of {chunk_size} items "
            f"({memory_per_chunk_mb:.1f}MB per chunk)"
        )
        
        return config
    
    def create_processing_config(self, total_items: int, 
                               memory_per_item_bytes: float,
                               processing_name: str = "processing",
                               target_memory_mb: Optional[float] = None) -> ProcessingConfig:
        """Create a processing configuration with optimal chunking."""
        chunk_config = self.calculate_optimal_chunk_size(
            total_items, memory_per_item_bytes, target_memory_mb=target_memory_mb
        )
        
        # Create chunk info objects
        chunks = []
        for i in range(chunk_config.num_chunks):
            start_idx = i * chunk_config.chunk_size
            end_idx = min(start_idx + chunk_config.chunk_size, total_items)
            
            chunk_info = ChunkInfo(
                index=i,
                start_item=start_idx,
                end_item=end_idx,
                size=end_idx - start_idx,
                estimated_memory_mb=chunk_config.memory_per_chunk_mb
            )
            chunks.append(chunk_info)
        
        return ProcessingConfig(
            name=processing_name,
            total_items=total_items,
            chunk_size=chunk_config.chunk_size,
            chunks=chunks,
            memory_estimate_mb=chunk_config.total_memory_estimate_mb,
            metadata={
                'safety_factor': chunk_config.safety_factor,
                'memory_per_item_bytes': memory_per_item_bytes,
                'target_memory_mb': target_memory_mb
            }
        )
    
    def estimate_memory_requirement(self, items: int, bytes_per_item: float,
                                  processing_overhead: float = 1.5) -> Dict[str, float]:
        """Estimate memory requirements for processing."""
        base_memory_mb = (items * bytes_per_item) / (1024**2)
        total_memory_mb = base_memory_mb * processing_overhead
        
        return {
            'base_memory_mb': base_memory_mb,
            'overhead_factor': processing_overhead,
            'total_memory_mb': total_memory_mb,
            'recommended_available_mb': total_memory_mb / self._config['safety_factor']
        }
    
    def configure(self, **config) -> None:
        """Update configuration."""
        self._config.update(config)
        logger.debug(f"Updated memory optimizer config: {config}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()
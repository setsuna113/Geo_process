"""Base processor class with memory tracking and batch processing capabilities."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union, Callable
from contextlib import contextmanager
from pathlib import Path
import logging
import time
import tracemalloc
import psutil
import json
from dataclasses import dataclass, asdict
from datetime import datetime

from ..core.registry import component_registry
from ..config import config
from ..database.schema import schema

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Standard result format for all processors."""
    success: bool
    items_processed: int
    items_failed: int
    elapsed_time: float
    memory_used_mb: float
    results: Optional[List[Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        return (f"Processed: {self.items_processed}, "
                f"Failed: {self.items_failed}, "
                f"Time: {self.elapsed_time:.2f}s, "
                f"Memory: {self.memory_used_mb:.1f}MB")


class MemoryTracker:
    """Track memory usage across operations."""
    
    def __init__(self):
        self.measurements: List[Dict[str, Any]] = []
        self.current_operation: Optional[str] = None
        self.start_memory: Optional[int] = None
        
    def start(self, operation: str):
        """Start tracking an operation."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            
        self.current_operation = operation
        self.start_memory = tracemalloc.get_traced_memory()[0]
        
    def stop(self) -> float:
        """Stop tracking and return memory used in MB."""
        if self.start_memory is None:
            return 0.0
            
        end_memory = tracemalloc.get_traced_memory()[0]
        memory_used = (end_memory - self.start_memory) / 1024 / 1024
        
        self.measurements.append({
            'operation': self.current_operation,
            'memory_mb': memory_used,
            'timestamp': datetime.now().isoformat()
        })
        
        self.current_operation = None
        self.start_memory = None
        
        return memory_used
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.measurements:
            return {'total_mb': 0, 'operations': []}
            
        total_mb = sum(m['memory_mb'] for m in self.measurements)
        return {
            'total_mb': total_mb,
            'peak_mb': max(m['memory_mb'] for m in self.measurements),
            'operations': self.measurements
        }


class BaseProcessor(ABC):
    """
    Base class for all processors in the biodiversity pipeline.
    
    Handles:
    - Memory tracking
    - Batch processing
    - Progress reporting
    - Error handling
    - Result storage
    """
    
    def __init__(self, 
                 batch_size: int = 1000,
                 max_workers: Optional[int] = None,
                 store_results: bool = True,
                 **kwargs):
        """
        Initialize base processor.
        
        Args:
            batch_size: Size of batches for processing
            max_workers: Maximum parallel workers (None for auto)
            store_results: Whether to store results in database
            **kwargs: Additional processor-specific parameters
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or (psutil.cpu_count() or 4) - 1
        self.store_results = store_results
        self.memory_tracker = MemoryTracker()
        self.config = self._merge_config(kwargs)
        
        # For progress tracking
        self._progress_callback = None
        self._job_id = None
        
    def _merge_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge kwargs with default config."""
        processor_name = self.__class__.__name__
        default_config = config.get(f'processors.{processor_name}', {})
        return {**default_config, **kwargs}
    
    @contextmanager
    def track_memory(self, operation: str):
        """Context manager for memory tracking."""
        self.memory_tracker.start(operation)
        try:
            yield
        finally:
            memory_used = self.memory_tracker.stop()
            logger.info(f"{operation}: {memory_used:.1f}MB")
    
    @abstractmethod
    def process_single(self, item: Any) -> Any:
        """
        Process a single item.
        
        Args:
            item: Item to process
            
        Returns:
            Processed result or None if failed
        """
        pass
    
    @abstractmethod
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate input item.
        
        Args:
            item: Item to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    def preprocess_batch(self, items: List[Any]) -> List[Any]:
        """
        Preprocess a batch of items (optional override).
        
        Args:
            items: Items to preprocess
            
        Returns:
            Preprocessed items
        """
        return items
    
    def postprocess_batch(self, results: List[Any]) -> List[Any]:
        """
        Postprocess batch results (optional override).
        
        Args:
            results: Results to postprocess
            
        Returns:
            Postprocessed results
        """
        return results
    
    def process_batch(self, items: List[Any]) -> ProcessingResult:
        """
        Process a batch of items with full tracking.
        
        Args:
            items: Items to process
            
        Returns:
            ProcessingResult with details
        """
        start_time = time.time()
        processed_items = []
        failed_items = []
        errors = []
        
        with self.track_memory(f"Batch processing ({len(items)} items)"):
            # Validate inputs
            valid_items = []
            for i, item in enumerate(items):
                is_valid, error_msg = self.validate_input(item)
                if is_valid:
                    valid_items.append(item)
                else:
                    failed_items.append(item)
                    errors.append({
                        'index': i,
                        'error': error_msg or 'Validation failed',
                        'item': str(item)[:100]  # Truncate for logging
                    })
            
            if not valid_items:
                return ProcessingResult(
                    success=False,
                    items_processed=0,
                    items_failed=len(items),
                    elapsed_time=time.time() - start_time,
                    memory_used_mb=0,
                    errors=errors
                )
            
            # Preprocess
            valid_items = self.preprocess_batch(valid_items)
            
            # Process items
            for item in valid_items:
                try:
                    result = self.process_single(item)
                    if result is not None:
                        processed_items.append(result)
                    else:
                        failed_items.append(item)
                        
                except Exception as e:
                    failed_items.append(item)
                    errors.append({
                        'error': str(e),
                        'item': str(item)[:100]
                    })
                    logger.error(f"Processing error: {e}")
            
            # Postprocess
            if processed_items:
                processed_items = self.postprocess_batch(processed_items)
                
                # Store results if configured
                if self.store_results:
                    self._store_results(processed_items)
        
        # Get memory summary
        memory_summary = self.memory_tracker.get_summary()
        
        return ProcessingResult(
            success=len(processed_items) > 0,
            items_processed=len(processed_items),
            items_failed=len(failed_items),
            elapsed_time=time.time() - start_time,
            memory_used_mb=memory_summary.get('total_mb', 0),
            results=processed_items if not self.store_results else None,
            errors=errors if errors else None,
            metadata={'memory_details': memory_summary}
        )
    
    def process_iterator(self, 
                        iterator: Iterator[Any],
                        total: Optional[int] = None) -> ProcessingResult:
        """
        Process items from an iterator with batching.
        
        Args:
            iterator: Iterator of items
            total: Total number of items (for progress)
            
        Returns:
            Combined ProcessingResult
        """
        total_processed = 0
        total_failed = 0
        all_errors = []
        total_time = 0
        total_memory = 0
        
        batch = []
        batch_num = 0
        
        for item in iterator:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                batch_num += 1
                logger.info(f"Processing batch {batch_num}")
                
                result = self.process_batch(batch)
                total_processed += result.items_processed
                total_failed += result.items_failed
                total_time += result.elapsed_time
                total_memory += result.memory_used_mb
                
                if result.errors:
                    all_errors.extend(result.errors)
                
                # Update progress
                if self._progress_callback and total:
                    progress = (total_processed + total_failed) / total * 100
                    self._progress_callback(progress)
                
                batch = []
        
        # Process remaining items
        if batch:
            result = self.process_batch(batch)
            total_processed += result.items_processed
            total_failed += result.items_failed
            total_time += result.elapsed_time
            total_memory += result.memory_used_mb
            
        return ProcessingResult(
            success=total_processed > 0,
            items_processed=total_processed,
            items_failed=total_failed,
            elapsed_time=total_time,
            memory_used_mb=total_memory,
            errors=all_errors if all_errors else None
        )
    
    def _store_results(self, results: List[Any]):
        """Store results in database (override in subclasses)."""
        logger.warning(f"{self.__class__.__name__} does not implement _store_results")
    
    def set_progress_callback(self, callback: Callable[[float], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def set_job_id(self, job_id: str):
        """Set job ID for tracking."""
        self._job_id = job_id
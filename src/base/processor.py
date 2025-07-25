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
import mmap
import signal
import threading
from dataclasses import dataclass, asdict
from datetime import datetime

from ..core.registry import component_registry
from src.config import config as global_config
from ..database.schema import schema
from .memory_tracker import get_memory_tracker
from ..core.progress_manager import get_progress_manager
from ..core.progress_events import get_event_bus
from ..core.checkpoint_manager import get_checkpoint_manager
from ..core.signal_handler import get_signal_handler

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


class LegacyMemoryTracker:
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
    Enhanced base class for all processors in the biodiversity pipeline.
    
    New Features:
    - Memory-mapped file tracking
    - GDAL cache usage tracking separately  
    - Tile-based progress tracking
    - Support for partial processing (specific regions)
    - estimate_memory_usage() method
    - supports_chunking property
    - Enhanced memory management with pressure callbacks
    """
    
    def __init__(self, 
                    batch_size: int = 1000,
                    max_workers: Optional[int] = None,
                    store_results: bool = True,
                    memory_limit_mb: Optional[int] = None,
                    tile_size: Optional[int] = None,
                    supports_chunking: bool = True,
                    config=None,
                    # New parameters
                    enable_progress: bool = True,
                    enable_checkpoints: bool = True,
                    checkpoint_interval: int = 100,
                    timeout_seconds: Optional[float] = None,
                    **kwargs):
        """
        Initialize enhanced base processor.
        
        Args:
            batch_size: Size of batches for processing
            max_workers: Maximum parallel workers (None for auto)
            store_results: Whether to store results in database
            memory_limit_mb: Memory limit from processor config
            tile_size: Tile size for tile-based processing
            supports_chunking: Whether processor supports chunking
            enable_progress: Enable progress tracking
            enable_checkpoints: Enable checkpoint support
            checkpoint_interval: Items between checkpoints
            timeout_seconds: Processing timeout
            **kwargs: Additional processor-specific parameters
        """
        # Existing initialization
        self.batch_size = batch_size
        self.max_workers = max_workers or (psutil.cpu_count() or 4) - 1
        self.store_results = store_results
        config_source = config if config is not None else global_config
        self.memory_limit_mb = memory_limit_mb or (config_source.get('processors.memory_limit_mb', 1024) if config_source else 1024)
        self.tile_size = tile_size or (config_source.get('processors.tile_size', 512) if config_source else 512)
        self.supports_chunking = supports_chunking
        
        # Legacy memory tracker for backward compatibility
        self.memory_tracker = LegacyMemoryTracker()
        
        # Enhanced memory tracking
        self._enhanced_memory_tracker = get_memory_tracker()
        self._memory_mapped_files: Dict[str, mmap.mmap] = {}
        self._gdal_cache_usage = {'initial': 0, 'peak': 0}
        
        # Configuration
        self._passed_config = config
        self.config = self._merge_config(kwargs)
        
        # Progress tracking
        self._progress_callback: Optional[Callable[[float], None]] = None
        self._job_id: Optional[str] = None
        self._tile_progress: Dict[str, float] = {}
        self._processing_region: Optional[Tuple[float, float, float, float]] = None
        
        # Memory pressure handling
        self._enhanced_memory_tracker.add_pressure_callback(self._handle_memory_pressure)
        
        # NEW: Progress management
        self.enable_progress = enable_progress
        self._progress_manager = get_progress_manager() if enable_progress else None
        self._event_bus = get_event_bus() if enable_progress else None
        self._progress_node_id: Optional[str] = None
        
        # NEW: Checkpoint management
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self._checkpoint_manager = get_checkpoint_manager() if enable_checkpoints else None
        self._last_checkpoint_items = 0
        self._checkpoint_data: Dict[str, Any] = {}
        
        # NEW: Signal handling
        self._signal_handler = get_signal_handler()
        self._processing_lock = threading.Lock()
        self._is_processing = False
        self._should_stop = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        # NEW: Timeout support
        self.timeout_seconds = timeout_seconds
        self._timeout_timer: Optional[threading.Timer] = None
        
        # Register signal handlers
        self._register_signal_handlers()
        
    def _merge_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge kwargs with passed config or default config."""
        processor_name = self.__class__.__name__
        
        # Use passed config if available, otherwise use global config
        config_source = self._passed_config if self._passed_config is not None else global_config
        default_config = config_source.get(f'processors.{processor_name}', {})
        return {**default_config, **kwargs}
    
    @contextmanager
    def track_enhanced_memory(self, operation: str):
        """Enhanced context manager for memory tracking."""
        # Start legacy tracking
        self.memory_tracker.start(operation)
        
        # Start enhanced tracking with prediction
        prediction = self._enhanced_memory_tracker.predict_memory_usage(
            data_size_mb=100,  # Default estimate
            operation_type=operation,
            additional_factors={'processor': self.__class__.__name__}
        )
        
        # Log prediction warnings
        for warning in prediction.warnings:
            logger.warning(f"Memory prediction: {warning}")
            
        initial_snapshot = self._enhanced_memory_tracker.get_current_snapshot()
        
        try:
            yield prediction
        finally:
            # Stop legacy tracking
            memory_used = self.memory_tracker.stop()
            
            # Get final snapshot
            final_snapshot = self._enhanced_memory_tracker.get_current_snapshot()
            
            # Log comprehensive memory info
            gdal_cache_delta = final_snapshot.gdal_cache_used_mb - initial_snapshot.gdal_cache_used_mb
            mapped_files_delta = final_snapshot.mapped_files_mb - initial_snapshot.mapped_files_mb
            
            logger.info(
                f"{operation}: Heap={memory_used:.1f}MB, "
                f"GDAL={gdal_cache_delta:.1f}MB, "
                f"Mapped={mapped_files_delta:.1f}MB"
            )
    
    def estimate_memory_usage(self, 
                            data_size_mb: float,
                            operation_type: str = "processing",
                            **factors) -> Dict[str, Any]:
        """
        Estimate memory usage for processing.
        
        Args:
            data_size_mb: Size of input data in MB
            operation_type: Type of operation
            **factors: Additional factors affecting memory usage
            
        Returns:
            Dictionary with memory estimates
        """
        # Add processor-specific factors
        factors.update({
            'processor': self.__class__.__name__,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'supports_chunking': self.supports_chunking,
            'tile_size': self.tile_size
        })
        
        prediction = self._enhanced_memory_tracker.predict_memory_usage(
            data_size_mb=data_size_mb,
            operation_type=operation_type,
            additional_factors=factors
        )
        
        return {
            'predicted_peak_mb': prediction.predicted_peak_mb,
            'predicted_duration_seconds': prediction.predicted_duration_seconds,
            'confidence': prediction.confidence,
            'current_available_mb': self._enhanced_memory_tracker.get_current_snapshot().system_available_mb,
            'within_limits': prediction.predicted_peak_mb <= self.memory_limit_mb,
            'warnings': prediction.warnings
        }

    def start_progress(self, operation_name: str, total_items: int) -> None:
        """Start progress tracking for an operation."""
        if not self.enable_progress or not self._progress_manager:
            return
        
        # Create progress node
        self._progress_node_id = f"{self.__class__.__name__}_{operation_name}_{int(time.time())}"
        self._progress_manager.create_step(
            self._progress_node_id,
            parent=self._job_id or "default",
            total_substeps=total_items,
            metadata={'operation': operation_name}
        )
        self._progress_manager.start(self._progress_node_id)
        
        # Publish start event
        if self._event_bus:
            event = create_processing_progress(
                operation_name=operation_name,
                processed=0,
                total=total_items,
                source=self.__class__.__name__,
                node_id=self._progress_node_id
            )
            event.event_type = EventType.PROCESSING_START
            self._event_bus.publish(event)
    
    def update_progress(self, items_processed: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update progress for current operation."""
        if not self.enable_progress or not self._progress_manager or not self._progress_node_id:
            return
        
        self._progress_manager.update(
            self._progress_node_id,
            completed_units=items_processed,
            metadata=metadata
        )
        
        # Publish progress event
        if self._event_bus:
            event = create_processing_progress(
                operation_name=self._progress_node_id,
                processed=items_processed,
                total=100,  # Will be overridden by node data
                source=self.__class__.__name__,
                node_id=self._progress_node_id
            )
            self._event_bus.publish(event)
        
        # Call legacy progress callback if set
        if self._progress_callback:
            progress_data = self._progress_manager.get_progress(self._progress_node_id)
            self._progress_callback(progress_data['progress_percent'])
    
    def complete_progress(self, status: str = "completed", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Complete progress tracking for current operation."""
        if not self.enable_progress or not self._progress_manager or not self._progress_node_id:
            return
        
        self._progress_manager.complete(self._progress_node_id, status, metadata)
        
        # Publish complete event
        if self._event_bus:
            event_type = EventType.PROCESSING_COMPLETE if status == "completed" else EventType.PROCESSING_ERROR
            event = create_processing_progress(
                operation_name=self._progress_node_id,
                processed=100,
                total=100,
                source=self.__class__.__name__,
                node_id=self._progress_node_id
            )
            event.event_type = event_type
            self._event_bus.publish(event)
        
        self._progress_node_id = None

    # NEW: Checkpoint methods
    
    def save_checkpoint(self, checkpoint_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> str:
        """Save processing checkpoint."""
        if not self.enable_checkpoints or not self._checkpoint_manager:
            return ""
        
        checkpoint_id = checkpoint_id or f"{self.__class__.__name__}_{int(time.time())}"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'processor_state': {
                'class': self.__class__.__name__,
                'config': self.config,
                'items_processed': self._last_checkpoint_items,
                'progress_node_id': self._progress_node_id
            }
        }
        
        # Add custom data
        if data:
            checkpoint_data.update(data)
        
        # Add stored checkpoint data
        checkpoint_data.update(self._checkpoint_data)
        
        # Save checkpoint
        path = self._checkpoint_manager.save_checkpoint(
            checkpoint_id=checkpoint_id,
            data=checkpoint_data,
            level="step",
            metadata={'processor': self.__class__.__name__}
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        return path
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load processing checkpoint."""
        if not self.enable_checkpoints or not self._checkpoint_manager:
            return {}
        
        data = self._checkpoint_manager.load_checkpoint(checkpoint_id)
        
        # Restore processor state
        if 'processor_state' in data:
            state = data['processor_state']
            self._last_checkpoint_items = state.get('items_processed', 0)
            self._progress_node_id = state.get('progress_node_id')
        
        logger.info(f"Loaded checkpoint: {checkpoint_id}")
        return data
    
    def should_checkpoint(self, items_processed: int) -> bool:
        """Check if checkpoint should be saved."""
        if not self.enable_checkpoints:
            return False
        
        return (items_processed - self._last_checkpoint_items) >= self.checkpoint_interval
    
    # NEW: Signal handling methods
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        self._signal_handler.register_handler('processor', self._handle_processor_signal)
    
    def _handle_processor_signal(self, sig: signal.Signals) -> None:
        """Handle signals for this processor."""
        if sig in [signal.SIGTERM, signal.SIGINT]:
            logger.info(f"Received {sig.name}, initiating graceful shutdown")
            self._initiate_graceful_shutdown()
        elif sig == signal.SIGUSR1:
            # Pause processing
            logger.info("Pausing processing")
            self._pause_event.clear()
        elif sig == signal.SIGUSR2:
            # Resume processing
            logger.info("Resuming processing")
            self._pause_event.set()
    
    def _initiate_graceful_shutdown(self) -> None:
        """Initiate graceful shutdown."""
        self._should_stop.set()
        
        # Save checkpoint before shutting down
        if self._is_processing and self.enable_checkpoints:
            try:
                self.save_checkpoint(checkpoint_id=f"shutdown_{int(time.time())}")
            except Exception as e:
                logger.error(f"Failed to save shutdown checkpoint: {e}")
        
        # Complete progress tracking
        if self._progress_node_id:
            self.complete_progress(status="cancelled", metadata={'reason': 'shutdown'})
    
    def _check_timeout(self) -> None:
        """Check if processing has timed out."""
        if self.timeout_seconds and self._is_processing:
            logger.error(f"Processing timeout after {self.timeout_seconds} seconds")
            self._should_stop.set()
            self.complete_progress(status="failed", metadata={'reason': 'timeout'})

    def set_processing_region(self, bounds: Tuple[float, float, float, float]) -> None:
        """
        Set specific region for partial processing.
        
        Args:
            bounds: (minx, miny, maxx, maxy) bounding box
        """
        self._processing_region = bounds
        logger.info(f"Processing region set to: {bounds}")
        
    def clear_processing_region(self) -> None:
        """Clear processing region to process full dataset."""
        self._processing_region = None
        logger.info("Processing region cleared - will process full dataset")
        
    def track_tile_progress(self, tile_id: str, progress: float) -> None:
        """
        Track progress for a specific tile.
        
        Args:
            tile_id: Unique identifier for the tile
            progress: Progress percentage (0.0 to 100.0)
        """
        self._tile_progress[tile_id] = progress
        
        # Calculate overall tile progress
        if self._tile_progress:
            overall_progress = sum(self._tile_progress.values()) / len(self._tile_progress)
            
            # Update main progress callback
            if self._progress_callback:
                self._progress_callback(overall_progress)
                
    def get_tile_progress_summary(self) -> Dict[str, Any]:
        """Get summary of tile processing progress."""
        if not self._tile_progress:
            return {'total_tiles': 0, 'average_progress': 0.0}
            
        completed_tiles = sum(1 for p in self._tile_progress.values() if p >= 100.0)
        average_progress = sum(self._tile_progress.values()) / len(self._tile_progress)
        
        return {
            'total_tiles': len(self._tile_progress),
            'completed_tiles': completed_tiles,
            'in_progress_tiles': len(self._tile_progress) - completed_tiles,
            'average_progress': average_progress,
            'tile_details': dict(self._tile_progress)
        }
        
    def open_memory_mapped_file(self, file_path: Union[str, Path], mode: str = 'r') -> mmap.mmap:
        """
        Open a memory-mapped file with tracking.
        
        Args:
            file_path: Path to file
            mode: Access mode
            
        Returns:
            Memory-mapped file object
        """
        file_path = str(file_path)
        
        # Close existing mapping if present
        if file_path in self._memory_mapped_files:
            self.close_memory_mapped_file(file_path)
            
        # Open new mapping
        with open(file_path, mode) as f:
            mmapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ if 'r' in mode else mmap.ACCESS_WRITE)
            
        self._memory_mapped_files[file_path] = mmapped
        
        # Track with enhanced memory tracker
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        self._enhanced_memory_tracker.track_mapped_file(file_path, file_size_mb)
        
        logger.debug(f"Memory-mapped file opened: {file_path} ({file_size_mb:.1f}MB)")
        
        return mmapped
        
    def close_memory_mapped_file(self, file_path: Union[str, Path]) -> None:
        """
        Close a memory-mapped file.
        
        Args:
            file_path: Path to file
        """
        file_path = str(file_path)
        
        if file_path in self._memory_mapped_files:
            self._memory_mapped_files[file_path].close()
            del self._memory_mapped_files[file_path]
            
            # Stop tracking
            self._enhanced_memory_tracker.untrack_mapped_file(file_path)
            
            logger.debug(f"Memory-mapped file closed: {file_path}")
            
    def cleanup_memory_mapped_files(self) -> None:
        """Close all memory-mapped files."""
        for file_path in list(self._memory_mapped_files.keys()):
            self.close_memory_mapped_file(file_path)
            
    def get_gdal_cache_usage(self) -> Dict[str, float]:
        """Get GDAL cache usage information."""
        try:
            from osgeo import gdal
            
            current_cache_mb = gdal.GetCacheUsed() / (1024 * 1024)
            max_cache_mb = gdal.GetCacheMax() / (1024 * 1024)
            
            # Update peak tracking
            if current_cache_mb > self._gdal_cache_usage['peak']:
                self._gdal_cache_usage['peak'] = current_cache_mb
                
            return {
                'current_mb': current_cache_mb,
                'max_mb': max_cache_mb,
                'peak_mb': self._gdal_cache_usage['peak'],
                'usage_percentage': (current_cache_mb / max_cache_mb * 100) if max_cache_mb > 0 else 0
            }
            
        except ImportError:
            return {'current_mb': 0, 'max_mb': 0, 'peak_mb': 0, 'usage_percentage': 0}
            
    def _handle_memory_pressure(self, pressure_level: str) -> None:
        """
        Handle memory pressure notifications.
        
        Args:
            pressure_level: 'medium', 'high', or 'critical'
        """
        logger.warning(f"Memory pressure detected ({pressure_level})")
        
        if pressure_level == 'critical':
            # Aggressive cleanup
            self.cleanup_memory_mapped_files()
            
            # Clear GDAL cache if available
            try:
                from osgeo import gdal
                gdal.SetCacheMax(gdal.GetCacheMax() // 2)  # Reduce cache by half
            except ImportError:
                pass
                
            # Force garbage collection
            self._enhanced_memory_tracker.force_garbage_collection()
            
        elif pressure_level == 'high':
            # Moderate cleanup
            # Close oldest memory-mapped files (implement LRU if needed)
            files_to_close = list(self._memory_mapped_files.keys())[:2]
            for file_path in files_to_close:
                self.close_memory_mapped_file(file_path)
                
        elif pressure_level == 'medium':
            # Light cleanup - just force GC
            self._enhanced_memory_tracker.force_garbage_collection()
            
    def get_comprehensive_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        # Legacy tracker summary
        legacy_summary = self.memory_tracker.get_summary()
        
        # Enhanced tracker summary
        enhanced_summary = self._enhanced_memory_tracker.get_memory_summary()
        
        # GDAL cache info
        gdal_info = self.get_gdal_cache_usage()
        
        # Memory-mapped files info
        mapped_files_info = {
            'count': len(self._memory_mapped_files),
            'files': list(self._memory_mapped_files.keys())
        }
        
        return {
            'legacy_tracking': legacy_summary,
            'enhanced_tracking': enhanced_summary,
            'gdal_cache': gdal_info,
            'memory_mapped_files': mapped_files_info,
            'tile_progress': self.get_tile_progress_summary(),
            'memory_limit_mb': self.memory_limit_mb
        }
    
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
        """Validate input item."""
        pass

    def preprocess_batch(self, items: List[Any]) -> List[Any]:
        """Preprocess a batch of items (optional override)."""
        return items
    
    def postprocess_batch(self, results: List[Any]) -> List[Any]:
        """Postprocess batch results (optional override)."""
        return results
    
    def process_batch(self, items: List[Any]) -> ProcessingResult:
        """
        Enhanced process_batch with checkpoint and cancellation support.
        
        Args:
            items: Items to process
            
        Returns:
            ProcessingResult with details
        """
        # Check if we should stop
        if self._should_stop.is_set():
            return ProcessingResult(
                success=False,
                items_processed=0,
                items_failed=len(items),
                elapsed_time=0,
                memory_used_mb=0,
                errors=[{'error': 'Processing cancelled'}]
            )
        
        # Wait if paused
        self._pause_event.wait()
        
        with self._processing_lock:
            self._is_processing = True
            
            # Start timeout timer if configured
            if self.timeout_seconds:
                self._timeout_timer = threading.Timer(self.timeout_seconds, self._check_timeout)
                self._timeout_timer.start()
            
            try:
                # Original process_batch implementation with enhancements
                start_time = time.time()
                processed_items = []
                failed_items = []
                errors = []
                
                with self.track_memory(f"Batch processing ({len(items)} items)"):
                    # Validate inputs
                    valid_items = []
                    for i, item in enumerate(items):
                        # Check cancellation
                        if self._should_stop.is_set():
                            break
                        
                        is_valid, error_msg = self.validate_input(item)
                        if is_valid:
                            valid_items.append(item)
                        else:
                            failed_items.append(item)
                            errors.append({
                                'index': i,
                                'error': error_msg or 'Validation failed',
                                'item': str(item)[:100]
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
                    for i, item in enumerate(valid_items):
                        # Check cancellation and pause
                        if self._should_stop.is_set():
                            break
                        self._pause_event.wait()
                        
                        try:
                            result = self.process_single(item)
                            if result is not None:
                                processed_items.append(result)
                            else:
                                failed_items.append(item)
                            
                            # Update progress
                            self.update_progress(len(processed_items))
                            
                            # Check if should checkpoint
                            if self.should_checkpoint(len(processed_items)):
                                self._checkpoint_data['processed_items'] = processed_items
                                self._checkpoint_data['failed_items'] = failed_items
                                self.save_checkpoint()
                                self._last_checkpoint_items = len(processed_items)
                                
                        except Exception as e:
                            failed_items.append(item)
                            errors.append({
                                'error': str(e),
                                'item': str(item)[:100]
                            })
                            logger.error(f"Processing error: {e}")
                            
                            # Save state on error if configured
                            if self.enable_checkpoints:
                                try:
                                    self._checkpoint_data['error'] = str(e)
                                    self._checkpoint_data['processed_items'] = processed_items
                                    self._checkpoint_data['failed_items'] = failed_items
                                    self.save_checkpoint(checkpoint_id=f"error_{int(time.time())}")
                                except Exception as cp_error:
                                    logger.error(f"Failed to save error checkpoint: {cp_error}")
                    
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
                
            finally:
                self._is_processing = False
                
                # Cancel timeout timer
                if self._timeout_timer:
                    self._timeout_timer.cancel()
                    self._timeout_timer = None

    def process_iterator(self, 
                        iterator: Iterator[Any],
                        total: Optional[int] = None,
                        resume_from_checkpoint: Optional[str] = None) -> ProcessingResult:
        """
        Enhanced process_iterator with resume support.
        
        Args:
            iterator: Iterator of items
            total: Total number of items (for progress)
            resume_from_checkpoint: Checkpoint ID to resume from
            
        Returns:
            Combined ProcessingResult
        """
        # Start progress tracking
        if total:
            self.start_progress(f"Iterator processing", total)
        
        # Resume from checkpoint if provided
        skip_items = 0
        if resume_from_checkpoint and self.enable_checkpoints:
            checkpoint_data = self.load_checkpoint(resume_from_checkpoint)
            skip_items = checkpoint_data.get('processor_state', {}).get('items_processed', 0)
            logger.info(f"Resuming from checkpoint, skipping {skip_items} items")
        
        total_processed = 0
        total_failed = 0
        all_errors = []
        total_time = 0.0
        total_memory = 0.0
        
        batch = []
        batch_num = 0
        items_seen = 0
        
        try:
            for item in iterator:
                # Skip items if resuming
                if items_seen < skip_items:
                    items_seen += 1
                    continue
                
                # Check cancellation
                if self._should_stop.is_set():
                    break
                
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
                    
                    batch = []
            
            # Process remaining items
            if batch and not self._should_stop.is_set():
                result = self.process_batch(batch)
                total_processed += result.items_processed
                total_failed += result.items_failed
                total_time += result.elapsed_time
                total_memory += result.memory_used_mb
            
            # Complete progress
            status = "completed" if not self._should_stop.is_set() else "cancelled"
            self.complete_progress(status=status)
            
            return ProcessingResult(
                success=total_processed > 0,
                items_processed=total_processed,
                items_failed=total_failed,
                elapsed_time=total_time,
                memory_used_mb=total_memory,
                errors=all_errors if all_errors else None
            )
            
        except Exception as e:
            logger.error(f"Iterator processing failed: {e}")
            self.complete_progress(status="failed", metadata={'error': str(e)})
            raise
    
    def _store_results(self, results: List[Any]):
        """Store results in database (override in subclasses)."""
        logger.warning(f"{self.__class__.__name__} does not implement _store_results")
    
    def set_progress_callback(self, callback: Callable[[float], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def set_job_id(self, job_id: str):
        """Set job ID for tracking."""
        self._job_id = job_id
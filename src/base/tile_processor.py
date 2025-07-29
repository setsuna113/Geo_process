"""Base class for tile-based raster processing with progress tracking and checkpoints."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
import numpy as np
from pathlib import Path
import logging
import time
import threading
from dataclasses import dataclass, field

from src.abstractions.types.processing_types import ProcessingStatus, TileProgress
from src.abstractions.types.checkpoint_types import CheckpointLevel
from src.abstractions.mixins import Tileable, TileSpec
from src.abstractions.mixins import Cacheable

logger = logging.getLogger(__name__)


# ProcessingStatus moved to abstractions.types.processing_types


# TileProgress moved to abstractions.types.processing_types


@dataclass
class ProcessingCheckpoint:
    """Checkpoint for resuming processing."""
    timestamp: float
    completed_tiles: List[str]
    failed_tiles: List[str] 
    total_tiles: int
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_tiles == 0:
            return 100.0
        return (len(self.completed_tiles) / self.total_tiles) * 100.0


class BaseTileProcessor(Tileable, Cacheable, ABC):
    """
    Abstract base class for tile-based raster processing.
    
    Handles tile boundaries, overlaps, progress tracking, and checkpoint support.
    """
    
    def __init__(self,
                 tile_size: int = 512,
                 overlap: int = 0,
                 num_workers: int = 1,
                 checkpoint_interval: int = 10,
                 checkpoint_manager=None,
                 **kwargs):
        """
        Initialize tile processor.
        
        Args:
            tile_size: Size of processing tiles in pixels
            overlap: Overlap between tiles for boundary handling
            num_workers: Number of worker threads
            checkpoint_interval: Save checkpoint every N completed tiles
            checkpoint_manager: Injected checkpoint manager (optional)
            **kwargs: Additional processor-specific parameters
        """
        super().__init__()
        
        self.set_tile_size(tile_size)
        self.set_overlap(overlap)
        self.num_workers = max(1, num_workers)
        self.checkpoint_interval = checkpoint_interval
        
        # Progress tracking
        self._tile_progress: Dict[str, TileProgress] = {}
        self._progress_lock = threading.RLock()
        self._progress_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Checkpoint management - injected to avoid architectural violation
        self._checkpoint_manager = checkpoint_manager
        self._auto_checkpoint = True
        self._last_checkpoint_time = time.time()
        self._process_id = f"tile_processor_{id(self)}"
        
        # Processing state
        self._is_processing = False
        self._should_stop = threading.Event()
        self._worker_threads: List[threading.Thread] = []
        
        # Configuration
        self._processor_config = {
            'memory_limit_mb': 1024,
            'skip_existing': True,
            'retry_failed': True,
            'max_retries': 3,
            'handle_boundaries': True,
            'merge_overlaps': True
        }
        self._processor_config.update(kwargs)
        
    @abstractmethod
    def process_tile(self, 
                    tile_data: np.ndarray,
                    tile_spec: TileSpec,
                    **kwargs) -> np.ndarray:
        """
        Process a single tile.
        
        Args:
            tile_data: Input tile data
            tile_spec: Tile specification with bounds and metadata
            **kwargs: Additional processing parameters
            
        Returns:
            Processed tile data
        """
        pass
        
    @abstractmethod
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get output shape for given input shape."""
        pass
        
    @abstractmethod
    def get_output_dtype(self, input_dtype: np.dtype) -> np.dtype:
        """Get output data type for given input type."""
        pass
        
    def configure_processor(self, **config) -> None:
        """Configure processor parameters."""
        self._processor_config.update(config)
        
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)
        
    def set_checkpoint_file(self, checkpoint_file: Union[str, Path]) -> None:
        """Set process ID for checkpointing (file parameter ignored in unified system)."""
        # In the unified system, we use the process ID instead of file paths
        self._process_id = Path(checkpoint_file).stem  # Use filename as process ID
        
    def process_dataset(self,
                       input_data: Any,
                       output_path: Optional[Union[str, Path]] = None,
                       resume_from_checkpoint: bool = True,
                       **processing_kwargs) -> Any:
        """
        Process entire dataset using tiles.
        
        Args:
            input_data: Input dataset or data source
            output_path: Path for output (if applicable)
            resume_from_checkpoint: Whether to resume from existing checkpoint
            **processing_kwargs: Additional processing parameters
            
        Returns:
            Processing results
        """
        try:
            self._is_processing = True
            self._should_stop.clear()
            
            # Load checkpoint if resuming
            checkpoint = None
            if resume_from_checkpoint:
                checkpoint_data = self._checkpoint_manager.load_latest(self._process_id, CheckpointLevel.STEP)
                if checkpoint_data:
                    checkpoint = self._deserialize_checkpoint(checkpoint_data.data)
                
            # Generate tile specifications
            tiles = self._prepare_tiles(input_data, checkpoint)
            
            # Initialize progress tracking
            self._initialize_progress(tiles)
            
            # Process tiles
            results = self._process_tiles(input_data, tiles, **processing_kwargs)
            
            # Finalize results
            output_path_obj = Path(output_path) if output_path else None
            final_result = self._finalize_processing(results, output_path_obj)
            
            # Cleanup is handled automatically by the unified checkpoint system
            pass
                
            return final_result
            
        finally:
            self._is_processing = False
            self._should_stop.set()
            self._cleanup_workers()
            
    def stop_processing(self) -> None:
        """Stop processing gracefully."""
        logger.info("Stopping tile processing...")
        self._should_stop.set()
        
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get processing progress summary."""
        with self._progress_lock:
            total_tiles = len(self._tile_progress)
            if total_tiles == 0:
                return {'completion_percentage': 0.0}
                
            completed = sum(1 for p in self._tile_progress.values() if p.status == ProcessingStatus.COMPLETED)
            failed = sum(1 for p in self._tile_progress.values() if p.status == ProcessingStatus.FAILED)
            processing = sum(1 for p in self._tile_progress.values() if p.status == ProcessingStatus.PROCESSING)
            
            total_memory = sum(p.memory_usage_mb for p in self._tile_progress.values())
            avg_processing_time = np.mean([
                p.processing_time_seconds for p in self._tile_progress.values()
                if p.processing_time_seconds is not None
            ]) if any(p.processing_time_seconds for p in self._tile_progress.values()) else 0.0
            
            return {
                'completion_percentage': (completed / total_tiles) * 100.0,
                'total_tiles': total_tiles,
                'completed_tiles': completed,
                'failed_tiles': failed,
                'processing_tiles': processing,
                'pending_tiles': total_tiles - completed - failed - processing,
                'total_memory_mb': total_memory,
                'average_processing_time_seconds': avg_processing_time,
                'is_processing': self._is_processing
            }
            
    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining processing time in seconds."""
        with self._progress_lock:
            completed_tiles = [p for p in self._tile_progress.values() if p.is_complete]
            if not completed_tiles:
                return None
                
            avg_time = np.mean([p.processing_time_seconds for p in completed_tiles if p.processing_time_seconds])
            remaining_tiles = len([p for p in self._tile_progress.values() if not p.is_complete])
            
            if avg_time > 0:
                return float(remaining_tiles * avg_time / max(1, self.num_workers))
            return None
            
    def _prepare_tiles(self, input_data: Any, checkpoint: Optional[ProcessingCheckpoint]) -> List[TileSpec]:
        """Prepare tile specifications for processing."""
        # Get data dimensions - override in subclasses
        width, height = self.get_dimensions()
        
        # Generate all tiles
        all_tiles = self.generate_tile_specs()
        
        # Filter based on checkpoint if resuming
        if checkpoint:
            remaining_tiles = [
                tile for tile in all_tiles 
                if tile.tile_id not in checkpoint.completed_tiles
            ]
            logger.info(f"Resuming from checkpoint: {len(remaining_tiles)} tiles remaining")
            return remaining_tiles
            
        return all_tiles
        
    def _initialize_progress(self, tiles: List[TileSpec]) -> None:
        """Initialize progress tracking for tiles."""
        with self._progress_lock:
            self._tile_progress.clear()
            for tile in tiles:
                self._tile_progress[tile.tile_id] = TileProgress(tile_id=tile.tile_id)
                
    def _process_tiles(self, input_data: Any, tiles: List[TileSpec], **kwargs) -> Dict[str, Any]:
        """Process tiles using worker threads."""
        if self.num_workers == 1:
            # Single-threaded processing
            return self._process_tiles_sequential(input_data, tiles, **kwargs)
        else:
            # Multi-threaded processing
            return self._process_tiles_parallel(input_data, tiles, **kwargs)
            
    def _process_tiles_sequential(self, input_data: Any, tiles: List[TileSpec], **kwargs) -> Dict[str, Any]:
        """Process tiles sequentially."""
        results = {}
        
        for tile in tiles:
            if self._should_stop.is_set():
                break
                
            result = self._process_single_tile(input_data, tile, **kwargs)
            if result is not None:
                results[tile.tile_id] = result
                
            # Update checkpoint periodically
            if len(results) % self.checkpoint_interval == 0:
                self._save_checkpoint()
                
        return results
        
    def _process_tiles_parallel(self, input_data: Any, tiles: List[TileSpec], **kwargs) -> Dict[str, Any]:
        """Process tiles in parallel using worker threads."""
        import queue
        from typing import Any
        
        tile_queue: queue.Queue[Any] = queue.Queue()
        result_queue: queue.Queue[Any] = queue.Queue()
        
        # Add tiles to queue
        for tile in tiles:
            tile_queue.put(tile)
            
        # Worker function
        def worker():
            while not self._should_stop.is_set():
                try:
                    tile = tile_queue.get(timeout=1.0)
                    result = self._process_single_tile(input_data, tile, **kwargs)
                    result_queue.put((tile.tile_id, result))
                    tile_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    
        # Start workers
        self._worker_threads = []
        for _ in range(self.num_workers):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            self._worker_threads.append(thread)
            
        # Collect results
        results = {}
        completed_count = 0
        
        while completed_count < len(tiles) and not self._should_stop.is_set():
            try:
                tile_id, result = result_queue.get(timeout=1.0)
                if result is not None:
                    results[tile_id] = result
                completed_count += 1
                
                # Update checkpoint periodically
                if completed_count % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                    
            except queue.Empty:
                continue
                
        return results
        
    def _process_single_tile(self, input_data: Any, tile_spec: TileSpec, **kwargs) -> Optional[Any]:
        """Process a single tile with error handling and progress tracking."""
        tile_id = tile_spec.tile_id
        
        with self._progress_lock:
            progress = self._tile_progress[tile_id]
            progress.status = ProcessingStatus.PROCESSING
            progress.start_time = time.time()
            
        try:
            # Check if should skip
            if self._should_skip_tile(tile_spec):
                progress.status = ProcessingStatus.SKIPPED
                return None
                
            # Extract tile data from input
            tile_data = self._extract_tile_data(input_data, tile_spec)
            
            # Estimate memory usage
            dtype = tile_data.dtype if hasattr(tile_data, 'dtype') else np.dtype(np.float32)
            memory_mb = self.estimate_tile_memory_usage(tile_spec, dtype)
            progress.memory_usage_mb = memory_mb
            
            # Process tile
            result = self.process_tile(tile_data, tile_spec, **kwargs)
            
            # Handle boundaries and overlaps if configured
            if self._processor_config.get('handle_boundaries', True):
                result = self._handle_tile_boundaries(result, tile_spec)
                
            progress.status = ProcessingStatus.COMPLETED
            progress.end_time = time.time()
            progress.progress_percentage = 100.0
            
            # Notify progress callbacks
            self._notify_progress_callbacks()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            progress.status = ProcessingStatus.FAILED
            progress.error_message = str(e)
            progress.end_time = time.time()
            return None
            
    def _extract_tile_data(self, input_data: Any, tile_spec: TileSpec) -> np.ndarray:
        """Extract data for a tile from input dataset."""
        # Override in subclasses based on input data type
        if hasattr(input_data, 'read_tile'):
            return input_data.read_tile(tile_spec)
        elif isinstance(input_data, np.ndarray):
            col_off, row_off, width, height = tile_spec.bounds
            return input_data[row_off:row_off+height, col_off:col_off+width]
        else:
            raise NotImplementedError("Unsupported input data type")
            
    def _should_skip_tile(self, tile_spec: TileSpec) -> bool:
        """Check if tile should be skipped."""
        # Override in subclasses for custom skip logic
        return False
        
    def _handle_tile_boundaries(self, tile_result: np.ndarray, tile_spec: TileSpec) -> np.ndarray:
        """Handle tile boundaries and overlaps."""
        # Override in subclasses for boundary-specific processing
        return tile_result
        
    def _finalize_processing(self, results: Dict[str, Any], output_path: Optional[Path]) -> Any:
        """Finalize processing results."""
        # Override in subclasses for result combination/output
        return results
        
    def _save_checkpoint(self) -> None:
        """Save processing checkpoint using unified system."""
        if not self._auto_checkpoint:
            return
            
        try:
            with self._progress_lock:
                completed_tiles = [
                    tile_id for tile_id, progress in self._tile_progress.items()
                    if progress.status == ProcessingStatus.COMPLETED
                ]
                failed_tiles = [
                    tile_id for tile_id, progress in self._tile_progress.items()
                    if progress.status == ProcessingStatus.FAILED
                ]
                
                checkpoint_data = {
                    'timestamp': time.time(),
                    'completed_tiles': completed_tiles,
                    'failed_tiles': failed_tiles,
                    'total_tiles': len(self._tile_progress),
                    'processing_parameters': self._processor_config.copy(),
                    'tile_progress': {
                        tile_id: {
                            'status': progress.status.value,
                            'start_time': progress.start_time,
                            'end_time': progress.end_time,
                            'error_message': progress.error_message,
                            'progress_percentage': progress.progress_percentage,
                            'memory_usage_mb': progress.memory_usage_mb
                        } for tile_id, progress in self._tile_progress.items()
                    }
                }
                
                self._checkpoint_manager.save(
                    process=self._process_id,
                    data=checkpoint_data,
                    level=CheckpointLevel.STEP,
                    tags=['tile_processor', 'processing']
                )
                    
                logger.debug(f"Saved checkpoint: {len(completed_tiles)} completed tiles")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def _deserialize_checkpoint(self, data: Dict[str, Any]) -> Optional[ProcessingCheckpoint]:
        """Deserialize checkpoint data from unified system."""
        try:
            # Restore tile progress
            tile_progress_data = data.get('tile_progress', {})
            for tile_id, progress_data in tile_progress_data.items():
                self._tile_progress[tile_id] = TileProgress(
                    tile_id=tile_id,
                    status=ProcessingStatus(progress_data['status']),
                    start_time=progress_data.get('start_time'),
                    end_time=progress_data.get('end_time'),
                    error_message=progress_data.get('error_message'),
                    progress_percentage=progress_data.get('progress_percentage', 0.0),
                    memory_usage_mb=progress_data.get('memory_usage_mb', 0.0)
                )
            
            # Create checkpoint object
            checkpoint = ProcessingCheckpoint(
                timestamp=data['timestamp'],
                completed_tiles=data['completed_tiles'],
                failed_tiles=data['failed_tiles'],
                total_tiles=data['total_tiles'],
                processing_parameters=data.get('processing_parameters', {})
            )
            
            logger.info(f"Loaded checkpoint: {checkpoint.completion_percentage:.1f}% complete")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoint: {e}")
            return None
            
    def _notify_progress_callbacks(self) -> None:
        """Notify registered progress callbacks."""
        if not self._progress_callbacks:
            return
            
        try:
            summary = self.get_progress_summary()
            for callback in self._progress_callbacks:
                try:
                    callback(summary)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
        except Exception as e:
            logger.error(f"Error in progress notification: {e}")
            
    def _cleanup_workers(self) -> None:
        """Clean up worker threads."""
        for thread in self._worker_threads:
            thread.join(timeout=5.0)
        self._worker_threads.clear()

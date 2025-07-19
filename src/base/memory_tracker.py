"""Enhanced memory tracking with virtual memory, memory-mapped files, and GDAL cache monitoring."""

import psutil
import os
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    
    # Process memory
    heap_memory_mb: float
    virtual_memory_mb: float
    rss_memory_mb: float  # Resident Set Size
    
    # Memory-mapped files
    mapped_files_mb: float
    mapped_file_count: int
    
    # GDAL cache
    gdal_cache_mb: float
    gdal_cache_used_mb: float
    
    # Shared memory
    shared_memory_mb: float
    
    # System memory
    system_total_mb: float
    system_available_mb: float
    system_used_percentage: float
    
    # Memory pressure indicators
    memory_pressure_level: str  # 'low', 'medium', 'high', 'critical'
    swap_usage_mb: float


@dataclass
class MemoryPrediction:
    """Memory usage prediction based on data characteristics."""
    predicted_peak_mb: float
    predicted_duration_seconds: float
    confidence: float  # 0.0 to 1.0
    factors: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class MemoryTracker:
    """
    Enhanced memory tracker with comprehensive monitoring capabilities.
    
    Tracks heap memory, virtual memory, memory-mapped files, GDAL cache,
    and provides memory prediction based on data characteristics.
    """
    
    def __init__(self,
                 monitoring_interval: float = 1.0,
                 history_size: int = 1000,
                 enable_predictions: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            monitoring_interval: Seconds between monitoring snapshots
            history_size: Number of snapshots to keep in history
            enable_predictions: Whether to enable memory prediction
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_predictions = enable_predictions
        
        # Memory tracking
        self._process = psutil.Process()
        self._memory_history: deque = deque(maxlen=history_size)
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._monitoring_lock = threading.RLock()
        
        # Memory pressure callbacks
        self._pressure_callbacks: List[Callable[[str], None]] = []
        self._pressure_thresholds = {
            'medium': 70.0,  # % of system memory
            'high': 85.0,
            'critical': 95.0
        }
        
        # GDAL tracking
        self._gdal_available = self._check_gdal_availability()
        
        # Memory-mapped file tracking
        self._mapped_files: Dict[str, float] = {}
        self._mapped_files_lock = threading.RLock()
        
        # Prediction model
        self._prediction_history: List[Tuple[Dict[str, Any], float]] = []
        
        # Start monitoring
        self.start_monitoring()
        
    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self._monitoring_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
        
    def get_current_snapshot(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        with self._monitoring_lock:
            # Process memory
            memory_info = self._process.memory_info()
            heap_memory_mb = memory_info.rss / (1024 * 1024)
            virtual_memory_mb = memory_info.vms / (1024 * 1024)
            rss_memory_mb = memory_info.rss / (1024 * 1024)
            
            # Memory-mapped files
            mapped_files_mb, mapped_file_count = self._get_mapped_files_info()
            
            # GDAL cache
            gdal_cache_mb, gdal_cache_used_mb = self._get_gdal_cache_info()
            
            # Shared memory
            shared_memory_mb = self._get_shared_memory_info()
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_total_mb = system_memory.total / (1024 * 1024)
            system_available_mb = system_memory.available / (1024 * 1024)
            system_used_percentage = system_memory.percent
            
            # Memory pressure
            memory_pressure_level = self._calculate_memory_pressure(system_used_percentage)
            
            # Swap usage
            swap_info = psutil.swap_memory()
            swap_usage_mb = swap_info.used / (1024 * 1024)
            
            return MemorySnapshot(
                timestamp=time.time(),
                heap_memory_mb=heap_memory_mb,
                virtual_memory_mb=virtual_memory_mb,
                rss_memory_mb=rss_memory_mb,
                mapped_files_mb=mapped_files_mb,
                mapped_file_count=mapped_file_count,
                gdal_cache_mb=gdal_cache_mb,
                gdal_cache_used_mb=gdal_cache_used_mb,
                shared_memory_mb=shared_memory_mb,
                system_total_mb=system_total_mb,
                system_available_mb=system_available_mb,
                system_used_percentage=system_used_percentage,
                memory_pressure_level=memory_pressure_level,
                swap_usage_mb=swap_usage_mb
            )
            
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        snapshot = self.get_current_snapshot()
        
        # Calculate trends if we have history
        trend_data = {}
        if len(self._memory_history) > 1:
            recent_snapshots = list(self._memory_history)[-10:]  # Last 10 snapshots
            
            # Calculate memory growth rate
            first_snapshot = recent_snapshots[0]
            time_diff = snapshot.timestamp - first_snapshot.timestamp
            if time_diff > 0:
                memory_growth_rate = (snapshot.heap_memory_mb - first_snapshot.heap_memory_mb) / time_diff
                trend_data['memory_growth_rate_mb_per_sec'] = memory_growth_rate
                
        return {
            'current': {
                'heap_memory_mb': snapshot.heap_memory_mb,
                'virtual_memory_mb': snapshot.virtual_memory_mb,
                'mapped_files_mb': snapshot.mapped_files_mb,
                'gdal_cache_mb': snapshot.gdal_cache_mb,
                'system_available_mb': snapshot.system_available_mb,
                'memory_pressure': snapshot.memory_pressure_level
            },
            'system': {
                'total_memory_mb': snapshot.system_total_mb,
                'used_percentage': snapshot.system_used_percentage,
                'swap_usage_mb': snapshot.swap_usage_mb
            },
            'trends': trend_data,
            'history_length': len(self._memory_history)
        }
        
    def predict_memory_usage(self,
                           data_size_mb: float,
                           operation_type: str,
                           additional_factors: Optional[Dict[str, Any]] = None) -> MemoryPrediction:
        """
        Predict memory usage based on data characteristics.
        
        Args:
            data_size_mb: Size of input data in MB
            operation_type: Type of operation ('processing', 'loading', 'resampling', etc.)
            additional_factors: Additional factors affecting memory usage
            
        Returns:
            Memory usage prediction
        """
        if not self.enable_predictions:
            return MemoryPrediction(
                predicted_peak_mb=data_size_mb * 2,  # Simple fallback
                predicted_duration_seconds=60.0,
                confidence=0.5
            )
            
        factors = additional_factors or {}
        
        # Base prediction factors
        base_multiplier = {
            'processing': 3.0,    # Processing typically uses 3x data size
            'loading': 1.5,       # Loading uses ~1.5x for buffers
            'resampling': 2.5,    # Resampling needs input + output + temp
            'analysis': 2.0,      # Analysis varies
            'tiling': 1.8         # Tiling is more memory efficient
        }.get(operation_type, 2.0)
        
        # Adjust based on factors
        multiplier = base_multiplier
        
        # Data type factor
        if 'dtype' in factors:
            dtype_str = str(factors['dtype'])
            if 'float64' in dtype_str:
                multiplier *= 1.2
            elif 'float32' in dtype_str:
                multiplier *= 1.0
            elif 'int32' in dtype_str:
                multiplier *= 0.8
            elif 'int16' in dtype_str:
                multiplier *= 0.6
                
        # Tile size factor
        if 'tile_size' in factors:
            tile_size = factors['tile_size']
            if tile_size > 2048:
                multiplier *= 1.3
            elif tile_size < 512:
                multiplier *= 0.8
                
        # Number of bands factor
        if 'bands' in factors:
            multiplier *= max(1.0, factors['bands'] * 0.2)
            
        # Parallel processing factor
        if 'num_workers' in factors and factors['num_workers'] > 1:
            multiplier *= min(2.0, 1.0 + factors['num_workers'] * 0.2)
            
        # Calculate prediction
        predicted_peak_mb = data_size_mb * multiplier
        
        # Estimate duration based on data size and operation
        base_time_per_mb = {
            'processing': 0.1,    # 100ms per MB
            'loading': 0.05,      # 50ms per MB
            'resampling': 0.15,   # 150ms per MB
            'analysis': 0.08,     # 80ms per MB
            'tiling': 0.03        # 30ms per MB
        }.get(operation_type, 0.1)
        
        predicted_duration_seconds = data_size_mb * base_time_per_mb
        
        # Adjust duration based on system load
        system_load = psutil.cpu_percent(interval=0.1)
        if system_load > 80:
            predicted_duration_seconds *= 1.5
        elif system_load > 50:
            predicted_duration_seconds *= 1.2
            
        # Calculate confidence based on historical accuracy
        confidence = self._calculate_prediction_confidence(operation_type, factors)
        
        # Generate warnings
        warnings = []
        current_snapshot = self.get_current_snapshot()
        
        if predicted_peak_mb > current_snapshot.system_available_mb * 0.8:
            warnings.append("Predicted memory usage may exceed available memory")
            
        if predicted_peak_mb > current_snapshot.system_total_mb * 0.5:
            warnings.append("Large memory allocation may impact system performance")
            
        if predicted_duration_seconds > 300:  # 5 minutes
            warnings.append("Long-running operation predicted")
            
        return MemoryPrediction(
            predicted_peak_mb=predicted_peak_mb,
            predicted_duration_seconds=predicted_duration_seconds,
            confidence=confidence,
            factors=factors,
            warnings=warnings
        )
        
    def track_mapped_file(self, file_path: str, size_mb: float) -> None:
        """Track a memory-mapped file."""
        with self._mapped_files_lock:
            self._mapped_files[file_path] = size_mb
            
    def untrack_mapped_file(self, file_path: str) -> None:
        """Stop tracking a memory-mapped file."""
        with self._mapped_files_lock:
            self._mapped_files.pop(file_path, None)
            
    def add_pressure_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for memory pressure notifications."""
        self._pressure_callbacks.append(callback)
        
    def set_pressure_thresholds(self, **thresholds) -> None:
        """Set memory pressure thresholds (percentages)."""
        self._pressure_thresholds.update(thresholds)
        
    def force_garbage_collection(self) -> float:
        """Force garbage collection and return freed memory in MB."""
        import gc
        
        before_snapshot = self.get_current_snapshot()
        gc.collect()
        time.sleep(0.1)  # Allow for cleanup
        after_snapshot = self.get_current_snapshot()
        
        freed_mb = before_snapshot.heap_memory_mb - after_snapshot.heap_memory_mb
        logger.info(f"Garbage collection freed {freed_mb:.2f} MB")
        
        return max(0, freed_mb)
        
    def _monitoring_worker(self) -> None:
        """Background worker for memory monitoring."""
        while not self._stop_monitoring.is_set():
            try:
                snapshot = self.get_current_snapshot()
                
                with self._monitoring_lock:
                    self._memory_history.append(snapshot)
                    
                # Check for memory pressure
                self._check_memory_pressure(snapshot)
                
                # Sleep until next monitoring interval
                if self._stop_monitoring.wait(self.monitoring_interval):
                    break
                    
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(1.0)
                
    def _check_gdal_availability(self) -> bool:
        """Check if GDAL is available for cache monitoring."""
        try:
            from osgeo import gdal
            return True
        except ImportError:
            return False
            
    def _get_mapped_files_info(self) -> Tuple[float, int]:
        """Get information about memory-mapped files."""
        with self._mapped_files_lock:
            total_size = sum(self._mapped_files.values())
            count = len(self._mapped_files)
            return total_size, count
            
    def _get_gdal_cache_info(self) -> Tuple[float, float]:
        """Get GDAL cache information."""
        if not self._gdal_available:
            return 0.0, 0.0
            
        try:
            from osgeo import gdal
            
            # Get cache size in MB
            cache_size_mb = gdal.GetCacheMax() / (1024 * 1024)
            
            # Get used cache (this is an approximation)
            # GDAL doesn't provide direct API for used cache
            cache_used_mb = cache_size_mb * 0.5  # Rough estimate
            
            return cache_size_mb, cache_used_mb
            
        except Exception as e:
            logger.debug(f"Failed to get GDAL cache info: {e}")
            return 0.0, 0.0
            
    def _get_shared_memory_info(self) -> float:
        """Get shared memory usage information."""
        try:
            # This is Linux-specific - adapt for other platforms
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('Shmem:'):
                            # Extract value in kB and convert to MB
                            shmem_kb = int(line.split()[1])
                            return shmem_kb / 1024
            return 0.0
        except Exception:
            return 0.0
            
    def _calculate_memory_pressure(self, system_used_percentage: float) -> str:
        """Calculate memory pressure level."""
        if system_used_percentage >= self._pressure_thresholds['critical']:
            return 'critical'
        elif system_used_percentage >= self._pressure_thresholds['high']:
            return 'high'
        elif system_used_percentage >= self._pressure_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
            
    def _check_memory_pressure(self, snapshot: MemorySnapshot) -> None:
        """Check for memory pressure and notify callbacks."""
        if snapshot.memory_pressure_level in ['high', 'critical']:
            for callback in self._pressure_callbacks:
                try:
                    callback(snapshot.memory_pressure_level)
                except Exception as e:
                    logger.error(f"Memory pressure callback error: {e}")
                    
    def _calculate_prediction_confidence(self,
                                       operation_type: str,
                                       factors: Dict[str, Any]) -> float:
        """Calculate confidence in memory usage prediction."""
        # Base confidence by operation type
        base_confidence = {
            'loading': 0.8,       # Loading is predictable
            'processing': 0.6,    # Processing varies more
            'resampling': 0.7,    # Resampling is moderately predictable
            'analysis': 0.5,      # Analysis is highly variable
            'tiling': 0.75        # Tiling is fairly predictable
        }.get(operation_type, 0.6)
        
        # Adjust based on available factors
        if 'dtype' in factors:
            base_confidence += 0.1
        if 'tile_size' in factors:
            base_confidence += 0.1
        if 'bands' in factors:
            base_confidence += 0.05
            
        # Historical accuracy would improve this
        # For now, cap at 0.9
        return min(0.9, base_confidence)


# Global memory tracker instance
_global_memory_tracker: Optional[MemoryTracker] = None


def get_memory_tracker() -> MemoryTracker:
    """Get the global memory tracker instance."""
    global _global_memory_tracker
    if _global_memory_tracker is None:
        _global_memory_tracker = MemoryTracker()
    return _global_memory_tracker


def cleanup_memory_tracker() -> None:
    """Clean up the global memory tracker."""
    global _global_memory_tracker
    if _global_memory_tracker is not None:
        _global_memory_tracker.stop_monitoring()
        _global_memory_tracker = None

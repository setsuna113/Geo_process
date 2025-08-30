from src.abstractions import get_logger
"""Uniform Numba thread control for SOM parallel operations.

This module provides centralized control over Numba's parallel execution threads,
ensuring consistent behavior across all parallel SOM operations.
"""

import os
import logging
import numba
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = get_logger(__name__)


class NumbaThreadController:
    """Centralized controller for Numba thread management.
    
    Provides uniform control over all Numba parallel operations in the SOM pipeline,
    preventing thread explosion and ensuring predictable resource usage.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure single controller."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize thread controller."""
        if not self._initialized:
            self._original_num_threads = None
            self._current_num_threads = None
            self._max_threads = None
            self._initialized = True
            self._setup_initial_config()
    
    def _setup_initial_config(self):
        """Setup initial thread configuration."""
        # Check if already set via environment
        env_threads = os.environ.get('NUMBA_NUM_THREADS')
        force_threads = os.environ.get('SOM_FORCE_THREADS')
        
        if force_threads:
            # Force specific thread count (highest priority)
            try:
                self._max_threads = int(force_threads)
                logger.info(f"FORCED Numba threads via SOM_FORCE_THREADS: {self._max_threads}")
                numba.set_num_threads(self._max_threads)
            except ValueError:
                logger.warning(f"Invalid SOM_FORCE_THREADS value: {force_threads}")
                
        elif env_threads:
            try:
                self._max_threads = int(env_threads)
                logger.info(f"Numba threads set via environment: {self._max_threads}")
            except ValueError:
                logger.warning(f"Invalid NUMBA_NUM_THREADS value: {env_threads}")
                self._max_threads = numba.config.NUMBA_NUM_THREADS
        else:
            self._max_threads = numba.config.NUMBA_NUM_THREADS
        
        self._original_num_threads = numba.get_num_threads()
        self._current_num_threads = self._original_num_threads
        
        logger.info(f"NumbaThreadController initialized: max_threads={self._max_threads}, "
                   f"current={self._current_num_threads}")
    
    def set_threads_from_config(self, config: Optional[Dict[str, Any]] = None):
        """Set number of threads based on SOM configuration.
        
        Args:
            config: Configuration dictionary with n_workers or other settings
        """
        if config is None:
            return
        
        # Check for n_workers in training config
        n_workers = config.get('training', {}).get('n_workers')
        if n_workers is None:
            n_workers = config.get('n_workers')
        
        if n_workers is not None:
            self.set_num_threads(n_workers)
    
    def set_num_threads(self, n_threads: int) -> int:
        """Set number of Numba threads.
        
        Args:
            n_threads: Desired number of threads
            
        Returns:
            Actual number of threads set (may be capped by NUMBA_NUM_THREADS)
        
        Note:
            Can only reduce threads from NUMBA_NUM_THREADS, not increase.
            Changes affect subsequent parallel regions, not currently executing ones.
        """
        if n_threads <= 0:
            logger.warning(f"Invalid thread count {n_threads}, using 1")
            n_threads = 1
        
        # Numba limitation: can't exceed initial NUMBA_NUM_THREADS
        if n_threads > self._max_threads:
            logger.warning(f"Requested {n_threads} threads but max is {self._max_threads}")
            n_threads = self._max_threads
        
        try:
            numba.set_num_threads(n_threads)
            self._current_num_threads = n_threads
            logger.info(f"Set Numba threads to {n_threads}")
            return n_threads
        except Exception as e:
            logger.error(f"Failed to set Numba threads: {e}")
            return self._current_num_threads
    
    def get_num_threads(self) -> int:
        """Get current number of Numba threads.
        
        Returns:
            Current number of active threads
        """
        return numba.get_num_threads()
    
    @contextmanager
    def temporary_threads(self, n_threads: int):
        """Context manager for temporary thread count change.
        
        Args:
            n_threads: Number of threads to use temporarily
            
        Example:
            with controller.temporary_threads(4):
                # Run with 4 threads
                som.train_epoch(...)
            # Automatically restores previous thread count
        """
        previous = self.get_num_threads()
        try:
            self.set_num_threads(n_threads)
            yield
        finally:
            self.set_num_threads(previous)
    
    def optimize_for_data_size(self, n_samples: int, n_features: int, 
                               n_neurons: int) -> int:
        """Automatically optimize thread count based on problem size.
        
        Args:
            n_samples: Number of data samples
            n_features: Number of features per sample
            n_neurons: Number of SOM neurons
            
        Returns:
            Recommended number of threads
        """
        import multiprocessing
        
        # Get system info
        cpu_count = multiprocessing.cpu_count()
        
        # Estimate computational complexity
        # Each sample needs to compute distance to all neurons
        ops_per_epoch = n_samples * n_neurons * n_features
        
        # Memory estimate (MB)
        memory_per_thread = (
            (n_samples * n_features * 4) / (1024 * 1024)  # Data
            + (n_neurons * n_features * 4) / (1024 * 1024)  # Weights
        ) / cpu_count
        
        # Heuristics for thread count
        if n_samples < 1000:
            # Small dataset - limited parallelism benefit
            recommended = min(4, cpu_count)
        elif n_samples < 10000:
            # Medium dataset
            recommended = min(cpu_count // 2, 32)
        elif n_samples < 100000:
            # Large dataset
            recommended = min(cpu_count - 2, 64)  # Leave some cores free
        else:
            # Very large dataset
            recommended = min(cpu_count - 4, 100)  # Leave cores for system
        
        # Adjust for memory pressure
        if memory_per_thread > 1000:  # > 1GB per thread
            recommended = min(recommended, 32)
        
        # Respect configured maximum
        recommended = min(recommended, self._max_threads)
        
        logger.info(f"Recommended threads for {n_samples} samples, "
                   f"{n_neurons} neurons: {recommended}")
        
        return recommended
    
    def reset(self):
        """Reset to original thread configuration."""
        if self._original_num_threads:
            self.set_num_threads(self._original_num_threads)
    
    def get_info(self) -> Dict[str, Any]:
        """Get current thread controller information.
        
        Returns:
            Dictionary with thread configuration info
        """
        return {
            'max_threads': self._max_threads,
            'current_threads': self.get_num_threads(),
            'original_threads': self._original_num_threads,
            'numba_config': numba.config.NUMBA_NUM_THREADS,
            'threading_layer': numba.config.THREADING_LAYER
        }


# Global singleton instance
_thread_controller = None


def get_thread_controller() -> NumbaThreadController:
    """Get the global Numba thread controller.
    
    Returns:
        Singleton NumbaThreadController instance
    """
    global _thread_controller
    if _thread_controller is None:
        _thread_controller = NumbaThreadController()
    return _thread_controller


def setup_numba_threads(config: Optional[Dict[str, Any]] = None,
                        n_workers: Optional[int] = None) -> int:
    """Convenience function to setup Numba threads.
    
    Args:
        config: Configuration dictionary
        n_workers: Explicit number of workers (overrides config)
        
    Returns:
        Number of threads actually set
    """
    controller = get_thread_controller()
    
    if n_workers is not None:
        return controller.set_num_threads(n_workers)
    elif config is not None:
        controller.set_threads_from_config(config)
        return controller.get_num_threads()
    else:
        return controller.get_num_threads()


def optimize_threads_for_som(data_shape: tuple, n_neurons: int) -> int:
    """Optimize thread count for SOM training.
    
    Args:
        data_shape: Shape of training data (n_samples, n_features)
        n_neurons: Number of SOM neurons
        
    Returns:
        Optimal number of threads
    """
    controller = get_thread_controller()
    n_samples, n_features = data_shape
    recommended = controller.optimize_for_data_size(n_samples, n_features, n_neurons)
    controller.set_num_threads(recommended)
    return recommended
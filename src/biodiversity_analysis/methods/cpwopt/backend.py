"""Compute backend for CPU/GPU flexible execution in CP-WOPT.

This module provides a unified interface for CPU and GPU computations,
allowing seamless switching between numpy (CPU) and cupy (GPU) backends.
"""

import numpy as np
from typing import Optional, Union, Literal, Any, List
import logging

logger = logging.getLogger(__name__)


class CPWOPTBackend:
    """Backend manager for CP-WOPT computations.
    
    Provides a unified interface for CPU and GPU operations, automatically
    falling back to CPU if GPU is not available.
    """
    
    def __init__(self, device: Literal['auto', 'cpu', 'gpu'] = 'auto'):
        """Initialize the compute backend.
        
        Args:
            device: Device selection mode
                - 'auto': Use GPU if available, otherwise CPU
                - 'cpu': Force CPU execution
                - 'gpu': Force GPU execution (error if unavailable)
        """
        self.device_type = device
        self.xp = self._setup_backend(device)
        self.device = 'gpu' if self.xp.__name__ == 'cupy' else 'cpu'
        
        logger.info(f"Initialized CP-WOPT backend: {self.device} "
                   f"(requested: {device})")
    
    def _setup_backend(self, device: str):
        """Setup numpy or cupy based on device selection.
        
        Args:
            device: Device selection ('auto', 'cpu', 'gpu')
            
        Returns:
            Array module (numpy or cupy)
            
        Raises:
            RuntimeError: If GPU requested but not available
        """
        if device == 'cpu':
            return np
            
        try:
            import cupy as cp
            if device == 'gpu' or (device == 'auto' and cp.cuda.is_available()):
                logger.info(f"CuPy available, using GPU (device {cp.cuda.Device()})")
                return cp
        except ImportError:
            if device == 'gpu':
                raise RuntimeError(
                    "GPU requested but CuPy not installed. "
                    "Install with: pip install cupy-cuda11x"
                )
            logger.info("CuPy not installed, falling back to CPU")
        
        return np
    
    def get_array_module(self):
        """Get the array module (numpy or cupy).
        
        Returns:
            The array module for computations
        """
        return self.xp
    
    def to_device(self, array: np.ndarray) -> Any:
        """Transfer array to the selected device.
        
        Args:
            array: Input numpy array
            
        Returns:
            Array on the target device
        """
        if array is None:
            return None
            
        if self.device == 'gpu':
            if not isinstance(array, self.xp.ndarray):
                return self.xp.asarray(array)
        return array
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """Transfer array back to CPU if needed.
        
        Args:
            array: Array on any device
            
        Returns:
            Numpy array on CPU
        """
        if array is None:
            return None
            
        if self.device == 'gpu' and hasattr(array, 'get'):
            return array.get()  # CuPy method to get numpy array
        return array
    
    def transfer_factors(self, factors: List[np.ndarray]) -> List[Any]:
        """Transfer factor matrices to device.
        
        Args:
            factors: List of factor matrices
            
        Returns:
            List of factor matrices on device
        """
        return [self.to_device(f) for f in factors]
    
    def transfer_tensor_data(self, tensor_data: Any) -> Any:
        """Transfer tensor data structure to device.
        
        Args:
            tensor_data: TensorData object
            
        Returns:
            TensorData with arrays on device
        """
        if hasattr(tensor_data, 'values'):
            tensor_data.values = self.to_device(tensor_data.values)
        if hasattr(tensor_data, 'weights'):
            tensor_data.weights = self.to_device(tensor_data.weights)
        if hasattr(tensor_data, 'indices'):
            tensor_data.indices = self.to_device(tensor_data.indices)
        return tensor_data
    
    def zeros(self, shape, dtype=np.float64):
        """Create zeros array on device."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float64):
        """Create ones array on device."""
        return self.xp.ones(shape, dtype=dtype)
    
    def dot(self, a, b):
        """Matrix multiplication on device."""
        return self.xp.dot(a, b)
    
    def solve(self, a, b):
        """Solve linear system on device."""
        return self.xp.linalg.solve(a, b)
    
    def einsum(self, subscripts, *operands):
        """Einstein summation on device."""
        return self.xp.einsum(subscripts, *operands)
    
    def get_memory_info(self) -> dict:
        """Get memory information for the device with safety checks.
        
        Returns:
            Dictionary with memory statistics
        """
        info = {
            'device': self.device,
            'available': True,
            'error': None
        }
        
        if self.device == 'gpu':
            try:
                import cupy as cp
                
                # Memory pool info
                mempool = cp.get_default_memory_pool()
                info['used_bytes'] = mempool.used_bytes()
                info['total_bytes'] = mempool.total_bytes()
                
                # Device memory info  
                device = cp.cuda.Device()
                mem_info = device.mem_info
                info['free_memory'] = mem_info[0]
                info['total_memory'] = mem_info[1]
                info['usage_percent'] = (1 - mem_info[0] / mem_info[1]) * 100
                
                # Check for memory pressure
                if info['usage_percent'] > 90:
                    logger.warning(f"High GPU memory usage: {info['usage_percent']:.1f}%")
                    info['memory_pressure'] = 'high'
                elif info['usage_percent'] > 75:
                    info['memory_pressure'] = 'medium'
                else:
                    info['memory_pressure'] = 'low'
                    
            except ImportError as e:
                info['available'] = False
                info['error'] = f"CuPy not available: {e}"
                logger.warning(f"Could not get GPU memory info: {e}")
            except Exception as e:
                info['available'] = False
                info['error'] = str(e)
                logger.error(f"Error getting GPU memory info: {e}")
        else:
            # CPU memory info
            import psutil
            vm = psutil.virtual_memory()
            info['free_memory'] = vm.available
            info['total_memory'] = vm.total
            info['usage_percent'] = vm.percent
            info['memory_pressure'] = 'high' if vm.percent > 90 else 'medium' if vm.percent > 75 else 'low'
        
        return info
    
    def check_memory_available(self, required_bytes: int) -> bool:
        """Check if sufficient memory is available.
        
        Args:
            required_bytes: Number of bytes needed
            
        Returns:
            True if memory available, False otherwise
        """
        info = self.get_memory_info()
        
        if not info.get('available', True):
            return False
            
        free_memory = info.get('free_memory', 0)
        
        # Leave 10% buffer
        available = free_memory * 0.9
        
        if required_bytes > available:
            logger.warning(f"Insufficient memory: need {required_bytes/1e9:.2f}GB, "
                         f"have {available/1e9:.2f}GB available")
            return False
            
        return True
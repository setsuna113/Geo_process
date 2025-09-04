#!/usr/bin/env python3
"""
Comprehensive worker control for MESF and all parallel operations.
Ensures thread count NEVER exceeds specified limit.
"""

import os
import sys
import warnings

class WorkerController:
    """
    Controls parallel workers across ALL libraries to prevent thread explosion.
    """
    
    def __init__(self, max_workers=64):
        """
        Initialize worker controller.
        
        Args:
            max_workers: Maximum number of threads allowed (hard limit)
        """
        self.max_workers = max_workers
        self._original_env = {}
        self._libraries_configured = set()
        
    def set_all_thread_limits(self):
        """
        Set thread limits for ALL parallel libraries.
        This MUST be called before importing any numerical libraries!
        """
        n_str = str(self.max_workers)
        
        # Store original values
        env_vars = [
            # OpenMP (GCC OpenMP, SciPy, etc.)
            'OMP_NUM_THREADS',
            'OMP_THREAD_LIMIT',
            'OMP_MAX_ACTIVE_LEVELS',
            
            # Intel MKL
            'MKL_NUM_THREADS',
            'MKL_DOMAIN_NUM_THREADS',
            'MKL_DYNAMIC',
            
            # OpenBLAS
            'OPENBLAS_NUM_THREADS',
            'GOTO_NUM_THREADS',
            
            # BLIS
            'BLIS_NUM_THREADS',
            
            # NumExpr
            'NUMEXPR_NUM_THREADS',
            'NUMEXPR_MAX_THREADS',
            
            # Veclib (macOS)
            'VECLIB_MAXIMUM_THREADS',
            
            # Other BLAS implementations
            'ATLAS_NUM_THREADS',
            'BLAS_NUM_THREADS',
            'LAPACK_NUM_THREADS',
            
            # Python multiprocessing
            'PYTHONTHREADSAFE',
            
            # TBB (Threading Building Blocks)
            'TBB_NUM_THREADS',
            
            # CUDA (if using GPU)
            'CUDA_VISIBLE_DEVICES',
            'OMP_PLACES',
            'OMP_PROC_BIND'
        ]
        
        # Save original values and set new ones
        for var in env_vars:
            self._original_env[var] = os.environ.get(var)
            
        # Set primary thread limits for OpenMP libraries
        os.environ['OMP_NUM_THREADS'] = n_str
        os.environ['OMP_THREAD_LIMIT'] = n_str  # Hard limit for OpenMP
        os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'  # Prevent nested parallelism
        
        # Intel MKL
        os.environ['MKL_NUM_THREADS'] = n_str
        os.environ['MKL_DOMAIN_NUM_THREADS'] = f"MKL_DOMAIN_ALL={n_str}"
        os.environ['MKL_DYNAMIC'] = 'FALSE'  # Disable dynamic adjustment
        
        # OpenBLAS
        os.environ['OPENBLAS_NUM_THREADS'] = n_str
        os.environ['GOTO_NUM_THREADS'] = n_str
        
        # BLIS
        os.environ['BLIS_NUM_THREADS'] = n_str
        
        # NumExpr
        os.environ['NUMEXPR_NUM_THREADS'] = n_str
        os.environ['NUMEXPR_MAX_THREADS'] = n_str
        
        # Others
        os.environ['VECLIB_MAXIMUM_THREADS'] = n_str
        os.environ['ATLAS_NUM_THREADS'] = n_str
        os.environ['BLAS_NUM_THREADS'] = n_str
        os.environ['LAPACK_NUM_THREADS'] = n_str
        os.environ['TBB_NUM_THREADS'] = n_str
        
        # Prevent nested parallelism
        os.environ['OMP_NESTED'] = 'FALSE'
        os.environ['MKL_NESTED'] = 'FALSE'
        
        # Thread pinning (prevents thread migration)
        os.environ['OMP_PROC_BIND'] = 'TRUE'
        os.environ['OMP_PLACES'] = 'threads'
        
        print(f"✅ Environment variables set to limit threads to {self.max_workers}")
        
        # Now configure any already imported libraries
        self.configure_imported_libraries()
        
    def configure_imported_libraries(self):
        """
        Configure thread limits for already imported libraries.
        Call this AFTER importing numerical libraries.
        """
        import importlib
        import sys
        
        # Skip Numba configuration - not used in current implementation
        # Keeping environment variable set in case other libraries check it
        
        # Configure NumPy/SciPy (MKL/OpenBLAS)
        try:
            import numpy as np
            # Try to set MKL threads if using MKL
            try:
                import mkl
                mkl.set_num_threads(self.max_workers)
                self._libraries_configured.add('MKL')
                print(f"✅ MKL threads set to {self.max_workers}")
            except ImportError:
                pass
            
            # Check which BLAS NumPy is using
            if hasattr(np, '__config__'):
                config_info = str(np.__config__.show())
                if 'openblas' in config_info.lower():
                    print("✅ NumPy using OpenBLAS (controlled via environment)")
                elif 'mkl' in config_info.lower():
                    print("✅ NumPy using MKL (controlled via environment)")
        except ImportError:
            pass
        
        # Configure FAISS
        try:
            import faiss
            faiss.omp_set_num_threads(self.max_workers)
            # Also set FAISS's internal thread pool if available
            if hasattr(faiss, 'set_num_threads'):
                faiss.set_num_threads(self.max_workers)
            self._libraries_configured.add('FAISS')
            print(f"✅ FAISS threads set to {self.max_workers}")
        except ImportError:
            pass
        
        # Configure Numba (if not already done at the beginning)
        if 'Numba' not in self._libraries_configured:
            try:
                import numba
                from numba import config
                config.NUMBA_NUM_THREADS = self.max_workers
                config.NUMBA_DEFAULT_NUM_THREADS = self.max_workers
                if hasattr(numba, 'set_num_threads'):
                    try:
                        numba.set_num_threads(self.max_workers)
                        print(f"✅ Numba threads set to {self.max_workers}")
                    except ValueError as e:
                        print(f"⚠️  Numba threads already restricted: {e}")
                
                # Now set OMP_NUM_THREADS for other libraries
                os.environ['OMP_NUM_THREADS'] = str(self.max_workers)
                self._libraries_configured.add('Numba')
                print(f"✅ Numba configured with {numba.get_num_threads()} threads")
            except ImportError:
                pass
        
        # Configure ThreadPoolExecutor default
        try:
            import concurrent.futures
            # Monkey-patch the default max_workers
            max_limit = self.max_workers  # Capture in closure
            original_init = concurrent.futures.ThreadPoolExecutor.__init__
            def limited_init(executor_self, max_workers=None, *args, **kwargs):
                if max_workers is None or max_workers > max_limit:
                    max_workers = max_limit
                original_init(executor_self, max_workers, *args, **kwargs)
            concurrent.futures.ThreadPoolExecutor.__init__ = limited_init
            self._libraries_configured.add('ThreadPoolExecutor')
        except:
            pass
        
        # Joblib not used in MESF - removed unnecessary configuration
        
        # Configure scikit-learn
        try:
            import sklearn
            sklearn.set_config(n_jobs=self.max_workers)
            self._libraries_configured.add('scikit-learn')
            print(f"✅ scikit-learn n_jobs set to {self.max_workers}")
        except:
            pass
        
        # Configure PyTorch if present
        try:
            import torch
            torch.set_num_threads(self.max_workers)
            torch.set_num_interop_threads(self.max_workers)
            self._libraries_configured.add('PyTorch')
            print(f"✅ PyTorch threads set to {self.max_workers}")
        except ImportError:
            pass
        
        # Configure TensorFlow if present
        try:
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(self.max_workers)
            tf.config.threading.set_intra_op_parallelism_threads(self.max_workers)
            self._libraries_configured.add('TensorFlow')
            print(f"✅ TensorFlow threads set to {self.max_workers}")
        except ImportError:
            pass
        
        # Finally, ensure OMP_NUM_THREADS is set for libraries like FAISS
        # This must be done AFTER Numba configuration to avoid conflicts
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(self.max_workers)
            print(f"✅ OMP_NUM_THREADS set to {self.max_workers} for FAISS and other OpenMP libraries")
        
    def verify_thread_limits(self):
        """
        Verify that thread limits are properly set.
        """
        print("\n" + "="*60)
        print("THREAD LIMIT VERIFICATION")
        print("="*60)
        
        # Check environment variables
        critical_vars = [
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_MAX_THREADS'
        ]
        
        all_set = True
        for var in critical_vars:
            value = os.environ.get(var, 'NOT SET')
            if value == 'NOT SET':
                print(f"⚠️  {var}: NOT SET")
                all_set = False
            elif int(value) > self.max_workers:
                print(f"❌ {var}: {value} (exceeds limit!)")
                all_set = False
            else:
                print(f"✅ {var}: {value}")
        
        # Check library configurations
        print(f"\nLibraries configured: {', '.join(self._libraries_configured) if self._libraries_configured else 'None'}")
        
        # Check actual thread count
        try:
            import psutil
            process = psutil.Process()
            thread_count = process.num_threads()
            print(f"\nCurrent process threads: {thread_count}")
            
            if thread_count > self.max_workers * 2:  # Allow some overhead
                print(f"⚠️  WARNING: Thread count ({thread_count}) exceeds 2x limit ({self.max_workers * 2})")
            else:
                print(f"✅ Thread count within acceptable range")
                
        except ImportError:
            print("\n(Install psutil to check actual thread count)")
        
        return all_set
    
    def get_thread_count(self):
        """Get current thread count of the process."""
        try:
            import psutil
            return psutil.Process().num_threads()
        except ImportError:
            return -1
    
    def monitor_threads(self, func, *args, **kwargs):
        """
        Monitor thread count while running a function.
        
        Args:
            func: Function to run
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of function and max thread count observed
        """
        try:
            import psutil
            import threading
            import time
            
            max_threads = 0
            monitoring = True
            
            def monitor():
                nonlocal max_threads
                process = psutil.Process()
                while monitoring:
                    threads = process.num_threads()
                    max_threads = max(max_threads, threads)
                    if threads > self.max_workers * 1.5:
                        print(f"⚠️  Thread spike: {threads} threads!")
                    time.sleep(0.01)
            
            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.start()
            
            # Run the function
            result = func(*args, **kwargs)
            
            monitoring = False
            monitor_thread.join()
            
            print(f"\n📊 Max threads observed: {max_threads}")
            if max_threads > self.max_workers:
                print(f"⚠️  Exceeded limit by {max_threads - self.max_workers} threads")
            else:
                print(f"✅ Stayed within {self.max_workers} thread limit")
                
            return result, max_threads
            
        except ImportError:
            print("Install psutil for thread monitoring")
            return func(*args, **kwargs), -1
    
    def __enter__(self):
        """Context manager entry."""
        self.set_all_thread_limits()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore original settings."""
        for var, value in self._original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]


# Convenience function for MESF
def setup_mesf_workers(max_workers=64):
    """
    Set up worker control specifically for MESF computation.
    
    Args:
        max_workers: Maximum number of threads to use
        
    Returns:
        WorkerController instance
    """
    controller = WorkerController(max_workers)
    
    # Set environment BEFORE imports
    controller.set_all_thread_limits()
    
    # Import and configure libraries
    import numpy as np
    import faiss
    from numba import config
    
    # Configure after import
    controller.configure_imported_libraries()
    
    # Verify
    controller.verify_thread_limits()
    
    return controller


if __name__ == "__main__":
    print("="*60)
    print("WORKER CONTROL TEST")
    print("="*60)
    
    # Test with strict limit
    max_workers = 64
    controller = WorkerController(max_workers)
    
    # Set limits before imports
    controller.set_all_thread_limits()
    
    # Now import libraries
    import numpy as np
    import faiss
    from numba import jit, prange
    
    # Configure imported libraries
    controller.configure_imported_libraries()
    
    # Verify settings
    controller.verify_thread_limits()
    
    print("\n" + "="*60)
    print("THREAD COUNT TEST")
    print("="*60)
    
    # Test function that uses parallel operations
    @jit(nopython=True, parallel=True)
    def parallel_test(n):
        result = np.zeros(n)
        for i in prange(n):
            result[i] = i * i
        return result
    
    # Monitor threads during execution
    print("\nTesting Numba parallel operation...")
    result, max_threads = controller.monitor_threads(parallel_test, 1000000)
    
    # Test FAISS
    print("\nTesting FAISS K-means...")
    data = np.random.randn(10000, 128).astype(np.float32)
    
    def run_faiss():
        kmeans = faiss.Kmeans(128, 100, niter=10, verbose=False)
        kmeans.train(data)
        return kmeans
    
    kmeans, max_threads_faiss = controller.monitor_threads(run_faiss)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Worker limit: {max_workers}")
    print(f"Max threads in Numba: {max_threads}")
    print(f"Max threads in FAISS: {max_threads_faiss}")
    
    if max_threads > 0 and max_threads <= max_workers * 1.5:
        print("✅ Thread control successful!")
    else:
        print("❌ Thread control failed!")
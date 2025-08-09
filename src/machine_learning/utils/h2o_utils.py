"""H2O utilities for machine learning models.

Centralized utilities for H2O initialization and management to avoid code duplication.
"""

import os
import logging
import h2o
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class H2OManager:
    """Singleton manager for H2O cluster initialization and management."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def ensure_initialized(self, 
                         n_threads: Optional[int] = None,
                         max_mem_size: str = "128G",
                         min_mem_size: str = "8G") -> None:
        """Ensure H2O cluster is initialized.
        
        This method is idempotent - safe to call multiple times.
        
        Args:
            n_threads: Number of threads (None for all available)
            max_mem_size: Maximum memory allocation
            min_mem_size: Minimum memory allocation
        """
        if self._initialized:
            try:
                # Check if cluster is still running
                h2o.cluster()
                return
            except:
                # Cluster died, need to reinitialize
                self._initialized = False
        
        try:
            # Check if H2O is already running
            h2o.cluster()
            logger.info("Using existing H2O cluster")
            self._initialized = True
        except:
            # Initialize new H2O cluster
            if n_threads is None:
                n_threads = os.cpu_count() or 256
            
            logger.info(f"Starting H2O cluster with {n_threads} threads, "
                       f"{max_mem_size} max memory")
            h2o.init(
                nthreads=n_threads,
                max_mem_size=max_mem_size,
                min_mem_size=min_mem_size,
                strict_version_check=False
            )
            self._initialized = True
            
            # Display cluster info
            cluster = h2o.cluster()
            logger.info(f"H2O cluster ready: {cluster}")
    
    def shutdown(self) -> None:
        """Shutdown H2O cluster if running."""
        if self._initialized:
            try:
                h2o.cluster().shutdown()
                logger.info("H2O cluster shutdown")
            except:
                pass
            finally:
                self._initialized = False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current H2O cluster status.
        
        Returns:
            Dictionary with cluster information
        """
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            cluster = h2o.cluster()
            return {
                "status": "running",
                "nodes": len(cluster.nodes),
                "total_memory": cluster.total_memory,
                "free_memory": cluster.free_memory,
                "cores": cluster.cores
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global instance for convenience
h2o_manager = H2OManager()


def ensure_h2o_initialized(n_threads: Optional[int] = None,
                          max_mem_size: str = "128G",
                          min_mem_size: str = "8G") -> None:
    """Convenience function to ensure H2O is initialized.
    
    Args:
        n_threads: Number of threads (None for all available)
        max_mem_size: Maximum memory allocation
        min_mem_size: Minimum memory allocation
    """
    h2o_manager.ensure_initialized(n_threads, max_mem_size, min_mem_size)


def get_h2o_status() -> Dict[str, Any]:
    """Get H2O cluster status.
    
    Returns:
        Dictionary with cluster information
    """
    return h2o_manager.get_cluster_status()
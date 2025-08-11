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
    
    def _validate_memory_setting(self, mem_str: str, mem_type: str, max_limit: Optional[str] = None) -> str:
        """Validate and adjust memory settings based on system resources.
        
        Args:
            mem_str: Memory string like "4G", "256G"
            mem_type: "max" or "min"
            max_limit: For min validation, the max memory setting
            
        Returns:
            Validated memory string
        """
        import psutil
        
        # Parse memory string
        import re
        match = re.match(r'^(\d+)([GM])$', mem_str.upper())
        if not match:
            logger.warning(f"Invalid memory format: {mem_str}, using default 4G")
            return "4G"
        
        requested_val = int(match.group(1))
        unit = match.group(2)
        
        # Convert to bytes
        requested_bytes = requested_val * (1024**3 if unit == 'G' else 1024**2)
        
        # Get available system memory
        available_mem = psutil.virtual_memory().available
        total_mem = psutil.virtual_memory().total
        
        # For H2O, we should use at most 75% of available memory
        safe_available = int(available_mem * 0.75)
        
        # Validate based on type
        if mem_type == "max":
            if requested_bytes > safe_available:
                # Adjust to safe available
                adjusted_gb = safe_available // (1024**3)
                adjusted_gb = max(1, adjusted_gb)  # At least 1GB
                logger.warning(f"Requested {mem_str} exceeds safe available memory ({safe_available/(1024**3):.1f}G). "
                             f"Adjusting to {adjusted_gb}G")
                return f"{adjusted_gb}G"
        else:  # min
            # Min should be at most 25% of max
            if max_limit:
                max_match = re.match(r'^(\d+)([GM])$', max_limit.upper())
                if max_match:
                    max_bytes = int(max_match.group(1)) * (1024**3 if max_match.group(2) == 'G' else 1024**2)
                    if requested_bytes > max_bytes * 0.25:
                        adjusted_gb = max(1, int(max_bytes * 0.25 / (1024**3)))
                        logger.info(f"Adjusting min memory from {mem_str} to {adjusted_gb}G")
                        return f"{adjusted_gb}G"
        
        return mem_str
    
    def ensure_initialized(self, 
                         n_threads: Optional[int] = None,
                         max_mem_size: str = "4G",
                         min_mem_size: str = "1G") -> None:
        """Ensure H2O cluster is initialized.
        
        This method is idempotent - safe to call multiple times.
        
        Args:
            n_threads: Number of threads (None for all available)
            max_mem_size: Maximum memory allocation
            min_mem_size: Minimum memory allocation
        """
        # Validate and adjust memory settings based on system resources
        max_mem_size = self._validate_memory_setting(max_mem_size, "max")
        min_mem_size = self._validate_memory_setting(min_mem_size, "min", max_mem_size)
        # Robust cluster health check
        def is_cluster_healthy() -> bool:
            """Check if H2O cluster is healthy and responsive."""
            try:
                cluster_info = h2o.cluster()
                if cluster_info is None:
                    return False
                
                # Verify cluster is actually responsive with minimal operation
                # Simply check if we can list frames (lightweight operation)
                try:
                    _ = h2o.ls()  # List frames - minimal overhead
                    return True
                except:
                    return False
            except Exception as e:
                logger.debug(f"Cluster health check failed: {e}")
                return False
        
        # Check existing cluster if we think it's initialized
        if self._initialized:
            if is_cluster_healthy():
                logger.info("Using existing healthy H2O cluster")
                return
            else:
                logger.warning("Existing H2O cluster is unhealthy, reinitializing...")
                self._initialized = False
                try:
                    h2o.cluster().shutdown()
                except:
                    pass  # Cluster might already be dead
        
        # Check if there's a cluster we don't know about
        if not self._initialized and is_cluster_healthy():
            logger.info("Found existing healthy H2O cluster")
            self._initialized = True
            return
        
        # Initialize new H2O cluster with retries
        if n_threads is None:
            n_threads = os.cpu_count() or 256
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting H2O cluster (attempt {attempt + 1}/{max_retries}) "
                           f"with {n_threads} threads, {max_mem_size} max memory")
                
                h2o.init(
                    nthreads=n_threads,
                    max_mem_size=max_mem_size,
                    min_mem_size=min_mem_size,
                    strict_version_check=False
                )
                
                # Verify the cluster is actually working
                if is_cluster_healthy():
                    self._initialized = True
                    cluster = h2o.cluster()
                    logger.info(f"H2O cluster ready: {cluster}")
                    return
                else:
                    raise RuntimeError("Cluster started but not healthy")
                    
            except Exception as e:
                logger.error(f"Failed to initialize H2O cluster (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    # Try to clean up before retry
                    try:
                        h2o.cluster().shutdown()
                    except:
                        pass
                else:
                    raise RuntimeError(f"Failed to initialize H2O cluster after {max_retries} attempts") from e
    
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
            # H2O cluster info is in the string representation
            cluster_info = str(cluster).split('\n')
            return {
                "status": "running",
                "nodes": len(cluster.nodes) if hasattr(cluster, 'nodes') else 1,
                "cluster_info": cluster_info
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global instance for convenience
h2o_manager = H2OManager()


def ensure_h2o_initialized(n_threads: Optional[int] = None,
                          max_mem_size: str = "4G",
                          min_mem_size: str = "1G") -> None:
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
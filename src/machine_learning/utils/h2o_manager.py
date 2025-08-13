"""H2O Session Manager to prevent multiple initializations.

This singleton pattern ensures H2O is initialized only once across
the entire application, preventing conflicts and resource waste.
"""

import logging
import h2o
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class H2OManager:
    """Singleton manager for H2O cluster initialization.
    
    Ensures H2O is initialized only once and provides centralized
    configuration management.
    """
    
    _instance = None
    _initialized = False
    _h2o_context = None
    
    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_or_init(cls, **kwargs) -> h2o:
        """Get existing H2O instance or initialize new one.
        
        Args:
            **kwargs: H2O initialization parameters
                max_mem_size: Maximum memory allocation (e.g., '64G')
                nthreads: Number of threads to use
                ice_root: Directory for H2O temporary files
                
        Returns:
            h2o module ready for use
            
        Examples:
            >>> h2o = H2OManager.get_or_init(max_mem_size='64G', nthreads=-1)
            >>> h2o = H2OManager.get_or_init()  # Uses defaults if already initialized
        """
        instance = cls()
        
        # Always check connection status
        try:
            h2o.cluster()
            if not cls._initialized:
                logger.info("H2O cluster already running, connecting to existing instance")
                cls._initialized = True
        except Exception:
            # H2O not running or connection lost, reinitialize
            logger.info(f"Initializing H2O cluster with parameters: {kwargs}")
            
            # Default configuration if not provided
            default_config = {
                'max_mem_size': '32G',
                'nthreads': -1,  # Use all available cores
                'ice_root': '/tmp/h2o_tmp'
            }
            
            # Merge user config with defaults
            config = {**default_config, **kwargs}
            
            try:
                h2o.init(**config)
                cls._initialized = True
                cls._h2o_context = h2o
                
                # Log cluster info
                cluster_info = h2o.cluster()
                logger.info(f"H2O cluster initialized successfully:")
                logger.info(f"  Version: {cluster_info.get('version', 'unknown')}")
                logger.info(f"  Memory: {config['max_mem_size']}")
                logger.info(f"  Threads: {config['nthreads']}")
                
            except Exception as e:
                logger.error(f"Failed to initialize H2O: {e}")
                raise
        
        return h2o
    
    @classmethod
    def shutdown(cls, prompt: bool = False) -> None:
        """Shutdown H2O cluster.
        
        Args:
            prompt: Whether to prompt before shutdown
        """
        if cls._initialized:
            try:
                logger.info("Shutting down H2O cluster")
                h2o.cluster().shutdown(prompt=prompt)
                cls._initialized = False
                cls._h2o_context = None
            except Exception as e:
                logger.warning(f"Error during H2O shutdown: {e}")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if H2O is initialized.
        
        Returns:
            True if H2O is initialized and running
        """
        if not cls._initialized:
            return False
        
        # Double-check that cluster is actually running
        try:
            h2o.cluster()
            return True
        except Exception:
            cls._initialized = False
            return False
    
    @classmethod
    def get_cluster_info(cls) -> Optional[Dict[str, Any]]:
        """Get information about the running H2O cluster.
        
        Returns:
            Dictionary with cluster information or None if not running
        """
        if cls.is_initialized():
            try:
                return h2o.cluster().as_data_frame().to_dict()
            except Exception as e:
                logger.error(f"Failed to get cluster info: {e}")
                return None
        return None
    
    @classmethod
    def remove_all(cls) -> None:
        """Remove all objects from H2O cluster.
        
        Useful for freeing memory without restarting cluster.
        """
        if cls.is_initialized():
            try:
                logger.info("Removing all objects from H2O cluster")
                h2o.remove_all()
            except Exception as e:
                logger.warning(f"Error removing H2O objects: {e}")
    
    @classmethod
    def reset(cls) -> None:
        """Reset the manager state.
        
        Note: This doesn't shutdown H2O, only resets the manager's state.
        Use with caution.
        """
        cls._initialized = False
        cls._h2o_context = None
        logger.warning("H2O manager state reset - H2O cluster may still be running")
"""Configuration loader for k-means hierarchical method."""

from typing import Dict, Any, Optional
from src.abstractions.interfaces.method_config import IMethodConfig


class KMeansHierarchicalConfigLoader(IMethodConfig):
    """Config loader for k-means hierarchical clustering method."""
    
    def load_config(self, config_name: Optional[str] = None, 
                   preset: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration for k-means hierarchical.
        
        Args:
            config_name: Configuration file name (unused for now)
            preset: Preset configuration ('fast', 'balanced', 'thorough')
            
        Returns:
            Configuration dictionary
        """
        # Base configuration
        base_config = {
            'method_params': {
                'n_pre_clusters': 750,
                'use_variance_preservation': True,
                'variance_threshold': 0.95,
                'variance_k_range': (100, 2000),
                'linkage_method': 'ward',
                'final_k_range': (4, 8),
                'final_k': None,
                'selection_method': 'bootstrap',
                'n_bootstrap': 30,
                'bootstrap_sample_ratio': 0.9,
                'use_minibatch': True,
                'batch_size': 10000,
                'distance_metric': 'euclidean',  # Required for ward
                'random_state': 42
            },
            'data_processing': {
                'handle_missing': 'complete_cases',
                'normalization': 'robust',
                'remove_outliers': False,
                'min_features_prefilter': 1
            },
            'visualization': {
                'generate_dendrogram': True,
                'generate_geographic_maps': True,
                'generate_3d_visualization': True,
                'generate_validation_plots': True,
                'n_samples_for_viz': 100000
            },
            'optimization': {
                'chunk_size': 10000,
                'n_jobs': -1,
                'calculate_silhouette': True,
                'silhouette_sample_size': 50000
            }
        }
        
        # Apply presets
        if preset == 'fast':
            base_config['method_params']['n_bootstrap'] = 10
            base_config['method_params']['use_variance_preservation'] = False
            base_config['method_params']['n_pre_clusters'] = 500
            base_config['visualization']['generate_3d_visualization'] = False
            base_config['optimization']['silhouette_sample_size'] = 10000
            
        elif preset == 'balanced':
            # Use defaults (already set)
            pass
            
        elif preset == 'thorough':
            base_config['method_params']['n_bootstrap'] = 100
            base_config['method_params']['variance_threshold'] = 0.97
            base_config['method_params']['final_k_range'] = (3, 10)
            base_config['optimization']['silhouette_sample_size'] = 100000
            
        elif preset == 'publication':
            base_config['method_params']['n_bootstrap'] = 100
            base_config['visualization']['dpi'] = 300
            base_config['visualization']['generate_all'] = True
            base_config['method_params']['variance_threshold'] = 0.95
            
        return base_config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return self.load_config(preset='balanced')
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        params = config.get('method_params', {})
        
        # Validate n_pre_clusters
        if 'n_pre_clusters' in params:
            n = params['n_pre_clusters']
            if not isinstance(n, int) or n < 10:
                raise ValueError(f"n_pre_clusters must be int >= 10, got {n}")
        
        # Validate variance threshold
        if 'variance_threshold' in params:
            v = params['variance_threshold']
            if not 0 < v <= 1:
                raise ValueError(f"variance_threshold must be in (0, 1], got {v}")
        
        # Validate linkage method
        if 'linkage_method' in params:
            valid = ['ward', 'complete', 'average', 'single']
            if params['linkage_method'] not in valid:
                raise ValueError(f"linkage_method must be one of {valid}")
        
        # Ward requires euclidean distance
        if params.get('linkage_method') == 'ward':
            if params.get('distance_metric') != 'euclidean':
                params['distance_metric'] = 'euclidean'
        
        return True
    
    def get_env_var_names(self) -> list:
        """Get list of environment variable names.
        
        Returns:
            List of environment variable names
        """
        return []  # No environment variables used
    
    def get_env_var_descriptions(self) -> dict:
        """Get descriptions of environment variables.
        
        Returns:
            Dictionary of variable names to descriptions
        """
        return {}  # No environment variables used
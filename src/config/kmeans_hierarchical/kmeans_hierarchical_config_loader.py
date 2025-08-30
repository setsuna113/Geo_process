"""Configuration loader for k-means hierarchical method."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from src.core_abstractions import IMethodConfig


class KMeansHierarchicalConfigLoader(IMethodConfig):
    """Config loader for k-means hierarchical clustering method."""
    
    def load_config(self, config_name: Optional[str] = None, 
                   preset: Optional[str] = None,
                   clustering_approach: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration for k-means hierarchical.
        
        Loads from YAML file first, then applies overrides.
        
        Args:
            config_name: Configuration file name (default: kmeans_hierarchical_config.yml)
            preset: Preset configuration ('fast', 'balanced', 'thorough')
            clustering_approach: Override for clustering approach
            
        Returns:
            Configuration dictionary
        """
        # Load from YAML file
        config_dir = Path(__file__).parent
        config_file = config_dir / (config_name or 'kmeans_hierarchical_config.yml')
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
        else:
            yaml_config = {}
        
        # Extract sections from YAML
        method_params = yaml_config.get('method_params', {})
        data_processing = yaml_config.get('data_processing', {})
        visualization = yaml_config.get('visualization', {})
        performance = yaml_config.get('performance', {})
        
        # Get clustering approach from YAML or override
        approach = clustering_approach or method_params.get('clustering_approach', 'ecological')
        
        # AUTOMATICALLY set linkage and distance based on approach
        if approach == 'geometric':
            linkage = 'ward'
            distance = 'euclidean'
        elif approach == 'ecological':
            linkage = 'average'
            distance = 'braycurtis'
        else:
            raise ValueError(f"Invalid clustering_approach: {approach}. Use 'geometric' or 'ecological'")
        
        # Override linkage and distance in method_params
        method_params['linkage_method'] = linkage
        method_params['distance_metric'] = distance
        method_params['clustering_approach'] = approach
        
        # Build complete config
        config = {
            'method_params': method_params,
            'data_processing': data_processing,
            'visualization': visualization,
            'optimization': {
                'chunk_size': performance.get('memory_chunk_size', 50000),
                'n_jobs': method_params.get('n_jobs', -1),
                'calculate_silhouette': visualization.get('calculate_silhouette', True),
                'silhouette_sample_size': visualization.get('silhouette_sample_size', 100000)
            }
        }
        
        # Apply presets if specified
        if preset == 'fast':
            config['method_params']['n_bootstrap'] = 10
            config['method_params']['use_elbow_method'] = False
            config['method_params']['n_pre_clusters'] = 500
            config['visualization']['generate_3d_visualization'] = False
            config['optimization']['silhouette_sample_size'] = 10000
            
        elif preset == 'thorough':
            config['method_params']['n_bootstrap'] = 100
            config['method_params']['elbow_threshold'] = 0.005  # 0.5% for high quality
            config['optimization']['silhouette_sample_size'] = 100000
            
        return config
    
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
        
        # Validate elbow threshold
        if 'elbow_threshold' in params:
            v = params['elbow_threshold']
            if not 0 < v <= 1:
                raise ValueError(f"elbow_threshold must be in (0, 1], got {v}")
        
        # Validate clustering approach and enforce valid combinations
        linkage = params.get('linkage_method', 'average')
        distance = params.get('distance_metric', 'braycurtis')
        approach = params.get('clustering_approach', 'ecological')
        
        # Define valid combinations
        valid_combinations = {
            'geometric': ('ward', 'euclidean'),
            'ecological': ('average', 'braycurtis')
        }
        
        # Check if approach is valid
        if approach not in valid_combinations:
            raise ValueError(f"Invalid clustering_approach: {approach}. Use 'geometric' or 'ecological'")
        
        # Get expected combination
        expected_linkage, expected_distance = valid_combinations[approach]
        
        # Validate combination
        if linkage != expected_linkage or distance != expected_distance:
            raise ValueError(
                f"Invalid combination for {approach} approach.\n"
                f"Expected: {expected_linkage} + {expected_distance}\n"
                f"Got: {linkage} + {distance}\n"
                f"Valid combinations:\n"
                f"  - geometric: ward + euclidean\n"
                f"  - ecological: average + braycurtis"
            )
        
        # Validate linkage method is in allowed list
        valid_linkages = ['ward', 'complete', 'average', 'single']
        if linkage not in valid_linkages:
            raise ValueError(f"linkage_method must be one of {valid_linkages}")
        
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
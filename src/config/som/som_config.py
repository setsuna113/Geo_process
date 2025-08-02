"""SOM Configuration Module

This module handles loading and managing SOM-specific configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SOMConfig:
    """Manages SOM-specific configuration with YAML override support."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize SOM configuration.
        
        Args:
            config_file: Path to som_config.yml. If None, searches default locations.
        """
        self.config_file = config_file or self._find_config_file()
        self.settings = self._load_config()
        
    def _find_config_file(self) -> Path:
        """Find som_config.yml in standard locations."""
        # Define search paths
        search_paths = [
            Path(__file__).parent / 'som_config.yml',  # Same directory as this file
            Path(__file__).parent.parent.parent.parent / 'som_config.yml',  # Old location
            Path.cwd() / 'som_config.yml',  # Current directory
            Path.home() / '.geo' / 'som_config.yml',  # User config directory
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found SOM config at: {path}")
                return path
                
        # If not found, use default location
        default_path = Path(__file__).parent / 'som_config.yml'
        logger.warning(f"No som_config.yml found, will use defaults at: {default_path}")
        return default_path
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or return defaults."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded SOM configuration from {self.config_file}")
                return config
        else:
            logger.warning("Using default SOM configuration")
            return self._get_defaults()
            
    def _get_defaults(self) -> Dict[str, Any]:
        """Return default SOM configuration."""
        return {
            'distance_config': {
                'input_space': 'bray_curtis',
                'missing_data_handling': 'pairwise',
                'min_valid_features': 2,
                'map_space': 'euclidean'
            },
            'preprocessing_config': {
                'transformation': 'log1p',
                'standardization': 'z_score_by_type',
                'missing_data': 'keep_nan',
                'spatial_sampling': {
                    'method': 'block_sampling',
                    'block_size': '750km',
                    'for_data_at': '100km_resolution'
                }
            },
            'architecture_config': {
                'type': 'GeoSOM_VLRSOM',
                'spatial_weight': 0.3,
                'geographic_distance': 'haversine',
                'combine_distances': 'weighted_sum',
                'initial_learning_rate': 0.5,
                'min_learning_rate': 0.001,
                'max_learning_rate': 0.8,
                'lr_increase_factor': 1.05,
                'lr_decrease_factor': 0.90,
                'high_qe_lr_range': [0.3, 0.6],
                'low_qe_lr_range': [0.001, 0.05],
                'neighborhood_function': 'gaussian',
                'initial_radius': None,
                'final_radius': 1.0,
                'radius_decay': 'linear',
                'topology': 'rectangular',
                'grid_size': 'determined_by_data',
                'convergence': {
                    'geographic_coherence_threshold': 0.6,
                    'lr_stability_threshold': 0.05,
                    'qe_improvement_threshold': 0.005,
                    'patience': 20,
                    'max_epochs': 200
                }
            },
            'training_config': {
                'mode': 'batch',
                'parallel_processing': True,
                'n_cores': 'auto',
                'memory_management': {
                    'chunk_if_exceeds': '8GB',
                    'chunk_size': 50000
                }
            },
            'validation_config': {
                'method': 'spatial_block_cv',
                'n_folds': 3,
                'block_size': '750km',
                'stratification': 'ensure_all_biodiversity_types',
                'metrics': [
                    'quantization_error',
                    'topographic_error',
                    'geographic_coherence',
                    'beta_diversity_preservation'
                ]
            },
            'sampling_config': {
                'qe_calculation': {
                    'sample_size': 100000,
                    'full_qe_frequency': 5
                },
                'geographic_coherence': {
                    'sample_size': 5000,
                    'calculation_frequency': 10
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'architecture_config.convergence.max_epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_distance_config(self) -> Dict[str, Any]:
        """Get distance calculation configuration."""
        return self.settings.get('distance_config', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.settings.get('preprocessing_config', {})
    
    def get_architecture_config(self) -> Dict[str, Any]:
        """Get architecture configuration."""
        return self.settings.get('architecture_config', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.settings.get('training_config', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.settings.get('validation_config', {})
    
    def get_sampling_config(self) -> Dict[str, Any]:
        """Get sampling configuration."""
        return self.settings.get('sampling_config', {})
    
    def get_convergence_config(self) -> Dict[str, Any]:
        """Get convergence configuration."""
        arch_config = self.get_architecture_config()
        return arch_config.get('convergence', {})
    
    def get_chunk_size(self) -> int:
        """Get chunk size for batch processing."""
        training_config = self.get_training_config()
        memory_mgmt = training_config.get('memory_management', {})
        return memory_mgmt.get('chunk_size', 50000)
    
    def get_qe_sample_size(self) -> int:
        """Get sample size for QE calculation."""
        sampling_config = self.get_sampling_config()
        qe_config = sampling_config.get('qe_calculation', {})
        return qe_config.get('sample_size', 100000)


# Global SOM configuration instance
som_config = SOMConfig()
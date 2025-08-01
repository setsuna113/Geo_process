"""
Biodiversity Analysis Configuration System

A modular, extensible configuration system for biodiversity analysis methods.
This separates biodiversity-specific configs from the main system config.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiodiversityConfig:
    """Configuration container for biodiversity analysis."""
    
    # Common data processing settings
    data_processing: Dict[str, Any] = field(default_factory=lambda: {
        'missing_value_strategy': 'adaptive',  # adaptive, mean, median, zero
        'normalization_method': 'standard',    # standard, minmax
        'handle_zero_inflation': True,
        'zero_inflation_threshold': 0.5,
        'remove_constant_features': True,
        'outlier_detection': 'iqr',           # iqr, zscore, none
        'outlier_threshold': 3.0
    })
    
    # Spatial validation settings
    spatial_validation: Dict[str, Any] = field(default_factory=lambda: {
        'strategy': 'random_blocks',  # random_blocks, systematic_blocks, latitudinal
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'autocorrelation_test': True,
        'min_spatial_distance': 0.0,
        'cv_folds': 5,
        'cv_buffer_size': None
    })
    
    # Method-specific configurations
    methods: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default method configs if not provided."""
        if not self.methods:
            self.methods = self._get_default_methods()
    
    def _get_default_methods(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations for each analysis method."""
        return {
            'som': {
                'grid_size': [10, 10],
                'max_iterations': 2000,
                'learning_rate': 0.5,
                'sigma': 1.0,
                'distance_metric': 'manhattan',  # Better for species data
                'topology': 'hexagonal',
                'neighborhood_function': 'gaussian',
                'early_stopping': True,
                'patience': 50,
                'min_improvement': 1e-6
            },
            'gwpca': {
                'n_components': 2,
                'bandwidth_method': 'AICc',
                'adaptive_bandwidth': True,
                'kernel': 'bisquare',
                'standardize': True,
                'max_iterations': 200,
                'convergence_tolerance': 1e-5
            },
            'maxp': {
                'min_region_size': 5,
                'max_region_size': None,
                'contiguity': 'queen',  # queen, rook
                'objective_function': 'variance',
                'n_iterations': 100,
                'cooling_rate': 0.85
            }
        }
    
    def get_method_config(self, method: str) -> Dict[str, Any]:
        """Get configuration for a specific method."""
        return self.methods.get(method, {})
    
    def register_method(self, method: str, config: Dict[str, Any]) -> None:
        """Register a new analysis method configuration."""
        self.methods[method] = config
        logger.info(f"Registered configuration for method: {method}")
    
    def update_method_config(self, method: str, updates: Dict[str, Any]) -> None:
        """Update configuration for an existing method."""
        if method not in self.methods:
            self.methods[method] = {}
        self.methods[method].update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data_processing': self.data_processing,
            'spatial_validation': self.spatial_validation,
            'methods': self.methods
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiodiversityConfig':
        """Create from dictionary."""
        return cls(
            data_processing=data.get('data_processing', {}),
            spatial_validation=data.get('spatial_validation', {}),
            methods=data.get('methods', {})
        )


class BiodiversityConfigManager:
    """Manager for biodiversity analysis configurations."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize with optional config file."""
        self.config = BiodiversityConfig()
        self.config_file = config_file
        
        if config_file and config_file.exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'biodiversity_analysis' in data:
                    bio_data = data['biodiversity_analysis']
                    self.config = BiodiversityConfig.from_dict(bio_data)
                    logger.info(f"Loaded biodiversity config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load biodiversity config: {e}")
    
    def save_to_file(self, config_file: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        file_to_use = config_file or self.config_file
        if not file_to_use:
            raise ValueError("No config file specified")
        
        data = {'biodiversity_analysis': self.config.to_dict()}
        
        with open(file_to_use, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Saved biodiversity config to {file_to_use}")
    
    def get_config(self) -> BiodiversityConfig:
        """Get the current configuration."""
        return self.config
    
    def register_custom_method(self, method: str, config: Dict[str, Any]) -> None:
        """Register a custom analysis method."""
        self.config.register_method(method, config)
    
    def get_method_config(self, method: str) -> Dict[str, Any]:
        """Get config for a specific method with common settings merged."""
        method_config = self.config.get_method_config(method)
        
        # Merge with common settings
        return {
            'data_processing': self.config.data_processing.copy(),
            'spatial_validation': self.config.spatial_validation.copy(),
            'method_params': method_config
        }


# Global instance for easy access
_biodiversity_config_manager: Optional[BiodiversityConfigManager] = None


def get_biodiversity_config() -> BiodiversityConfigManager:
    """Get or create the global biodiversity config manager."""
    global _biodiversity_config_manager
    
    if _biodiversity_config_manager is None:
        # Try to load from standard locations
        project_root = Path(__file__).parent.parent.parent
        potential_files = [
            project_root / 'biodiversity_config.yml',
            project_root / 'config' / 'biodiversity.yml',
            Path.cwd() / 'biodiversity_config.yml'
        ]
        
        config_file = None
        for f in potential_files:
            if f.exists():
                config_file = f
                break
        
        _biodiversity_config_manager = BiodiversityConfigManager(config_file)
    
    return _biodiversity_config_manager


# Convenience function for method configs
def get_method_config(method: str) -> Dict[str, Any]:
    """Get configuration for a specific analysis method."""
    manager = get_biodiversity_config()
    return manager.get_method_config(method)
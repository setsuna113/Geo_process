"""Tests for biodiversity configuration system."""

import pytest
from pathlib import Path
import tempfile
import yaml

from src.config.biodiversity_config import (
    BiodiversityConfig, 
    BiodiversityConfigManager,
    get_biodiversity_config,
    get_method_config
)


class TestBiodiversityConfig:
    """Test BiodiversityConfig dataclass."""
    
    def test_default_initialization(self):
        """Test config initializes with defaults."""
        config = BiodiversityConfig()
        
        # Check data processing defaults
        assert config.data_processing['missing_value_strategy'] == 'adaptive'
        assert config.data_processing['normalization_method'] == 'standard'
        assert config.data_processing['handle_zero_inflation'] is True
        
        # Check spatial validation defaults
        assert config.spatial_validation['strategy'] == 'random_blocks'
        assert config.spatial_validation['train_ratio'] == 0.7
        
        # Check method defaults
        assert 'som' in config.methods
        assert 'gwpca' in config.methods
        assert 'maxp' in config.methods
    
    def test_get_method_config(self):
        """Test getting method-specific config."""
        config = BiodiversityConfig()
        
        som_config = config.get_method_config('som')
        assert som_config['grid_size'] == [10, 10]
        assert som_config['distance_metric'] == 'manhattan'
        
        # Non-existent method returns empty dict
        assert config.get_method_config('nonexistent') == {}
    
    def test_register_method(self):
        """Test registering new method."""
        config = BiodiversityConfig()
        
        custom_config = {
            'algorithm': 'custom',
            'param1': 42
        }
        
        config.register_method('custom_method', custom_config)
        assert config.get_method_config('custom_method') == custom_config
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        config = BiodiversityConfig()
        config.register_method('test', {'param': 'value'})
        
        # Serialize
        data = config.to_dict()
        assert 'data_processing' in data
        assert 'spatial_validation' in data
        assert 'methods' in data
        assert data['methods']['test']['param'] == 'value'
        
        # Deserialize
        config2 = BiodiversityConfig.from_dict(data)
        assert config2.get_method_config('test')['param'] == 'value'


class TestBiodiversityConfigManager:
    """Test BiodiversityConfigManager."""
    
    def test_initialization_without_file(self):
        """Test manager initializes without config file."""
        manager = BiodiversityConfigManager()
        config = manager.get_config()
        
        assert isinstance(config, BiodiversityConfig)
        assert 'som' in config.methods
    
    def test_load_from_file(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml_content = {
                'biodiversity_analysis': {
                    'data_processing': {
                        'missing_value_strategy': 'mean'
                    },
                    'methods': {
                        'som': {
                            'grid_size': [20, 20]
                        }
                    }
                }
            }
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)
        
        try:
            manager = BiodiversityConfigManager(temp_path)
            config = manager.get_config()
            
            # Check loaded values
            assert config.data_processing['missing_value_strategy'] == 'mean'
            assert config.methods['som']['grid_size'] == [20, 20]
            
        finally:
            temp_path.unlink()
    
    def test_save_to_file(self):
        """Test saving config to YAML file."""
        manager = BiodiversityConfigManager()
        manager.register_custom_method('test_method', {'param': 123})
        
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            manager.save_to_file(temp_path)
            
            # Load and verify
            with open(temp_path) as f:
                data = yaml.safe_load(f)
            
            assert 'biodiversity_analysis' in data
            assert data['biodiversity_analysis']['methods']['test_method']['param'] == 123
            
        finally:
            temp_path.unlink()
    
    def test_get_method_config_merged(self):
        """Test getting method config with common settings merged."""
        manager = BiodiversityConfigManager()
        
        som_config = manager.get_method_config('som')
        
        # Should have common settings
        assert 'data_processing' in som_config
        assert 'spatial_validation' in som_config
        assert 'method_params' in som_config
        
        # Method params should have SOM-specific settings
        assert som_config['method_params']['distance_metric'] == 'manhattan'


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_biodiversity_config(self):
        """Test global config manager access."""
        manager1 = get_biodiversity_config()
        manager2 = get_biodiversity_config()
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_get_method_config(self):
        """Test convenience function for method config."""
        config = get_method_config('som')
        
        assert 'data_processing' in config
        assert 'method_params' in config
        assert config['method_params']['grid_size'] == [10, 10]
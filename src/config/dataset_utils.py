# src/config/dataset_utils.py
"""Utilities for dataset configuration handling with backward compatibility."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DatasetPathResolver:
    """Resolves dataset paths with backward compatibility for path_key and direct path approaches."""
    
    def __init__(self, config):
        self.config = config
        self.data_files = config.get('data_files', {})
        self.data_dir = Path(config.get('paths.data_dir', 'data'))
    
    def resolve_dataset_path(self, dataset_config: Dict[str, Any]) -> Path:
        """
        Resolve dataset path with backward compatibility.
        
        Priority order:
        1. Direct 'path' field (new approach)
        2. 'path_key' lookup in data_files (legacy approach)
        3. Fallback to data_dir/name pattern
        
        Args:
            dataset_config: Dataset configuration dictionary
            
        Returns:
            Path: Resolved file path
            
        Raises:
            ValueError: If path cannot be resolved
        """
        dataset_name = dataset_config.get('name', 'unknown')
        
        # Method 1: Direct path (preferred)
        if 'path' in dataset_config:
            path = Path(dataset_config['path'])
            
            # Handle relative paths by combining with data_dir
            if not path.is_absolute():
                path = self.data_dir / path
                
            logger.debug(f"Dataset '{dataset_name}': resolved via direct path -> {path}")
            return path
        
        # Method 2: Legacy path_key lookup
        if 'path_key' in dataset_config:
            path_key = dataset_config['path_key']
            
            if path_key in self.data_files:
                filename = self.data_files[path_key]
                path = self.data_dir / filename
                logger.debug(f"Dataset '{dataset_name}': resolved via path_key '{path_key}' -> {path}")
                return path
            else:
                logger.warning(f"Dataset '{dataset_name}': path_key '{path_key}' not found in data_files")
        
        # Method 3: Fallback - try to construct from name
        if dataset_name != 'unknown':
            # Try common patterns
            potential_files = [
                f"{dataset_name}.tif",
                f"{dataset_name.replace('-', '_')}.tif",
                f"{dataset_name.replace('_', '-')}.tif"
            ]
            
            for filename in potential_files:
                path = self.data_dir / filename
                if path.exists():
                    logger.info(f"Dataset '{dataset_name}': auto-resolved to {path}")
                    return path
        
        raise ValueError(f"Cannot resolve path for dataset '{dataset_name}'. "
                        f"Provide either 'path' or valid 'path_key' in configuration.")
    
    def validate_dataset_config(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize dataset configuration.
        
        Args:
            dataset_config: Raw dataset configuration
            
        Returns:
            Dict: Validated and normalized configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['name', 'data_type', 'band_name']
        
        # Check required fields
        for field in required_fields:
            if field not in dataset_config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data type
        valid_data_types = ['richness_data', 'continuous_data', 'categorical_data']
        data_type = dataset_config.get('data_type')
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type '{data_type}'. Must be one of: {valid_data_types}")
        
        # Validate band name (database-safe)
        band_name = dataset_config.get('band_name', '')
        if not band_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Band name '{band_name}' contains invalid characters. "
                           f"Use only letters, numbers, hyphens, and underscores.")
        
        # Resolve and validate file path
        try:
            resolved_path = self.resolve_dataset_path(dataset_config)
            if not resolved_path.exists():
                raise ValueError(f"Dataset file not found: {resolved_path}")
        except Exception as e:
            raise ValueError(f"Path resolution failed for dataset '{dataset_config.get('name')}': {e}")
        
        # Create normalized config
        normalized_config = dataset_config.copy()
        normalized_config['resolved_path'] = str(resolved_path)
        normalized_config['enabled'] = dataset_config.get('enabled', True)
        
        return normalized_config
    
    def get_enabled_datasets(self) -> List[Dict[str, Any]]:
        """
        Get all enabled and validated dataset configurations.
        
        Returns:
            List[Dict]: List of normalized dataset configurations
        """
        datasets = self.config.get('datasets.target_datasets', [])
        enabled_datasets = []
        
        for dataset_config in datasets:
            if not dataset_config.get('enabled', True):
                continue
                
            try:
                normalized_config = self.validate_dataset_config(dataset_config)
                enabled_datasets.append(normalized_config)
            except ValueError as e:
                logger.error(f"Skipping invalid dataset configuration: {e}")
                continue
        
        return enabled_datasets
    
    def generate_database_fields(self) -> Dict[str, str]:
        """
        Generate database field mappings from enabled datasets.
        
        Returns:
            Dict: Mapping of band_name -> resolved_path for all enabled datasets
        """
        enabled_datasets = self.get_enabled_datasets()
        fields = {}
        
        for dataset in enabled_datasets:
            band_name = dataset['band_name']
            resolved_path = dataset['resolved_path']
            fields[band_name] = resolved_path
            
        logger.info(f"Generated {len(fields)} database fields: {list(fields.keys())}")
        return fields


def create_dataset_resolver(config) -> DatasetPathResolver:
    """Factory function to create a dataset path resolver."""
    return DatasetPathResolver(config)
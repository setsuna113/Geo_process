import logging
# src/pipelines/unified_resampling/dataset_processor.py
"""Dataset-specific processing utilities for resampling pipeline."""

from typing import List, Tuple, Dict, Any, List, Optional, Tuple

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.raster_data.catalog import RasterCatalog

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Handles dataset-specific operations for the resampling pipeline."""
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        self.config = config
        self.db = db_connection
        self.catalog = RasterCatalog(db_connection, config)
        
        # Dataset type mappings
        self.data_type_handlers = {
            'richness_data': self._handle_richness_data,
            'continuous_data': self._handle_continuous_data,
            'categorical_data': self._handle_categorical_data
        }
    
    def prepare_dataset_for_resampling(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare dataset for resampling based on its type.
        
        Args:
            dataset_config: Dataset configuration dictionary
            
        Returns:
            Enhanced dataset configuration with preprocessing parameters
        """
        data_type = dataset_config.get('data_type', 'continuous_data')
        dataset_name = dataset_config['name']
        
        logger.info(f"Preparing {data_type} dataset: {dataset_name}")
        
        # Get data type handler
        handler = self.data_type_handlers.get(data_type, self._handle_continuous_data)
        
        # Apply type-specific preprocessing
        enhanced_config = handler(dataset_config)
        
        return enhanced_config
    
    def _handle_richness_data(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle species richness data preprocessing."""
        enhanced_config = dataset_config.copy()
        
        # Richness data specific settings
        enhanced_config.update({
            'preprocessing': {
                'clip_negative_values': True,  # Richness cannot be negative
                'integer_values': True,        # Richness should be integers
                'nodata_handling': 'zero',     # Missing areas = 0 richness
                'preserve_sum': True           # Important for count data
            },
            'validation': {
                'min_value': 0,
                'max_value': 50000,  # Reasonable upper bound for richness
                'check_integer': True
            }
        })
        
        logger.info(f"Configured richness data preprocessing for {dataset_config['name']}")
        return enhanced_config
    
    def _handle_continuous_data(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle continuous data (e.g., climate variables) preprocessing."""
        enhanced_config = dataset_config.copy()
        
        # Continuous data specific settings
        enhanced_config.update({
            'preprocessing': {
                'outlier_detection': True,
                'smooth_interpolation': True,
                'nodata_handling': 'interpolate',
                'preserve_sum': False
            },
            'validation': {
                'check_outliers': True,
                'outlier_threshold': 3.0  # Standard deviations
            }
        })
        
        logger.info(f"Configured continuous data preprocessing for {dataset_config['name']}")
        return enhanced_config
    
    def _handle_categorical_data(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle categorical data preprocessing."""
        enhanced_config = dataset_config.copy()
        
        # Categorical data specific settings
        enhanced_config.update({
            'preprocessing': {
                'mode_aggregation': True,
                'category_preservation': True,
                'nodata_handling': 'majority',
                'preserve_sum': False
            },
            'validation': {
                'check_categories': True,
                'valid_categories': None  # To be populated from data
            }
        })
        
        logger.info(f"Configured categorical data preprocessing for {dataset_config['name']}")
        return enhanced_config
    
    def validate_dataset_compatibility(self, datasets_info: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Validate that datasets are compatible for merging.
        
        Args:
            datasets_info: List of dataset information dictionaries
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(datasets_info) < 2:
            return False, "Need at least 2 datasets for merging"
        
        # Check target resolution consistency
        target_resolutions = set()
        target_crs_set = set()
        
        for info in datasets_info:
            target_resolutions.add(info.get('target_resolution'))
            target_crs_set.add(info.get('target_crs'))
        
        if len(target_resolutions) > 1:
            return False, f"Inconsistent target resolutions: {target_resolutions}"
        
        if len(target_crs_set) > 1:
            return False, f"Inconsistent target CRS: {target_crs_set}"
        
        # Check for duplicate band names
        band_names = [info.get('band_name') for info in datasets_info]
        if len(band_names) != len(set(band_names)):
            duplicates = [name for name in band_names if band_names.count(name) > 1]
            return False, f"Duplicate band names: {duplicates}"
        
        # Check spatial compatibility (bounds overlap)
        valid_bounds = [info['bounds'] for info in datasets_info if 'bounds' in info and info['bounds'] is not None]
        if valid_bounds:
            overlap_bounds = self._calculate_bounds_overlap(valid_bounds)
            if overlap_bounds is None:
                return False, "Datasets have no spatial overlap"
        
        return True, None
    
    def _calculate_bounds_overlap(self, bounds_list: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
        """Calculate overlapping bounds from multiple datasets."""
        if not bounds_list:
            return None
        
        # Find intersection of all bounds
        minx = max(bounds[0] for bounds in bounds_list)
        miny = max(bounds[1] for bounds in bounds_list)
        maxx = min(bounds[2] for bounds in bounds_list)
        maxy = min(bounds[3] for bounds in bounds_list)
        
        # Check if there's actually an overlap
        if minx >= maxx or miny >= maxy:
            return None
        
        return (minx, miny, maxx, maxy)
    
    def estimate_processing_requirements(self, datasets_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate processing requirements for the pipeline.
        
        Args:
            datasets_config: List of dataset configurations
            
        Returns:
            Dictionary with processing estimates
        """
        target_resolution = self.config.get('resampling.target_resolution', 0.05)
        
        total_datasets = len([ds for ds in datasets_config if ds.get('enabled', True)])
        
        # Estimate output dimensions based on target resolution and typical global bounds
        # This is a rough estimate - actual bounds may vary
        typical_global_bounds = (-180, -90, 180, 90)
        estimated_width = int((typical_global_bounds[2] - typical_global_bounds[0]) / target_resolution)
        estimated_height = int((typical_global_bounds[3] - typical_global_bounds[1]) / target_resolution)
        
        estimated_pixels_per_dataset = estimated_width * estimated_height
        
        # Memory estimates (rough calculations)
        bytes_per_pixel = 4  # float32
        memory_per_dataset_mb = (estimated_pixels_per_dataset * bytes_per_pixel) / (1024 * 1024)
        total_memory_mb = memory_per_dataset_mb * total_datasets
        
        # Processing time estimates (very rough)
        base_time_per_dataset_minutes = 5  # Base processing time
        resolution_factor = (0.1 / target_resolution) ** 2  # Scale with resolution
        estimated_time_minutes = base_time_per_dataset_minutes * total_datasets * resolution_factor
        
        return {
            'datasets_count': total_datasets,
            'target_resolution': target_resolution,
            'estimated_dimensions': {
                'width': estimated_width,
                'height': estimated_height,
                'pixels_per_dataset': estimated_pixels_per_dataset
            },
            'memory_estimates': {
                'per_dataset_mb': round(memory_per_dataset_mb, 2),
                'total_mb': round(total_memory_mb, 2),
                'total_gb': round(total_memory_mb / 1024, 2)
            },
            'time_estimates': {
                'total_minutes': round(estimated_time_minutes, 1),
                'total_hours': round(estimated_time_minutes / 60, 2)
            },
            'disk_space_estimates': {
                'database_storage_mb': round(total_memory_mb * 1.5, 2),  # Include metadata overhead
                'output_files_mb': round(total_memory_mb * 0.8, 2)  # Compressed outputs
            }
        }
    
    def get_dataset_statistics(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific dataset."""
        try:
            # Try to get from catalog first
            raster_entry = self.catalog.get_raster(dataset_name)
            if raster_entry:
                return {
                    'source_path': str(raster_entry.path),
                    'resolution_degrees': raster_entry.resolution_degrees,
                    'bounds': raster_entry.bounds,
                    'data_type': raster_entry.data_type,
                    'file_size_mb': raster_entry.file_size_mb,
                    'last_validated': raster_entry.last_validated
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get statistics for dataset {dataset_name}: {e}")
            return None
    
    def create_dataset_summary_report(self, datasets_info: List[Dict[str, Any]]) -> str:
        """Create a summary report of all datasets."""
        report_lines = [
            "Dataset Summary Report",
            "=" * 50,
            f"Total Datasets: {len(datasets_info)}",
            f"Target Resolution: {self.config.get('resampling.target_resolution')}°",
            f"Target CRS: {self.config.get('resampling.target_crs')}",
            "",
            "Dataset Details:",
            "-" * 20
        ]
        
        for i, info in enumerate(datasets_info, 1):
            report_lines.extend([
                f"{i}. {info['name']}",
                f"   Type: {info.get('data_type', 'unknown')}",
                f"   Band: {info.get('band_name', 'unknown')}",
                f"   Method: {info.get('resampling_method', 'unknown')}",
                f"   Shape: {info.get('shape', 'unknown')}",
                ""
            ])
        
        return "\n".join(report_lines)
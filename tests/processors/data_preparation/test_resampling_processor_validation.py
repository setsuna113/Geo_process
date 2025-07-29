"""Integration tests for ResamplingProcessor with validation framework."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
from dataclasses import dataclass

from src.processors.data_preparation.resampling_processor import (
    ResamplingProcessor, ResampledDatasetInfo
)
from src.abstractions.interfaces.validator import ValidationSeverity


@dataclass
class MockRasterEntry:
    """Mock raster entry for testing."""
    name: str
    path: Path
    bounds: tuple
    crs: str = 'EPSG:4326'
    resolution: float = 0.1
    width: int = 100
    height: int = 100
    count: int = 1
    dtype: str = 'float32'


class TestResamplingProcessorValidation:
    """Test ResamplingProcessor validation integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.db = Mock()
        
        # Configure mock config
        self.config.get.side_effect = self._mock_config_get
        
        # Create processor with mocked dependencies
        with patch('src.processors.data_preparation.resampling_processor.RasterCatalog'):
            with patch('src.processors.data_preparation.resampling_processor.ResamplingCacheManager'):
                with patch('src.processors.data_preparation.resampling_processor.get_memory_manager'):
                    self.processor = ResamplingProcessor(self.config, self.db)
    
    def _mock_config_get(self, key, default=None):
        """Mock config.get() method."""
        config_map = {
            'resampling': {
                'target_resolution': 0.05,
                'target_crs': 'EPSG:4326',
                'strategies': {'continuous': 'bilinear'},
                'engine': 'numpy'
            },
            'resampling.allow_skip_resampling': False,
            'raster_processing.max_chunk_size_mb': 512,
            'raster_processing.tile_size': 1024,
            'raster_processing.tile_overlap': 0,
            'datasets': {'target_datasets': []},
            'processors.memory_limit_mb': 1024
        }
        
        # Handle nested keys
        keys = key.split('.')
        value = config_map
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def test_valid_source_bounds_validation(self):
        """Test validation passes for valid source bounds."""
        # Create mock raster entry with valid bounds
        raster_entry = MockRasterEntry(
            name='test_raster',
            path=Path('/fake/path.tif'),
            bounds=(-180.0, -90.0, 180.0, 90.0),
            crs='EPSG:4326'
        )
        
        dataset_config = {
            'name': 'test_dataset',
            'path': '/fake/path.tif',
            'data_type': 'continuous',
            'band_name': 'test_band'
        }
        
        # Mock the catalog and path resolution
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            # Mock _get_or_register_raster to return our mock entry
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            
            # Mock the actual resampling methods
            self.processor._check_resolution_match = Mock(return_value=False)
            self.processor._estimate_memory_requirements = Mock(return_value=100.0)
            self.processor._resample_single = Mock(return_value=np.random.rand(100, 100))
            self.processor._calculate_output_bounds_from_data = Mock(
                return_value=(-180.0, -90.0, 180.0, 90.0)
            )
            
            # Process dataset
            result = self.processor.resample_dataset(dataset_config)
            
            # Check that validation was called
            assert hasattr(self.processor, 'validation_results')
            
            # Find bounds validation results
            bounds_validations = [
                v for v in self.processor.validation_results 
                if v['stage'] == 'source_bounds'
            ]
            
            assert len(bounds_validations) > 0
            assert all(v['result'].is_valid for v in bounds_validations)
    
    def test_invalid_source_bounds_detected(self):
        """Test detection of invalid source bounds."""
        # Create raster with invalid bounds (minx > maxx)
        raster_entry = MockRasterEntry(
            name='bad_bounds_raster',
            path=Path('/fake/bad.tif'),
            bounds=(180.0, -90.0, -180.0, 90.0),  # Invalid!
            crs='EPSG:4326'
        )
        
        dataset_config = {
            'name': 'bad_dataset',
            'path': '/fake/bad.tif',
            'data_type': 'continuous',
            'band_name': 'test_band'
        }
        
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            
            # Should raise ValueError due to invalid bounds
            with pytest.raises(ValueError) as exc_info:
                self.processor.resample_dataset(dataset_config)
            
            assert "Invalid source bounds" in str(exc_info.value)
            assert "bad_dataset" in str(exc_info.value)
    
    def test_coordinate_transformation_validation(self):
        """Test validation of coordinate transformations."""
        # Test transformation from UTM to WGS84
        raster_entry = MockRasterEntry(
            name='utm_raster',
            path=Path('/fake/utm.tif'),
            bounds=(500000, 4000000, 600000, 4100000),  # UTM coordinates
            crs='EPSG:32633'  # UTM Zone 33N
        )
        
        dataset_config = {
            'name': 'utm_dataset',
            'path': '/fake/utm.tif',
            'data_type': 'continuous',
            'band_name': 'test_band'
        }
        
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            self.processor._check_resolution_match = Mock(return_value=False)
            self.processor._estimate_memory_requirements = Mock(return_value=100.0)
            self.processor._resample_single = Mock(return_value=np.random.rand(100, 100))
            self.processor._calculate_output_bounds_from_data = Mock(
                return_value=(13.0, 36.0, 14.0, 37.0)  # Approximate lat/lon
            )
            
            # Process with transformation
            result = self.processor.resample_dataset(dataset_config)
            
            # Check transformation validation was performed
            transform_validations = [
                v for v in self.processor.validation_results
                if v['stage'] == 'coordinate_transform'
            ]
            
            assert len(transform_validations) > 0
            # Should have warning about datum shift (UTM uses WGS84 datum)
            assert any(v['result'].warning_count > 0 for v in transform_validations)
    
    def test_resampled_data_quality_validation(self):
        """Test validation of resampled data quality."""
        raster_entry = MockRasterEntry(
            name='quality_test',
            path=Path('/fake/quality.tif'),
            bounds=(0.0, 0.0, 10.0, 10.0),
            crs='EPSG:4326'
        )
        
        dataset_config = {
            'name': 'quality_dataset',
            'path': '/fake/quality.tif',
            'data_type': 'continuous',
            'band_name': 'test_band'
        }
        
        # Create data with some NaN values and outliers
        test_data = np.random.normal(0, 1, (100, 100))
        test_data[0:10, 0:10] = np.nan  # Add NaN region
        test_data[50, 50] = 100  # Add outlier
        
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            self.processor._check_resolution_match = Mock(return_value=False)
            self.processor._estimate_memory_requirements = Mock(return_value=100.0)
            self.processor._resample_single = Mock(return_value=test_data)
            self.processor._calculate_output_bounds_from_data = Mock(
                return_value=(0.0, 0.0, 10.0, 10.0)
            )
            
            # Process dataset
            result = self.processor.resample_dataset(dataset_config)
            
            # Check data quality validation
            data_validations = [
                v for v in self.processor.validation_results
                if v['stage'] == 'resampled_data'
            ]
            
            assert len(data_validations) > 0
            # Should detect outliers and generate warnings
            assert any(v['result'].warning_count > 0 for v in data_validations)
    
    def test_output_bounds_consistency_validation(self):
        """Test validation of output bounds consistency."""
        original_bounds = (0.0, 0.0, 100.0, 100.0)
        raster_entry = MockRasterEntry(
            name='bounds_test',
            path=Path('/fake/bounds.tif'),
            bounds=original_bounds,
            crs='EPSG:4326'
        )
        
        dataset_config = {
            'name': 'bounds_dataset',
            'path': '/fake/bounds.tif',
            'data_type': 'continuous',
            'band_name': 'test_band'
        }
        
        # Output data with slightly different calculated bounds
        output_data = np.random.rand(95, 95)  # Slightly smaller
        calculated_bounds = (0.0, 0.0, 95.0, 95.0)  # Mismatch!
        
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            self.processor._check_resolution_match = Mock(return_value=False)
            self.processor._estimate_memory_requirements = Mock(return_value=100.0)
            self.processor._resample_single = Mock(return_value=output_data)
            self.processor._calculate_output_bounds_from_data = Mock(
                return_value=calculated_bounds
            )
            
            # Process dataset
            result = self.processor.resample_dataset(dataset_config)
            
            # Check bounds consistency validation
            bounds_validations = [
                v for v in self.processor.validation_results
                if v['stage'] == 'output_bounds'
            ]
            
            assert len(bounds_validations) > 0
            # Should detect bounds mismatch
            assert any(v['result'].warning_count > 0 for v in bounds_validations)
    
    def test_validation_summary_reporting(self):
        """Test that validation summary is properly reported."""
        raster_entry = MockRasterEntry(
            name='summary_test',
            path=Path('/fake/summary.tif'),
            bounds=(0.0, 0.0, 10.0, 10.0),
            crs='EPSG:4326'
        )
        
        datasets_config = [{
            'name': 'summary_dataset',
            'path': '/fake/summary.tif',
            'data_type': 'continuous',
            'band_name': 'test_band',
            'enabled': True
        }]
        
        # Configure processor for batch processing
        self.processor.datasets_config = datasets_config
        
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            self.processor._check_resolution_match = Mock(return_value=False)
            self.processor._estimate_memory_requirements = Mock(return_value=100.0)
            self.processor._resample_single = Mock(return_value=np.random.rand(100, 100))
            self.processor._calculate_output_bounds_from_data = Mock(
                return_value=(0.0, 0.0, 10.0, 10.0)
            )
            
            # Mock progress tracking methods
            self.processor.start_progress = Mock()
            self.processor.complete_progress = Mock()
            self.processor.update_progress = Mock()
            self.processor._should_stop = Mock()
            self.processor._should_stop.is_set.return_value = False
            
            # Capture log output
            with patch('src.processors.data_preparation.resampling_processor.logger') as mock_logger:
                # Process all datasets
                results = self.processor.resample_all_datasets()
                
                # Check that validation summary was logged
                summary_logged = any(
                    "RESAMPLING VALIDATION SUMMARY" in str(call)
                    for call in mock_logger.info.call_args_list
                )
                assert summary_logged
                
                # Check results
                assert len(results) == 1
                assert results[0].name == 'summary_dataset'


class TestResamplingProcessorErrorScenarios:
    """Test error scenarios for resampling validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.db = Mock()
        
        # Basic config
        self.config.get.return_value = {}
        
        with patch('src.processors.data_preparation.resampling_processor.RasterCatalog'):
            with patch('src.processors.data_preparation.resampling_processor.ResamplingCacheManager'):
                with patch('src.processors.data_preparation.resampling_processor.get_memory_manager'):
                    self.processor = ResamplingProcessor(self.config, self.db)
    
    def test_invalid_crs_transformation(self):
        """Test handling of invalid CRS transformations."""
        raster_entry = MockRasterEntry(
            name='invalid_crs',
            path=Path('/fake/invalid.tif'),
            bounds=(0.0, 0.0, 10.0, 10.0),
            crs='INVALID:9999'  # Invalid CRS
        )
        
        dataset_config = {
            'name': 'invalid_crs_dataset',
            'path': '/fake/invalid.tif',
            'data_type': 'continuous',
            'band_name': 'test_band'
        }
        
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            
            # Should raise error due to invalid CRS
            with pytest.raises(ValueError) as exc_info:
                self.processor.resample_dataset(dataset_config)
            
            assert "Invalid coordinate transformation" in str(exc_info.value)
    
    def test_all_nan_data_handling(self):
        """Test handling of data that becomes all NaN after resampling."""
        raster_entry = MockRasterEntry(
            name='nan_test',
            path=Path('/fake/nan.tif'),
            bounds=(0.0, 0.0, 10.0, 10.0),
            crs='EPSG:4326'
        )
        
        dataset_config = {
            'name': 'nan_dataset',
            'path': '/fake/nan.tif',
            'data_type': 'continuous',
            'band_name': 'test_band'
        }
        
        # Create all-NaN data
        nan_data = np.full((100, 100), np.nan)
        
        with patch('src.processors.data_preparation.resampling_processor.DatasetPathResolver'):
            self.processor._get_or_register_raster = Mock(return_value=raster_entry)
            self.processor._check_resolution_match = Mock(return_value=False)
            self.processor._estimate_memory_requirements = Mock(return_value=100.0)
            self.processor._resample_single = Mock(return_value=nan_data)
            self.processor._calculate_output_bounds_from_data = Mock(
                return_value=(0.0, 0.0, 10.0, 10.0)
            )
            
            # Should complete but log warning
            result = self.processor.resample_dataset(dataset_config)
            
            # Check validation detected all-NaN data
            data_validations = [
                v for v in self.processor.validation_results
                if v['stage'] == 'resampled_data'
            ]
            
            # Should have handled gracefully
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
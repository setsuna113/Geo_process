# tests/test_integration/test_full_pipeline.py
"""End-to-end integration tests for the complete data processing pipeline."""

import pytest
from unittest import mock
import numpy as np
import tempfile
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.raster.loaders.geotiff_loader import GeoTIFFLoader
from src.processors.data_preparation.raster_cleaner import RasterCleaner
from src.processors.data_preparation.data_normalizer import DataNormalizer
from src.processors.data_preparation.array_converter import ArrayConverter
from src.spatial_analysis.som.som_trainer import SOMAnalyzer
from src.spatial_analysis.som.som_visualizer import SOMVisualizer
from src.spatial_analysis.som.som_reporter import SOMReporter


class TestFullPipeline:
    """Test the complete pipeline from raster input to spatial analysis output."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        config = Config()
        config.config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_biodiversity',
                'user': 'test_user',
                'password': 'test_pass'
            },
            'raster': {
                'catalog_path': 'test_catalog.json',
                'cache_dir': 'test_cache',
                'chunk_size': 512
            },
            'processors': {
                'data_preparation': {
                    'cleaning': {
                        'fill_method': 'interpolate',
                        'max_gap_size': 3
                    },
                    'normalization': {
                        'method': 'min_max',
                        'feature_range': [0, 1]
                    }
                }
            },
            'spatial_analysis': {
                'normalize_data': True,
                'save_results': True,
                'output_dir': 'test_output',
                'som': {
                    'grid_size': [3, 3],
                    'iterations': 50,
                    'sigma': 1.0,
                    'learning_rate': 0.5
                }
            },
            'grid_systems': {
                'default_grid': 'hexagonal',
                'resolution': 5000  # 5km
            }
        }
        return config
    
    @pytest.fixture
    def create_test_rasters(self):
        """Create test GeoTIFF files with known patterns."""
        def _create_rasters(output_dir, size=(10, 10), bands=['P', 'A', 'F']):
            files = {}
            
            # Create coordinate arrays
            transform = from_origin(-120, 50, 0.1, 0.1)  # 0.1 degree resolution
            
            for band in bands:
                filepath = output_dir / f'test_{band}.tif'
                
                # Create different patterns for each band
                if band == 'P':  # Productivity - gradient
                    data = np.linspace(0.1, 0.9, size[0]*size[1]).reshape(size)
                elif band == 'A':  # Activity - random with structure
                    np.random.seed(42)
                    data = 0.5 + 0.3 * np.random.randn(*size)
                    data = np.clip(data, 0, 1)
                elif band == 'F':  # Fragmentation - patchy
                    x, y = np.meshgrid(range(size[1]), range(size[0]))
                    data = np.sin(x/2) * np.cos(y/2) * 0.5 + 0.5
                else:
                    data = np.random.rand(*size)
                
                # Write GeoTIFF
                with rasterio.open(
                    filepath,
                    'w',
                    driver='GTiff',
                    height=size[0],
                    width=size[1],
                    count=1,
                    dtype='float32',
                    crs='EPSG:4326',
                    transform=transform,
                    compress='deflate'
                ) as dst:
                    dst.write(data.astype('float32'), 1)
                    dst.update_tags(band_name=band)
                
                files[band] = filepath
            
            return files
        
        return _create_rasters
    
    def test_normal_pipeline_flow(self, test_config, create_test_rasters):
        """Test the normal flow through the entire pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Step 1: Create test raster files
            raster_files = create_test_rasters(tmpdir)
            
            # Step 2: Initialize database (mock for testing)
            with mock.patch('psycopg2.connect'):
                mock_db = mock.Mock(spec=DatabaseManager)
                
                # Step 3: Load rasters using window-based approach
                loader = GeoTIFFLoader(test_config)
                loaded_data = {}
                
                for band, filepath in raster_files.items():
                    # Get metadata first
                    metadata = loader.extract_metadata(filepath)
                    
                    # Load entire raster using window
                    window_bounds = metadata.bounds
                    data = loader.load_window(filepath, window_bounds)
                    
                    loaded_data[band] = {
                        'data': data,
                        'metadata': {
                            'crs': metadata.crs,
                            'band_count': metadata.band_count,
                            'transform': (metadata.pixel_size[0], 0, metadata.bounds[0], 
                                        0, metadata.pixel_size[1], metadata.bounds[3])
                        }
                    }
                    
                    # Verify loading
                    assert data.shape == (10, 10)
                    assert metadata.crs is not None
                    assert metadata.band_count == 1
                
                # Step 4: Create combined dataset
                combined_data = xr.Dataset({
                    band: xr.DataArray(
                        loaded_data[band]['data'],
                        dims=['y', 'x'],
                        attrs=loaded_data[band]['metadata']
                    )
                    for band in ['P', 'A', 'F']
                })
                
                # Step 5: Clean data  
                cleaner = RasterCleaner(test_config, mock_db)
                # Clean each band's raster file first, then combine
                cleaned_datasets = {}
                for band, filepath in raster_files.items():
                    cleaned_result = cleaner.clean_raster(filepath, dataset_type='all')
                    cleaned_datasets[band] = cleaned_result['cleaned_data']
                
                # Create combined cleaned dataset
                cleaned_data = xr.Dataset({
                    band: cleaned_datasets[band] for band in ['P', 'A', 'F']
                })
                
                # Verify cleaning
                assert not np.any(np.isnan(cleaned_data['P'].values))
                assert cleaned_data.dims == combined_data.dims
                
                # Step 6: Normalize data
                normalizer = DataNormalizer(test_config, mock_db)
                normalized_result = normalizer.normalize(cleaned_data)
                normalized_data = normalized_result.get('normalized_data', cleaned_data)
                
                # Verify normalization
                for band in ['P', 'A', 'F']:
                    values = normalized_data[band].values
                    assert values.min() >= 0
                    assert values.max() <= 1
                
                # Step 7: Run SOM analysis
                som_analyzer = SOMAnalyzer(test_config)
                som_result = som_analyzer.analyze(
                    normalized_data,
                    grid_size=[3, 3],
                    iterations=20  # Reduced for testing
                )
                
                # Step 8: Verify SOM results structure
                assert som_result.metadata.analysis_type == 'SOM'
                assert som_result.labels.shape == (100,)  # 10x10 flattened
                assert 0 <= som_result.labels.max() < 9  # 3x3 grid
                
                # Verify statistics
                assert 'n_clusters' in som_result.statistics
                assert 'quantization_error' in som_result.statistics
                assert 'cluster_statistics' in som_result.statistics
                
                # Verify spatial output (with None safety)
                assert som_result.spatial_output is not None
                assert hasattr(som_result.spatial_output, 'values')
                assert som_result.spatial_output.shape == (10, 10)
                
                # Verify additional outputs (with None safety)
                assert som_result.additional_outputs is not None
                assert 'distance_map' in som_result.additional_outputs
                assert 'component_planes' in som_result.additional_outputs
                assert som_result.additional_outputs['distance_map'].shape == (3, 3)
                
                # Step 9: Generate visualizations
                visualizer = SOMVisualizer()
                cluster_fig = visualizer.plot_cluster_map(som_result)
                assert cluster_fig is not None
                plt.close(cluster_fig)
                
                # Step 10: Generate report
                reporter = SOMReporter()
                report = reporter.generate_full_report(som_result)
                assert 'quality_metrics' in report
                assert 'interpretation' in report
    
    def test_edge_case_single_pixel(self, test_config, create_test_rasters):
        """Test pipeline with single pixel input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create 1x1 rasters
            raster_files = create_test_rasters(tmpdir, size=(1, 1))
            
            # Load and combine
            loader = GeoTIFFLoader(test_config)
            data_dict = {}
            
            for band, filepath in raster_files.items():
                metadata = loader.extract_metadata(filepath)
                data = loader.load_window(filepath, metadata.bounds)
                data_dict[band] = xr.DataArray(data, dims=['y', 'x'])
            
            # Process through pipeline with mock database
            mock_db = mock.Mock(spec=DatabaseManager)
            
            # Clean each raster file
            cleaner = RasterCleaner(test_config, mock_db)
            cleaned_datasets = {}
            for band, filepath in raster_files.items():
                cleaned_result = cleaner.clean_raster(filepath, dataset_type='all')
                cleaned_datasets[band] = cleaned_result['cleaned_data']
            
            cleaned = xr.Dataset({band: cleaned_datasets[band] for band in ['P', 'A', 'F']})
            
            normalizer = DataNormalizer(test_config, mock_db)
            normalized_result = normalizer.normalize(cleaned)
            normalized = normalized_result.get('normalized_data', cleaned)
            
            # SOM should handle single pixel gracefully
            som_analyzer = SOMAnalyzer(test_config)
            with pytest.warns(UserWarning, match="small dataset"):
                result = som_analyzer.analyze(normalized, grid_size=[1, 1])
            
            assert result.labels.shape == (1,)
            assert result.labels[0] == 0  # Only one cluster possible
    
    def test_edge_case_all_nan_band(self, test_config):
        """Test pipeline with a band that's all NaN."""
        # Create dataset with one NaN band
        data = xr.Dataset({
            'P': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
            'A': xr.DataArray(np.full((5, 5), np.nan), dims=['y', 'x']),
            'F': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x'])
        })
        
        # Mock database for processors
        mock_db = mock.Mock(spec=DatabaseManager)
        
        # Normalizer should handle NaN band (we'll use normalize directly on xarray)
        normalizer = DataNormalizer(test_config, mock_db)
        normalized_result = normalizer.normalize(data)
        normalized = normalized_result.get('normalized_data', data)
        
        # Check if NaNs were handled (filled)
        assert not np.all(np.isnan(normalized['A'].values))
        
        # SOM should work with normalized data
        som_analyzer = SOMAnalyzer(test_config)
        result = som_analyzer.analyze(normalized, grid_size=[2, 2])
        
        assert result.metadata.analysis_type == 'SOM'
        assert len(np.unique(result.labels)) <= 4
    
    def test_edge_case_constant_values(self, test_config):
        """Test pipeline with constant value bands."""
        # Create dataset with constant values
        data = xr.Dataset({
            'P': xr.DataArray(np.full((5, 5), 0.5), dims=['y', 'x']),
            'A': xr.DataArray(np.full((5, 5), 0.5), dims=['y', 'x']),
            'F': xr.DataArray(np.full((5, 5), 0.5), dims=['y', 'x'])
        })
        
        # Mock database for processors
        mock_db = mock.Mock(spec=DatabaseManager)
        
        # Normalizer should handle constant values
        normalizer = DataNormalizer(test_config, mock_db)
        normalized_result = normalizer.normalize(data)
        normalized = normalized_result.get('normalized_data', data)
        
        # All values should remain constant (or become 0 after normalization)
        for band in ['P', 'A', 'F']:
            unique_vals = np.unique(normalized[band].values)
            assert len(unique_vals) == 1
        
        # SOM should still work but find only one pattern
        som_analyzer = SOMAnalyzer(test_config)
        result = som_analyzer.analyze(normalized, grid_size=[3, 3])
        
        # With identical inputs, might use only one cluster
        assert len(np.unique(result.labels)) >= 1
    
    def test_strange_input_mismatched_dimensions(self, test_config):
        """Test pipeline with mismatched band dimensions."""
        # Create dataset with different sized bands - this should fail at xr.Dataset creation
        with pytest.raises((ValueError, Exception)):  # More general exception catching
            # This should fail when creating the dataset due to mismatched coordinates
            data = xr.Dataset({
                'P': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
                'A': xr.DataArray(np.random.rand(6, 6), dims=['y', 'x']),  # Different size
                'F': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x'])
            })
            
            # If somehow the dataset creation succeeds, the processing should fail
            mock_db = mock.Mock(spec=DatabaseManager)
            normalizer = DataNormalizer(test_config, mock_db)
            normalizer.normalize(data)
    
    def test_strange_input_missing_band(self, test_config):
        """Test pipeline with missing expected band."""
        # Create dataset missing one band
        data = xr.Dataset({
            'P': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
            'A': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x'])
            # Missing 'F' band
        })
        
        # Pipeline should still work with available bands
        mock_db = mock.Mock(spec=DatabaseManager)
        
        normalizer = DataNormalizer(test_config, mock_db)
        normalized_result = normalizer.normalize(data)
        normalized = normalized_result.get('normalized_data', data)
        
        som_analyzer = SOMAnalyzer(test_config)
        result = som_analyzer.analyze(normalized, grid_size=[2, 2])
        
        # Should work with 2 bands instead of 3
        assert result.metadata.input_bands == ['P', 'A']
        assert result.statistics['n_clusters'] <= 4
    
    def test_output_structure_validation(self, test_config):
        """Thoroughly test the output structure of the analysis."""
        # Create simple test data
        data = xr.Dataset({
            'P': xr.DataArray(np.random.rand(8, 8), dims=['y', 'x']),
            'A': xr.DataArray(np.random.rand(8, 8), dims=['y', 'x']),
            'F': xr.DataArray(np.random.rand(8, 8), dims=['y', 'x'])
        })
        
        # Run through pipeline
        som_analyzer = SOMAnalyzer(test_config)
        result = som_analyzer.analyze(data, grid_size=[2, 2])
        
        # Validate AnalysisResult structure
        assert hasattr(result, 'labels')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'statistics')
        assert hasattr(result, 'spatial_output')
        assert hasattr(result, 'additional_outputs')
        
        # Validate metadata
        metadata = result.metadata
        assert metadata.analysis_type == 'SOM'
        assert metadata.input_shape == (3, 8, 8)
        assert metadata.input_bands == ['P', 'A', 'F']
        assert isinstance(metadata.parameters, dict)
        assert metadata.parameters['grid_size'] == [2, 2]
        assert isinstance(metadata.processing_time, float)
        assert isinstance(metadata.timestamp, str)
        
        # Validate statistics
        stats = result.statistics
        assert isinstance(stats['n_clusters'], int)
        assert 0 < stats['n_clusters'] <= 4
        assert 0 <= stats['quantization_error'] <= 1
        assert 0 <= stats['topographic_error'] <= 1
        assert isinstance(stats['cluster_statistics'], dict)
        
        # Validate each cluster statistic
        for _, cluster_stats in stats['cluster_statistics'].items():
            assert 'count' in cluster_stats
            assert 'percentage' in cluster_stats
            assert 'mean' in cluster_stats
            assert 'std' in cluster_stats
            assert len(cluster_stats['mean']) == 3  # 3 bands
        
        # Validate spatial output
        assert isinstance(result.spatial_output, xr.DataArray)
        assert result.spatial_output.shape == (8, 8)
        assert result.spatial_output.dims == ('lat', 'lon')
        
        # Validate additional outputs (with None safety)
        assert result.additional_outputs is not None
        assert 'som_weights' in result.additional_outputs
        assert 'distance_map' in result.additional_outputs
        assert 'activation_map' in result.additional_outputs
        assert 'component_planes' in result.additional_outputs
        
        # Check component planes structure
        planes = result.additional_outputs['component_planes']
        assert set(planes.keys()) == {'P', 'A', 'F'}
        for _, plane in planes.items():
            assert plane.shape == (2, 2)  # Grid size
    
    def test_database_integration(self, test_config):
        """Test database storage and retrieval of results."""
        # Create test data
        data = xr.Dataset({
            'P': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
            'A': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
            'F': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x'])
        })
        
        # Run analysis
        som_analyzer = SOMAnalyzer(test_config)
        result = som_analyzer.analyze(data)
        
        # Mock database storage
        with mock.patch('psycopg2.connect') as mock_connect:
            mock_conn = mock.Mock()
            mock_cursor = mock.Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Test storing in database
            som_analyzer.store_in_database(result)
            
            # Verify INSERT was called
            assert mock_cursor.execute.called
            insert_calls = [call for call in mock_cursor.execute.call_args_list 
                          if 'INSERT' in str(call)]
            assert len(insert_calls) > 0
    
    def test_coordinate_preservation(self, test_config, create_test_rasters):
        """Test that geographic coordinates are preserved through pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create georeferenced rasters
            raster_files = create_test_rasters(tmpdir, size=(6, 8))
            
            # Load with coordinates
            loader = GeoTIFFLoader(test_config)
            data_dict = {}
            coords_info = None
            
            for band, filepath in raster_files.items():
                metadata = loader.extract_metadata(filepath)
                data = loader.load_window(filepath, metadata.bounds)
                
                if coords_info is None:
                    # Extract coordinate info from metadata
                    height, width = data.shape
                    
                    # Calculate lat/lon arrays from bounds
                    west, south, east, north = metadata.bounds
                    lons = np.linspace(west, east, width)
                    lats = np.linspace(north, south, height)
                    coords_info = {'lat': lats, 'lon': lons}
                
                data_dict[band] = xr.DataArray(
                    data,
                    dims=['lat', 'lon'],
                    coords=coords_info
                )
            
            combined_data = xr.Dataset(data_dict)
            
            # Process through pipeline
            som_analyzer = SOMAnalyzer(test_config)
            result = som_analyzer.analyze(combined_data)
            
            # Check coordinates are preserved in spatial output (with None safety)
            assert result.spatial_output is not None
            assert 'lat' in result.spatial_output.coords
            assert 'lon' in result.spatial_output.coords
            assert len(result.spatial_output.coords['lat']) == 6
            assert len(result.spatial_output.coords['lon']) == 8
    
    def test_array_converter_integration(self, test_config):
        """Test array converter handles different input formats."""
        converter = ArrayConverter(test_config)
        
        # Test various input formats
        test_cases = [
            # NumPy array
            np.random.rand(5, 5, 3),
            # xarray DataArray
            xr.DataArray(np.random.rand(5, 5, 3), dims=['y', 'x', 'band']),
            # xarray Dataset
            xr.Dataset({
                'P': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
                'A': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
                'F': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x'])
            }),
            # Pandas DataFrame (spatial data)
            pd.DataFrame({
                'x': np.repeat(range(5), 5),
                'y': np.tile(range(5), 5),
                'P': np.random.rand(25),
                'A': np.random.rand(25),
                'F': np.random.rand(25)
            })
        ]
        
        for test_input in test_cases:
            # Each should be convertible using xarray_to_numpy
            result = converter.xarray_to_numpy(test_input)
            
            assert isinstance(result['array'], np.ndarray)
            assert result['array'].ndim >= 2


# Import matplotlib for closing figures
import matplotlib.pyplot as plt
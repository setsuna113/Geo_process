"""Integration tests for the upgraded merge pipeline."""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.stages.merge_stage import MergeStage
from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo
from src.processors.data_preparation.raster_alignment import RasterAligner
from src.database.connection import DatabaseManager
from src.config import config


class TestMergePipelineIntegration(unittest.TestCase):
    """Test full merge pipeline with mixed passthrough/resampled data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.output_dir = Path(cls.temp_dir) / "output"
        cls.output_dir.mkdir(exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
        
    def setUp(self):
        """Set up test fixtures."""
        self.db = DatabaseManager()
        
        # Create mock pipeline context
        self.context = Mock()
        self.context.config = config
        self.context.db = self.db
        self.context.output_dir = self.output_dir
        self.context.get = Mock()
        self.context.set = Mock()
        
        # Mock logging context to handle @log_stage decorator
        mock_logging_context = Mock()
        mock_stage_context = Mock()
        mock_stage_context.__enter__ = Mock(return_value=None)
        mock_stage_context.__exit__ = Mock(return_value=None)
        mock_logging_context.stage = Mock(return_value=mock_stage_context)
        self.context.logging_context = mock_logging_context
        
    def test_mixed_passthrough_resampled_merge(self):
        """Test merging mixed passthrough and resampled datasets."""
        # Create mock resampled dataset info
        datasets = [
            ResampledDatasetInfo(
                name="passthrough-dataset",
                source_path=Path("/fake/path1.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={
                    'passthrough': True,
                    'memory_aware': False
                }
            ),
            ResampledDatasetInfo(
                name="resampled-dataset",
                source_path=Path("/fake/path2.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="bilinear",
                band_name="band1",
                metadata={
                    'passthrough': False,
                    'memory_aware': True,
                    'storage_table': 'windowed_resampled_dataset'
                }
            )
        ]
        
        self.context.get.return_value = datasets
        
        # Create and execute merge stage
        merge_stage = MergeStage()
        
        # Mock the CoordinateMerger to avoid database operations
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            # Create a real file for the test
            test_parquet = self.output_dir / "ml_ready.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = []
            
            # Execute stage
            result = merge_stage.execute(self.context)
            
            # Verify result
            self.assertTrue(result.success)
            self.assertEqual(result.metrics['datasets_merged'], 2)
            
            # Verify correct table names were generated
            call_args = mock_merger.create_ml_ready_parquet.call_args
            dataset_dicts = call_args[0][0]
            
            # Check passthrough dataset
            passthrough_dict = next(d for d in dataset_dicts if d['name'] == 'passthrough-dataset')
            self.assertEqual(passthrough_dict['table_name'], 'passthrough_passthrough_dataset')
            self.assertTrue(passthrough_dict['passthrough'])
            self.assertFalse(passthrough_dict['memory_aware'])
            
            # Check resampled dataset
            resampled_dict = next(d for d in dataset_dicts if d['name'] == 'resampled-dataset')
            self.assertEqual(resampled_dict['table_name'], 'windowed_resampled_dataset')
            self.assertFalse(resampled_dict['passthrough'])
            self.assertTrue(resampled_dict['memory_aware'])
            
    def test_alignment_detection_integration(self):
        """Test that alignment issues are detected and reported."""
        # Create datasets with alignment issues
        datasets = [
            ResampledDatasetInfo(
                name="aligned-dataset",
                source_path=Path("/fake/aligned.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True}
            ),
            ResampledDatasetInfo(
                name="misaligned-dataset",
                source_path=Path("/fake/misaligned.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-179.5, -89.5, 180.5, 90.5),  # Half pixel shift
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True}
            )
        ]
        
        self.context.get.return_value = datasets
        
        # Create merge stage
        merge_stage = MergeStage()
        
        # Mock CoordinateMerger
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            # Create a real file for the test
            test_parquet = self.output_dir / "ml_ready.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = []
            
            # Execute stage
            result = merge_stage.execute(self.context)
            
            # Check that alignment metrics were recorded
            self.assertIn('datasets_requiring_alignment', result.metrics)
            self.assertIn('max_alignment_shift_degrees', result.metrics)
            
            # The stage should detect 1 dataset needing alignment
            self.assertEqual(result.metrics['datasets_requiring_alignment'], 1)
            self.assertAlmostEqual(result.metrics['max_alignment_shift_degrees'], 0.5)
            
    def test_chunked_processing_config(self):
        """Test that chunked processing is configured correctly."""
        # Set config for chunked processing
        with patch.object(config, 'get') as mock_get:
            mock_get.side_effect = lambda k, d=None: {
                'merge.enable_chunked_processing': True,
                'merge.chunk_size': 1000
            }.get(k, d)
            
            datasets = [
                ResampledDatasetInfo(
                    name="large-dataset",
                    source_path=Path("/fake/large.tif"),
                    target_resolution=0.1,  # High resolution
                    target_crs="EPSG:4326",
                    bounds=(-180, -90, 180, 90),
                    shape=(3600, 1800),  # Large dataset
                    data_type="float32",
                    resampling_method="average",
                    band_name="band1",
                    metadata={'memory_aware': True, 'storage_table': 'windowed_large'}
                ),
                ResampledDatasetInfo(
                    name="large-dataset2",
                    source_path=Path("/fake/large2.tif"),
                    target_resolution=0.1,  # High resolution
                    target_crs="EPSG:4326",
                    bounds=(-180, -90, 180, 90),
                    shape=(3600, 1800),  # Large dataset
                    data_type="float32",
                    resampling_method="average",
                    band_name="band1",
                    metadata={'memory_aware': True, 'storage_table': 'windowed_large2'}
                )
            ]
            
            self.context.get.return_value = datasets
            
            merge_stage = MergeStage()
            
            with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
                # Create a real file for the test
                test_parquet = self.output_dir / "ml_ready.parquet"
                test_parquet.write_text("dummy parquet content")
                
                mock_merger = MockMerger.return_value
                mock_merger.create_ml_ready_parquet.return_value = test_parquet
                mock_merger.get_validation_results.return_value = []
                
                result = merge_stage.execute(self.context)
                
                # Verify chunked processing was requested
                call_args = mock_merger.create_ml_ready_parquet.call_args
                self.assertEqual(call_args[1]['chunk_size'], 1000)
                self.assertTrue(result.metrics['chunked_processing'])
                
    def test_validation_error_handling(self):
        """Test handling of validation errors during merge."""
        datasets = [
            ResampledDatasetInfo(
                name="dataset1",
                source_path=Path("/fake/data1.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True}
            ),
            ResampledDatasetInfo(
                name="dataset2",
                source_path=Path("/fake/data2.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True}
            )
        ]
        
        self.context.get.return_value = datasets
        
        merge_stage = MergeStage()
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            mock_merger = MockMerger.return_value
            # Simulate validation error
            mock_merger.create_ml_ready_parquet.side_effect = ValueError("Validation error: Invalid bounds")
            mock_merger.get_validation_results.return_value = [
                {
                    'stage': 'dataset_bounds',
                    'dataset': 'dataset1',
                    'result': Mock(is_valid=False, error_count=1, warning_count=0)
                }
            ]
            
            result = merge_stage.execute(self.context)
            
            # Should handle validation error gracefully
            self.assertFalse(result.success)
            self.assertIn("Validation error", result.data.get('error', ''))
            self.assertIn('validation_results', result.data)
            
    def test_monitoring_integration(self):
        """Test that monitoring hooks are properly integrated."""
        # This test verifies that the @log_stage decorator works
        datasets = [
            ResampledDatasetInfo(
                name="monitored-dataset",
                source_path=Path("/fake/monitored.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True}
            ),
            ResampledDatasetInfo(
                name="monitored-dataset2",
                source_path=Path("/fake/monitored2.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True}
            )
        ]
        
        self.context.get.return_value = datasets
        
        merge_stage = MergeStage()
        
        # Check that the execute method has the log_stage decorator
        self.assertTrue(hasattr(merge_stage.execute, '__wrapped__'))
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            # Create a real file for the test
            test_parquet = self.output_dir / "ml_ready.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = []
            
            # The decorator should log the stage execution
            with patch('src.infrastructure.logging.get_logger') as mock_logger:
                result = merge_stage.execute(self.context)
                
                # Verify success
                self.assertTrue(result.success)


if __name__ == '__main__':
    unittest.main()
"""Integration tests focused on merge stage logic without full processor initialization."""

import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.base_stage import StageResult
from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo


class TestMergeStageLogic(unittest.TestCase):
    """Test merge stage logic and data flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = Path(tempfile.mkdtemp())
        
        # Create mock context with all required attributes
        self.context = Mock()
        self.context.output_dir = self.output_dir
        self.context.get = Mock()
        self.context.set = Mock()
        
        # Mock config
        self.context.config = Mock()
        self.context.config.get = Mock(return_value=None)
        
        # Mock database
        self.context.db = Mock()
        
        # Mock logging context
        mock_logging_context = Mock()
        mock_stage_context = MagicMock()
        mock_logging_context.stage = Mock(return_value=mock_stage_context)
        self.context.logging_context = mock_logging_context
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.output_dir)
        
    def test_table_name_generation_logic(self):
        """Test that correct table names are generated for different storage types."""
        datasets = [
            # Legacy passthrough
            ResampledDatasetInfo(
                name="legacy-passthrough",
                source_path=Path("/fake/legacy.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True, 'memory_aware': False}
            ),
            # Legacy resampled
            ResampledDatasetInfo(
                name="legacy-resampled",
                source_path=Path("/fake/resampled.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="bilinear",
                band_name="band1",
                metadata={'passthrough': False, 'memory_aware': False}
            ),
            # New windowed storage
            ResampledDatasetInfo(
                name="windowed-data",
                source_path=Path("/fake/windowed.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'memory_aware': True, 'storage_table': 'custom_windowed_table'}
            )
        ]
        
        self.context.get.return_value = datasets
        
        # Patch CoordinateMerger to capture the dataset_dicts
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            mock_merger_instance = MockMerger.return_value
            captured_dicts = []
            
            def capture_call(*args, **kwargs):
                captured_dicts.extend(args[0])
                return self.output_dir / "test.parquet"
                
            mock_merger_instance.create_ml_ready_parquet.side_effect = capture_call
            mock_merger_instance.get_validation_results.return_value = []
            
            merge_stage = MergeStage()
            result = merge_stage.execute(self.context)
            
            # Check captured dataset dictionaries
            self.assertEqual(len(captured_dicts), 3)
            
            # Check legacy passthrough
            legacy_pt = next(d for d in captured_dicts if d['name'] == 'legacy-passthrough')
            self.assertEqual(legacy_pt['table_name'], 'passthrough_legacy_passthrough')
            self.assertTrue(legacy_pt['passthrough'])
            self.assertFalse(legacy_pt['memory_aware'])
            
            # Check legacy resampled
            legacy_rs = next(d for d in captured_dicts if d['name'] == 'legacy-resampled')
            self.assertEqual(legacy_rs['table_name'], 'resampled_legacy_resampled')
            self.assertFalse(legacy_rs['passthrough'])
            self.assertFalse(legacy_rs['memory_aware'])
            
            # Check windowed storage
            windowed = next(d for d in captured_dicts if d['name'] == 'windowed-data')
            self.assertEqual(windowed['table_name'], 'custom_windowed_table')
            self.assertFalse(windowed['passthrough'])
            self.assertTrue(windowed['memory_aware'])
            
    def test_alignment_metrics_calculation(self):
        """Test alignment detection and metrics calculation."""
        # Create datasets with known alignment issues
        datasets = [
            ResampledDatasetInfo(
                name="reference",
                source_path=Path("/fake/ref.tif"),
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
                name="shifted",
                source_path=Path("/fake/shifted.tif"),
                target_resolution=1.0,
                target_crs="EPSG:4326",
                bounds=(-179.75, -89.75, 180.25, 90.25),  # 0.25 degree shift
                shape=(360, 180),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'passthrough': True}
            )
        ]
        
        self.context.get.return_value = datasets
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            mock_merger_instance = MockMerger.return_value
            
            # Create a real file for the test
            test_parquet = self.output_dir / "test.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger_instance.create_ml_ready_parquet.return_value = test_parquet
            mock_merger_instance.get_validation_results.return_value = []
            
            merge_stage = MergeStage()
            result = merge_stage.execute(self.context)
            
            # Check alignment metrics
            self.assertTrue(result.success)
            self.assertEqual(result.metrics['datasets_requiring_alignment'], 1)
            self.assertAlmostEqual(result.metrics['max_alignment_shift_degrees'], 0.25, places=2)
            
    def test_chunked_processing_configuration(self):
        """Test chunked processing is configured based on config."""
        datasets = [
            ResampledDatasetInfo(
                name="dataset1",
                source_path=Path("/fake/data1.tif"),
                target_resolution=0.1,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(3600, 1800),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'memory_aware': True, 'storage_table': 'windowed_1'}
            ),
            ResampledDatasetInfo(
                name="dataset2",
                source_path=Path("/fake/data2.tif"),
                target_resolution=0.1,
                target_crs="EPSG:4326",
                bounds=(-180, -90, 180, 90),
                shape=(3600, 1800),
                data_type="float32",
                resampling_method="average",
                band_name="band1",
                metadata={'memory_aware': True, 'storage_table': 'windowed_2'}
            )
        ]
        
        self.context.get.return_value = datasets
        
        # Configure chunked processing
        self.context.config.get.side_effect = lambda k, d=None: {
            'merge.enable_chunked_processing': True,
            'merge.chunk_size': 2000
        }.get(k, d)
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            mock_merger_instance = MockMerger.return_value
            captured_chunk_size = None
            
            def capture_chunk_size(*args, **kwargs):
                nonlocal captured_chunk_size
                captured_chunk_size = kwargs.get('chunk_size')
                # Create a real file for the test
                test_parquet = self.output_dir / "test.parquet"
                test_parquet.write_text("dummy parquet content")
                return test_parquet
                
            mock_merger_instance.create_ml_ready_parquet.side_effect = capture_chunk_size
            mock_merger_instance.get_validation_results.return_value = []
            
            merge_stage = MergeStage()
            result = merge_stage.execute(self.context)
            
            # Verify chunk size was passed
            self.assertEqual(captured_chunk_size, 2000)
            self.assertTrue(result.metrics['chunked_processing'])
            
    def test_insufficient_datasets_handling(self):
        """Test handling when fewer than 2 datasets are provided."""
        # Only one dataset
        datasets = [
            ResampledDatasetInfo(
                name="single",
                source_path=Path("/fake/single.tif"),
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
        result = merge_stage.execute(self.context)
        
        # Should fail with appropriate warning
        self.assertFalse(result.success)
        self.assertIn('Need at least 2 datasets', result.warnings[0])
        
    def test_validation_metrics_aggregation(self):
        """Test aggregation of validation metrics."""
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
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            mock_merger_instance = MockMerger.return_value
            
            # Create a real file for the test
            test_parquet = self.output_dir / "test.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger_instance.create_ml_ready_parquet.return_value = test_parquet
            
            # Mock validation results
            mock_validation_results = [
                {
                    'stage': 'bounds_check',
                    'dataset': 'dataset1',
                    'result': Mock(is_valid=True, error_count=0, warning_count=1)
                },
                {
                    'stage': 'coordinate_check',
                    'dataset': 'dataset2',
                    'result': Mock(is_valid=False, error_count=2, warning_count=3)
                }
            ]
            mock_merger_instance.get_validation_results.return_value = mock_validation_results
            
            merge_stage = MergeStage()
            result = merge_stage.execute(self.context)
            
            # Check validation metrics
            self.assertTrue(result.success)
            self.assertEqual(result.metrics['validation_checks'], 2)
            self.assertEqual(result.metrics['validation_errors'], 2)
            self.assertEqual(result.metrics['validation_warnings'], 4)
            self.assertEqual(result.metrics['validation_failures'], 1)
            
            # Check warnings generated
            self.assertEqual(len(result.warnings), 2)
            self.assertIn('4 validation warnings', result.warnings[0])
            self.assertIn('1 validation checks failed', result.warnings[1])


if __name__ == '__main__':
    unittest.main()
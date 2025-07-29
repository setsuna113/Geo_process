"""Performance tests for the upgraded merge pipeline."""

import unittest
import time
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import psutil

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.stages.merge_stage import MergeStage
from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo
from src.processors.data_preparation.coordinate_merger import CoordinateMerger
from src.database.connection import DatabaseManager
from src.config import config


class TestMergePerformance(unittest.TestCase):
    """Performance tests for merge operations."""
    
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
        self.context = Mock()
        self.context.output_dir = self.output_dir
        self.context.get = Mock()
        self.context.set = Mock()
        self.context.config = config
        self.context.db = Mock()
        
        # Mock logging context
        mock_logging_context = Mock()
        mock_stage_context = MagicMock()
        mock_stage_context.__enter__ = Mock(return_value=None)
        mock_stage_context.__exit__ = Mock(return_value=None)
        mock_logging_context.stage = Mock(return_value=mock_stage_context)
        self.context.logging_context = mock_logging_context
        
    def create_large_dataset_info(self, name, num_pixels):
        """Create dataset info representing a large dataset."""
        # Calculate shape to achieve desired number of pixels
        side_length = int(np.sqrt(num_pixels))
        resolution = 360.0 / side_length  # degrees per pixel
        
        return ResampledDatasetInfo(
            name=name,
            source_path=Path(f"/fake/{name}.tif"),
            target_resolution=resolution,
            target_crs="EPSG:4326",
            bounds=(-180, -90, 180, 90),
            shape=(side_length, side_length),
            data_type="float32",
            resampling_method="average",
            band_name="band1",
            metadata={'memory_aware': True, 'storage_table': f'windowed_{name}'}
        )
        
    def test_merge_stage_execution_time(self):
        """Test execution time for different dataset sizes."""
        sizes = [10000, 100000, 1000000]  # 10K, 100K, 1M pixels
        execution_times = []
        
        for size in sizes:
            # Create two datasets of the given size
            datasets = [
                self.create_large_dataset_info(f"dataset1_{size}", size),
                self.create_large_dataset_info(f"dataset2_{size}", size)
            ]
            
            self.context.get.return_value = datasets
            
            merge_stage = MergeStage()
            
            with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
                # Create a real file for the test
                test_parquet = self.output_dir / f"test_{size}.parquet"
                test_parquet.write_text("dummy parquet content")
                
                mock_merger = MockMerger.return_value
                mock_merger.create_ml_ready_parquet.return_value = test_parquet
                mock_merger.get_validation_results.return_value = []
                
                # Measure execution time
                start_time = time.time()
                result = merge_stage.execute(self.context)
                end_time = time.time()
                
                execution_time = end_time - start_time
                execution_times.append((size, execution_time))
                
                self.assertTrue(result.success)
                self.assertLess(execution_time, 5.0)  # Should complete within 5 seconds
                
        # Verify execution times scale reasonably
        print("\nMerge Stage Execution Times:")
        for size, exec_time in execution_times:
            print(f"  {size:,} pixels: {exec_time:.3f}s")
            
    def test_memory_usage_tracking(self):
        """Test that memory usage is tracked correctly."""
        # Create large datasets
        datasets = [
            self.create_large_dataset_info("mem_test1", 1000000),  # 1M pixels
            self.create_large_dataset_info("mem_test2", 1000000)
        ]
        
        self.context.get.return_value = datasets
        
        merge_stage = MergeStage()
        
        # Track memory before execution
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            test_parquet = self.output_dir / "mem_test.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = []
            
            result = merge_stage.execute(self.context)
            
            # Check memory tracking in result
            self.assertTrue(result.success)
            self.assertIn('memory_peak_mb', result.__dict__)
            self.assertGreaterEqual(result.memory_peak_mb, 0)
            
    def test_chunked_vs_inmemory_performance(self):
        """Compare performance of chunked vs in-memory processing."""
        # Create datasets
        datasets = [
            self.create_large_dataset_info("chunk_test1", 500000),
            self.create_large_dataset_info("chunk_test2", 500000)
        ]
        
        self.context.get.return_value = datasets
        
        # Test in-memory processing
        self.context.config = Mock()
        self.context.config.get = Mock(side_effect=lambda k, d=None: {
            'merge.enable_chunked_processing': False
        }.get(k, d))
        
        merge_stage_inmemory = MergeStage()
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            test_parquet = self.output_dir / "inmemory.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = []
            
            start_time = time.time()
            result_inmemory = merge_stage_inmemory.execute(self.context)
            inmemory_time = time.time() - start_time
            
        # Test chunked processing
        self.context.config.get = Mock(side_effect=lambda k, d=None: {
            'merge.enable_chunked_processing': True,
            'merge.chunk_size': 1000
        }.get(k, d))
        
        merge_stage_chunked = MergeStage()
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            test_parquet = self.output_dir / "chunked.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = []
            
            start_time = time.time()
            result_chunked = merge_stage_chunked.execute(self.context)
            chunked_time = time.time() - start_time
            
        # Both should succeed
        self.assertTrue(result_inmemory.success)
        self.assertTrue(result_chunked.success)
        
        # Check chunked processing was enabled
        self.assertFalse(result_inmemory.metrics.get('chunked_processing', False))
        self.assertTrue(result_chunked.metrics.get('chunked_processing', False))
        
        print(f"\nProcessing Time Comparison:")
        print(f"  In-memory: {inmemory_time:.3f}s")
        print(f"  Chunked: {chunked_time:.3f}s")
        
    def test_alignment_detection_performance(self):
        """Test performance of alignment detection with many datasets."""
        # Create many datasets with various alignments
        num_datasets = 20
        datasets = []
        
        for i in range(num_datasets):
            # Half aligned, half with slight shifts
            if i % 2 == 0:
                bounds = (-180, -90, 180, 90)
            else:
                shift = i * 0.1  # Increasing shifts
                bounds = (-180 + shift, -90 + shift, 180 + shift, 90 + shift)
                
            datasets.append(
                ResampledDatasetInfo(
                    name=f"align_test_{i}",
                    source_path=Path(f"/fake/align_{i}.tif"),
                    target_resolution=1.0,
                    target_crs="EPSG:4326",
                    bounds=bounds,
                    shape=(360, 180),
                    data_type="float32",
                    resampling_method="average",
                    band_name="band1",
                    metadata={'passthrough': True}
                )
            )
            
        self.context.get.return_value = datasets
        
        merge_stage = MergeStage()
        
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            test_parquet = self.output_dir / "align_test.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = []
            
            start_time = time.time()
            result = merge_stage.execute(self.context)
            alignment_time = time.time() - start_time
            
        self.assertTrue(result.success)
        self.assertEqual(result.metrics['datasets_merged'], num_datasets)
        self.assertEqual(result.metrics['datasets_requiring_alignment'], num_datasets // 2)
        
        # Alignment detection should be fast even with many datasets
        self.assertLess(alignment_time, 2.0)
        
        print(f"\nAlignment Detection Performance:")
        print(f"  {num_datasets} datasets processed in {alignment_time:.3f}s")
        print(f"  {result.metrics['datasets_requiring_alignment']} datasets required alignment")
        
    def test_validation_performance(self):
        """Test performance of validation with many validation checks."""
        datasets = [
            self.create_large_dataset_info("val_test1", 100000),
            self.create_large_dataset_info("val_test2", 100000)
        ]
        
        self.context.get.return_value = datasets
        
        merge_stage = MergeStage()
        
        # Create many validation results
        validation_results = []
        for i in range(50):  # 50 validation checks
            validation_results.append({
                'stage': f'check_{i}',
                'dataset': f'dataset{i % 2 + 1}',
                'result': Mock(
                    is_valid=i % 3 != 0,  # Every 3rd check fails
                    error_count=1 if i % 3 == 0 else 0,
                    warning_count=i % 5  # Varying warnings
                )
            })
            
        with patch('src.pipelines.stages.merge_stage.CoordinateMerger') as MockMerger:
            test_parquet = self.output_dir / "val_test.parquet"
            test_parquet.write_text("dummy parquet content")
            
            mock_merger = MockMerger.return_value
            mock_merger.create_ml_ready_parquet.return_value = test_parquet
            mock_merger.get_validation_results.return_value = validation_results
            
            start_time = time.time()
            result = merge_stage.execute(self.context)
            validation_time = time.time() - start_time
            
        self.assertTrue(result.success)
        self.assertEqual(result.metrics['validation_checks'], 50)
        
        # Validation aggregation should be fast
        self.assertLess(validation_time, 1.0)
        
        print(f"\nValidation Performance:")
        print(f"  {len(validation_results)} validation checks in {validation_time:.3f}s")
        print(f"  {result.metrics['validation_errors']} errors, {result.metrics['validation_warnings']} warnings")


if __name__ == '__main__':
    unittest.main()
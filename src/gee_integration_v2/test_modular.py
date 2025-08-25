"""
模块化架构测试
=============

测试新的模块化GEE集成架构。

作者: Linus式重构
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gee_integration_v2 import (
    GEEPipeline, 
    GEEConfig,
    TIFValidator,
    GEEOperations,
    DataProcessor,
    ExportManager,
    PipelineState
)


class TestModularArchitecture(unittest.TestCase):
    """测试模块化架构"""
    
    def test_module_imports(self):
        """测试所有模块可以正确导入"""
        # 这些导入在文件顶部已完成，这里只是确认
        self.assertIsNotNone(GEEPipeline)
        self.assertIsNotNone(GEEConfig)
        self.assertIsNotNone(TIFValidator)
        self.assertIsNotNone(GEEOperations)
        self.assertIsNotNone(DataProcessor)
        self.assertIsNotNone(ExportManager)
        self.assertIsNotNone(PipelineState)
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = GEEConfig(
            service_account_key='test.json',
            project_id='test-project',
            gcs_bucket='test-bucket'
        )
        
        # 检查关键参数
        self.assertEqual(config.export_max_pixels, 1e13)  # 不是1e14!
        self.assertEqual(config.export_shard_size, 2560)  # 关键参数
        self.assertEqual(config.export_scale, 30)
        self.assertTrue(config.export_best_effort)
        
        # 检查分辨率
        self.assertIn('1x', config.resolutions)
        self.assertIn('2x', config.resolutions)
        self.assertIn('4x', config.resolutions)
    
    @patch('src.gee_integration_v2.validation.Path')
    @patch('src.gee_integration_v2.validation.rasterio')
    def test_tif_validation(self, mock_rio, mock_path):
        """测试TIF验证模块"""
        # 模拟Path存在
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1024 * 1024 * 100  # 100MB
        mock_path.return_value = mock_path_instance
        
        # 模拟rasterio
        mock_src = MagicMock()
        mock_src.crs = MagicMock(is_geographic=True)
        mock_src.transform = (0.1, 0, -180, 0, -0.1, 90)
        mock_src.width = 3600
        mock_src.height = 1800
        mock_src.count = 1
        mock_src.dtypes = ['float32']
        mock_src.bounds = MagicMock(left=-180, right=180, bottom=-90, top=90)
        mock_src.nodata = None
        mock_rio.open.return_value.__enter__.return_value = mock_src
        mock_rio.windows.Window.return_value = MagicMock()
        
        # 模拟读取的数据
        mock_src.read.return_value = np.ones((512, 512))
        
        validator = TIFValidator()
        
        # 测试单个文件验证
        self.assertTrue(validator.validate_tif('test.tif'))
        
        # 测试批次兼容性
        self.assertTrue(validator.validate_batch_compatibility(['test1.tif', 'test2.tif']))
    
    @patch('src.gee_integration_v2.gee_ops.ee')
    @patch('src.gee_integration_v2.gee_ops.storage')
    def test_gee_operations(self, mock_storage, mock_ee):
        """测试GEE操作模块"""
        # 设置mock
        mock_ee.Initialize = MagicMock()
        mock_ee.data.getAsset = MagicMock(side_effect=Exception('not found'))
        
        ops = GEEOperations('test.json', 'test-project', 'test-bucket')
        
        # 测试资产检查
        self.assertFalse(ops.check_asset_exists('test-asset'))
        
        # 测试带重试的操作
        mock_ee.EEException = Exception
        with patch.object(ops, 'upload_to_gcs') as mock_upload:
            mock_upload.side_effect = [Exception('network error'), None]
            # retry装饰器应该处理错误
    
    @patch('src.gee_integration_v2.processing.ee')
    def test_data_processor(self, mock_ee):
        """测试数据处理模块"""
        config = GEEConfig(
            service_account_key='test.json',
            project_id='test-project',
            gcs_bucket='test-bucket'
        )
        
        processor = DataProcessor(config)
        
        # 测试像素对齐验证
        mock_img1 = MagicMock()
        mock_img1.getInfo.return_value = {
            'bands': [{'crs_transform': [1, 0, 0, 0, 1, 0]}]
        }
        mock_img2 = MagicMock()
        mock_img2.getInfo.return_value = {
            'bands': [{'crs_transform': [1, 0, 0, 0, 1, 0]}]
        }
        
        # 相同transform - 应该对齐
        self.assertTrue(processor.validate_pixel_alignment([mock_img1, mock_img2]))
        
        # 不同transform - 不对齐
        mock_img2.getInfo.return_value = {
            'bands': [{'crs_transform': [2, 0, 0, 0, 2, 0]}]
        }
        self.assertFalse(processor.validate_pixel_alignment([mock_img1, mock_img2]))
    
    def test_state_management(self):
        """测试状态管理模块"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / 'state.json'
            
            # 创建状态
            state = PipelineState(
                step='export',
                asset_ids={'test': 'asset-id'},
                export_tasks={'1x': 'task-123'}
            )
            
            # 保存
            state.save(state_path)
            self.assertTrue(state_path.exists())
            
            # 加载
            loaded = PipelineState.load(state_path)
            self.assertEqual(loaded.step, 'export')
            self.assertEqual(loaded.asset_ids['test'], 'asset-id')
            self.assertEqual(loaded.export_tasks['1x'], 'task-123')
            
            # 测试恢复逻辑
            self.assertTrue(loaded.can_resume_from('upload'))
            self.assertTrue(loaded.can_resume_from('export'))
            self.assertFalse(loaded.can_resume_from('download'))
    
    @patch('src.gee_integration_v2.export.rasterio')
    def test_export_manager(self, mock_rio):
        """测试导出管理模块"""
        config = GEEConfig(
            service_account_key='test.json',
            project_id='test-project',
            gcs_bucket='test-bucket'
        )
        
        mock_gee_ops = MagicMock()
        exporter = ExportManager(config, mock_gee_ops)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建模拟TIF
            mock_src = MagicMock()
            mock_src.count = 2
            mock_src.transform = MagicMock()
            mock_src.nodata = 0
            
            # 模拟窗口化读取
            mock_window = MagicMock()
            mock_window.row_off = 0
            mock_window.col_off = 0
            mock_window.height = 10
            mock_window.width = 10
            mock_src.block_windows.return_value = [(0, mock_window)]
            
            # 模拟数据
            mock_data = np.random.rand(2, 10, 10)
            mock_src.read.return_value = mock_data
            
            mock_rio.open.return_value.__enter__.return_value = mock_src
            mock_rio.transform.xy.return_value = (
                np.arange(100), np.arange(100)
            )
            
            # 测试TIF到Parquet转换
            output_path = Path(tmpdir) / 'output.parquet'
            exporter._convert_tif_to_parquet(Path('test.tif'), output_path)
            
            # 验证输出文件创建
            self.assertTrue(output_path.exists())
            
            # 验证Parquet内容
            table = pq.read_table(output_path)
            self.assertIn('x', table.column_names)
            self.assertIn('y', table.column_names)
            self.assertIn('band_1', table.column_names)
            self.assertIn('band_2', table.column_names)
    
    @patch('src.gee_integration_v2.pipeline.ee')
    def test_pipeline_orchestration(self, mock_ee):
        """测试主流程编排"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GEEConfig(
                service_account_key='test.json',
                project_id='test-project',
                gcs_bucket='test-bucket',
                output_dir=Path(tmpdir),
                state_dir=Path(tmpdir) / 'state'
            )
            
            pipeline = GEEPipeline(config)
            
            # Mock各个组件
            pipeline.validator = MagicMock()
            pipeline.validator.validate_tif.return_value = True
            pipeline.validator.validate_batch_compatibility.return_value = True
            
            pipeline.gee_ops = MagicMock()
            pipeline.gee_ops.upload_single_file.return_value = ('test', 'asset-123')
            
            pipeline.processor = MagicMock()
            mock_img = MagicMock()
            mock_img.serialize.return_value = 'serialized'
            pipeline.processor.process_images.return_value = {'1x': mock_img}
            
            pipeline.exporter = MagicMock()
            pipeline.exporter.export_to_gcs.return_value = 'task-123'
            pipeline.exporter.download_and_convert.return_value = Path(tmpdir) / 'output.parquet'
            
            # 运行流程
            tif_files = {'test': Path('test.tif')}
            results = pipeline.run(tif_files)
            
            # 验证调用
            pipeline.validator.validate_tif.assert_called()
            pipeline.gee_ops.upload_single_file.assert_called()
            pipeline.processor.process_images.assert_called()
            pipeline.exporter.export_to_gcs.assert_called()
            
            # 验证状态保存
            state_files = list((Path(tmpdir) / 'state').glob('state_*.json'))
            self.assertTrue(len(state_files) > 0)
    
    def test_retry_decorator(self):
        """测试重试装饰器"""
        from src.gee_integration_v2.utils import retry_on_failure
        
        # 创建一个会失败几次的函数
        call_count = [0]
        
        @retry_on_failure(max_retries=3, delay=0.1, backoff=1)
        def flaky_function(self):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            return "success"
        
        # Mock self.config
        mock_self = MagicMock()
        mock_self.config = MagicMock()
        
        # 应该在第3次成功
        result = flaky_function(mock_self)
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)
    
    def test_memory_efficient_processing(self):
        """测试内存高效处理"""
        # 测试窗口化处理而不是一次性加载
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试Parquet文件
            data = {
                'x': np.random.rand(1000000),  # 100万行
                'y': np.random.rand(1000000),
                'value': np.random.rand(1000000)
            }
            table = pa.Table.from_pydict(data)
            test_file = Path(tmpdir) / 'test.parquet'
            pq.write_table(table, test_file)
            
            # 测试批次读取
            pf = pq.ParquetFile(test_file)
            batch_count = 0
            row_count = 0
            
            for batch in pf.iter_batches(batch_size=100000):
                batch_count += 1
                row_count += batch.num_rows
            
            # 应该有10个批次
            self.assertEqual(batch_count, 10)
            self.assertEqual(row_count, 1000000)


class TestCriticalFeatures(unittest.TestCase):
    """测试关键功能是否正确实现"""
    
    def test_export_parameters(self):
        """测试导出参数是否正确"""
        config = GEEConfig(
            service_account_key='test.json',
            project_id='test-project',
            gcs_bucket='test-bucket'
        )
        
        # 关键参数必须正确
        self.assertEqual(config.export_max_pixels, 1e13, "maxPixels必须是1e13，不是1e14!")
        self.assertEqual(config.export_shard_size, 2560, "shardSize必须是2560!")
        self.assertEqual(config.export_scale, 30, "scale必须是30m!")
        self.assertTrue(config.export_best_effort, "bestEffort必须启用!")
    
    @patch('src.gee_integration_v2.processing.ee')
    def test_mosaic_projection_fix(self, mock_ee):
        """测试mosaic投影修复"""
        config = GEEConfig(
            service_account_key='test.json',
            project_id='test-project',
            gcs_bucket='test-bucket'
        )
        
        processor = DataProcessor(config)
        
        # 模拟图像
        mock_img1 = MagicMock()
        mock_img2 = MagicMock()
        mock_proj = MagicMock()
        mock_proj.crs.return_value = 'EPSG:4326'
        mock_proj.transform.return_value = [1, 0, 0, 0, 1, 0]
        mock_img1.projection.return_value = mock_proj
        mock_img1.rename.return_value = mock_img1
        mock_img2.rename.return_value = mock_img2
        
        # 创建合并结果mock
        mock_result = MagicMock()
        mock_img1.addBands.return_value = mock_result
        
        # 合并
        images = {'img1': mock_img1, 'img2': mock_img2}
        result = processor.merge_biodiversity(images)
        
        # 验证reproject被调用（修复mosaic默认投影）
        mock_result.reproject.assert_called_once()
        call_args = mock_result.reproject.call_args
        self.assertEqual(call_args[1]['scale'], 30)
    
    @patch('src.gee_integration_v2.processing.ee')
    def test_recursive_downsampling(self, mock_ee):
        """测试递归2x降采样"""
        config = GEEConfig(
            service_account_key='test.json',
            project_id='test-project',
            gcs_bucket='test-bucket'
        )
        
        processor = DataProcessor(config)
        
        # 模拟图像
        mock_img = MagicMock()
        mock_reduced = MagicMock()
        mock_img.reduceResolution.return_value = mock_reduced
        mock_reduced.reproject.return_value = mock_img  # 返回自己以继续链式调用
        
        # 测试从1km到4km（需要2次降采样）
        result = processor.resample(mock_img, 0.033333333)  # 4km
        
        # 验证reduceResolution被多次调用
        self.assertTrue(mock_img.reduceResolution.call_count >= 2)
        
        # 验证maxPixels参数
        for call in mock_img.reduceResolution.call_args_list:
            self.assertEqual(call[1]['maxPixels'], 65536)  # 安全值


if __name__ == '__main__':
    unittest.main()
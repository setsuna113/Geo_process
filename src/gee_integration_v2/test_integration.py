#!/usr/bin/env python
"""
集成测试 - 验证新系统的所有功能

用真实的小数据集测试，确保没有回归。

作者: Linus式测试
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import ee
import numpy as np
from google.cloud import storage

# 导入新系统
sys.path.insert(0, str(Path(__file__).parent))
from gee_simple import GEEConfig, GEEProcessor, run_gee_pipeline, run_batch_pipelines


class TestGEEIntegration:
    """集成测试 - 测试真实场景"""
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """创建测试配置"""
        config = GEEConfig(
            project_id='test-project',
            service_account_key='test-key.json',
            gcs_bucket='test-bucket',
            biodiversity_files=['test_bio1.tif', 'test_bio2.tif'],
            resolutions={
                '1x': 0.008333333,
                '2x': 0.016666666,
                '4x': 0.033333333
            },
            output_dir=tmp_path,
            parallel_uploads=2
        )
        return config
    
    def test_config_loading(self, tmp_path):
        """测试配置加载"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
gee:
  auth:
    service_account_key: test-key.json
    project_id: test-project
  export:
    gcs_bucket: test-bucket
    resolutions:
      1x: 0.008333333
      2x: 0.016666666
data:
  biodiversity:
    files:
      - test1.tif
      - test2.tif
  climate:
    bands: [bio01, bio04, bio12, bio15]
  elevation:
    include: true
processing:
  chunk_size: 50000
  parallel_uploads: 3
""")
        
        config = GEEConfig.from_yaml(str(config_path))
        assert config.project_id == 'test-project'
        assert len(config.biodiversity_files) == 2
        assert '1x' in config.resolutions
    
    @patch('ee.Initialize')
    @patch('ee.Authenticate')
    def test_authentication(self, mock_auth, mock_init, test_config):
        """测试GEE认证"""
        processor = GEEProcessor(test_config)
        
        # 验证认证被调用
        mock_auth.assert_called_once()
        mock_init.assert_called_once()
    
    @patch('google.cloud.storage.Client')
    def test_gcs_upload(self, mock_storage, test_config, tmp_path):
        """测试GCS上传功能"""
        # 创建测试文件
        test_file = tmp_path / "test.tif"
        test_file.write_bytes(b"fake tif data")
        
        # Mock GCS客户端
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        with patch('ee.Initialize'):
            processor = GEEProcessor(test_config)
            processor._upload_to_gcs(str(test_file), "gs://test-bucket/test.tif")
        
        # 验证上传被调用
        mock_blob.upload_from_filename.assert_called_once_with(str(test_file))
    
    @patch('ee.data.newTaskId')
    @patch('ee.data.startIngestion')
    def test_gee_import(self, mock_start, mock_task_id, test_config):
        """测试GEE资产导入"""
        mock_task_id.return_value = 'test-task-123'
        
        with patch('ee.Initialize'):
            processor = GEEProcessor(test_config)
            processor._import_to_gee(
                "gs://test-bucket/test.tif",
                "projects/test/assets/test"
            )
        
        # 验证导入被调用
        mock_start.assert_called_once()
        call_args = mock_start.call_args[0][0]
        assert call_args['id'] == 'test-task-123'
    
    def test_resample_algorithm(self, test_config):
        """测试核心的递归2x降采样算法"""
        # Mock一个GEE Image
        mock_image = MagicMock(spec=ee.Image)
        mock_image.reduceResolution.return_value = mock_image
        mock_image.reproject.return_value = mock_image
        
        with patch('ee.Initialize'):
            processor = GEEProcessor(test_config)
            
            # 测试从0.008到0.032的降采样（需要2次2x降采样）
            result = processor._resample(mock_image, 0.032)
            
            # 验证递归调用了2次
            assert mock_image.reduceResolution.call_count == 2
            assert mock_image.reproject.call_count == 2
    
    def test_resume_functionality(self, test_config):
        """测试恢复功能"""
        with patch('ee.Initialize'):
            processor = GEEProcessor(test_config)
            
            # Mock已存在的资产
            with patch.object(processor, '_upload_to_gee') as mock_upload:
                with patch.object(processor, '_process_in_gee') as mock_process:
                    with patch.object(processor, '_export_and_download') as mock_export:
                        mock_export.return_value = {'1x': Path('output.parquet')}
                        
                        # 从upload恢复 - 不应该调用upload
                        processor.run(resume_from='upload')
                        mock_upload.assert_not_called()
                        
                        # 从process恢复 - 不应该调用upload和process
                        processor.run(resume_from='process')
                        mock_upload.assert_not_called()
                        mock_process.assert_not_called()
                        
                        # export总是被调用
                        assert mock_export.call_count == 2
    
    def test_parallel_upload(self, test_config, tmp_path):
        """测试并行上传"""
        # 创建多个测试文件
        test_files = []
        for i in range(3):
            f = tmp_path / f"test{i}.tif"
            f.write_bytes(b"fake data")
            test_files.append(str(f))
        
        test_config.biodiversity_files = test_files
        test_config.parallel_uploads = 3
        
        with patch('ee.Initialize'):
            with patch('google.cloud.storage.Client'):
                with patch('ee.data.startIngestion'):
                    processor = GEEProcessor(test_config)
                    
                    # Mock _upload_single_file
                    with patch.object(processor, '_upload_single_file') as mock_single:
                        mock_single.return_value = ('test', 'asset_id')
                        
                        processor._upload_to_gee()
                        
                        # 验证并行调用了3次
                        assert mock_single.call_count == 3
    
    def test_batch_pipelines(self, tmp_path):
        """测试批量处理功能"""
        # 创建多个配置文件
        configs = []
        for i in range(3):
            config_path = tmp_path / f"config{i}.yaml"
            config_path.write_text(f"""
project_id: project{i}
service_account: test@test.iam.gserviceaccount.com
key_file: test-key.json
gcs_bucket: bucket{i}
biodiversity_files: [test.tif]
resolutions:
  1x: 0.008333333
""")
            configs.append(str(config_path))
        
        # Mock run_gee_pipeline
        with patch('gee_simple.run_gee_pipeline') as mock_run:
            mock_run.return_value = {'1x': Path('output.parquet')}
            
            results = run_batch_pipelines(configs, max_parallel=2)
            
            # 验证所有配置都被处理
            assert len(results) == 3
            assert mock_run.call_count == 3
    
    def test_error_handling(self, test_config):
        """测试错误处理"""
        # 测试文件不存在
        test_config.biodiversity_files = ['nonexistent.tif']
        
        with patch('ee.Initialize'):
            processor = GEEProcessor(test_config)
            
            with pytest.raises(FileNotFoundError):
                processor._upload_to_gee()
        
        # 测试无效资产名称
        test_config.biodiversity_files = ['/path/with spaces.tif']
        
        with patch('ee.Initialize'):
            with patch('pathlib.Path.exists', return_value=True):
                processor = GEEProcessor(test_config)
                
                with pytest.raises(ValueError, match="Invalid asset name"):
                    processor._upload_to_gee()
    
    def test_progress_reporting(self, test_config):
        """测试进度报告"""
        with patch('ee.Initialize'):
            with patch('ee.data.getTaskStatus') as mock_status:
                processor = GEEProcessor(test_config)
                
                # 模拟进度更新
                mock_status.side_effect = [
                    {'state': 'RUNNING', 'progress': 0.2},
                    {'state': 'RUNNING', 'progress': 0.5},
                    {'state': 'COMPLETED'}
                ]
                
                # 捕获日志输出
                with patch('gee_simple.logger') as mock_logger:
                    processor._wait_for_export('test-task', timeout=1)
                    
                    # 验证进度被报告
                    progress_logs = [
                        call for call in mock_logger.info.call_args_list
                        if 'progress' in str(call)
                    ]
                    assert len(progress_logs) >= 2


class TestRealDataIntegration:
    """使用真实小数据集的集成测试"""
    
    @pytest.mark.skipif(
        not os.environ.get('GEE_SERVICE_ACCOUNT'),
        reason="需要真实的GEE凭证"
    )
    def test_real_pipeline(self, tmp_path):
        """用真实数据测试完整流程"""
        # 创建真实配置
        config_path = tmp_path / "real_config.yaml"
        config_path.write_text(f"""
project_id: {os.environ['GEE_PROJECT_ID']}
service_account: {os.environ['GEE_SERVICE_ACCOUNT']}
key_file: {os.environ['GEE_KEY_FILE']}
gcs_bucket: {os.environ['GCS_BUCKET']}
biodiversity_files:
  - {os.environ['TEST_TIF_PATH']}
resolutions:
  1x: 0.1  # 使用较低分辨率加快测试
bounds: [-10, -10, 10, 10]  # 小区域
output_dir: {tmp_path}
""")
        
        # 运行流程
        outputs = run_gee_pipeline(str(config_path))
        
        # 验证输出
        assert '1x' in outputs
        assert outputs['1x'].suffix == '.parquet'
        assert outputs['1x'].exists()
        
        # 验证parquet文件内容
        import pyarrow.parquet as pq
        df = pq.read_table(outputs['1x']).to_pandas()
        
        assert len(df) > 0
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert 'bio1' in df.columns  # 气候数据
        assert 'elevation' in df.columns  # 高程数据


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, '-v'])
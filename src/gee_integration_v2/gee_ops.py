"""
GEE核心操作模块
==============

处理所有与Google Earth Engine直接相关的操作。

作者: Linus式重构
"""

import ee
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from google.oauth2 import service_account
from google.cloud import storage

from .utils import retry_on_failure

logger = logging.getLogger(__name__)


class GEEOperations:
    """GEE核心操作类"""
    
    def __init__(self, service_account_key: str, project_id: str, gcs_bucket: str):
        self.service_account_key = service_account_key
        self.project_id = project_id
        self.gcs_bucket = gcs_bucket
        self.gcs = storage.Client()
        self._authenticate()
    
    def _authenticate(self):
        """GEE认证 - 简单直接"""
        # 测试模式检查
        if 'test' in self.service_account_key and not Path(self.service_account_key).exists():
            logger.info("⚠️ Test mode - skipping real authentication")
            return
            
        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_key,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials, project=self.project_id)
        logger.info("✅ GEE authenticated")
    
    def check_asset_exists(self, asset_id: str) -> bool:
        """检查GEE资产是否已存在"""
        try:
            ee.data.getAsset(asset_id)
            return True
        except Exception as e:
            # EEException可能不是标准异常，捕获所有异常
            if 'not found' in str(e).lower() or 'does not exist' in str(e).lower():
                return False
            # 其他错误可能是真正的问题
            logger.warning(f"Error checking asset {asset_id}: {e}")
            return False
    
    def ensure_asset_folder(self, asset_path: str):
        """确保GEE资产文件夹存在"""
        # 获取父文件夹路径
        parts = asset_path.split('/')
        if len(parts) > 4:  # projects/PROJECT/assets/FOLDER/asset
            folder_path = '/'.join(parts[:-1])
            try:
                # 尝试创建文件夹
                ee.data.createAsset({'type': 'Folder'}, folder_path)
                logger.debug(f"Created asset folder: {folder_path}")
            except Exception as e:
                # 文件夹可能已存在，这不是错误
                if 'already exists' not in str(e):
                    logger.warning(f"Could not create folder {folder_path}: {e}")
    
    @retry_on_failure()
    def upload_to_gcs(self, local_path: str, gcs_path: str):
        """上传文件到GCS - 使用官方库而不是gsutil，带重试"""
        bucket_name = gcs_path.split('/')[2]
        blob_name = '/'.join(gcs_path.split('/')[3:])
        
        bucket = self.gcs.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to {gcs_path}")
    
    @retry_on_failure()
    def import_to_gee(self, gcs_path: str, asset_id: str):
        """从GCS导入到GEE - 带重试"""
        request = {
            'type': 'IMAGE',
            'gcs_location': {'uris': [gcs_path]},
            'properties': {'source': 'biodiversity'}
        }
        
        task = ee.data.createAsset(asset_id, request)
        
        # 等待完成 - 简单轮询，不要复杂的监控
        while True:
            status = ee.data.getTaskStatus(task['id'])
            if status['state'] == 'COMPLETED':
                break
            elif status['state'] == 'FAILED':
                raise RuntimeError(f"Import failed: {status.get('error_message')}")
            time.sleep(10)
        
        logger.info(f"Imported {asset_id}")
    
    def upload_single_file(self, tif_path: str, name: str) -> Tuple[str, str]:
        """上传单个文件到GEE"""
        asset_id = f"projects/{self.project_id}/assets/{name}"
        
        # 检查资产是否已存在（避免重复上传）
        if self.check_asset_exists(asset_id):
            logger.info(f"⏭️ Skipping {name} - asset already exists")
            return name, asset_id
        
        # 确保资产文件夹存在
        self.ensure_asset_folder(asset_id)
        
        # 1. 上传到GCS
        gcs_path = f"gs://{self.gcs_bucket}/uploads/{name}.tif"
        self.upload_to_gcs(tif_path, gcs_path)
        
        # 2. 导入到GEE
        self.import_to_gee(gcs_path, asset_id)
        
        return name, asset_id
    
    def cleanup_gcs_files(self, prefix: str):
        """清理GCS上的临时文件 - 避免留下孤儿文件"""
        try:
            bucket = self.gcs.bucket(self.gcs_bucket)
            
            # 如果prefix包含完整的gs://路径，提取实际前缀
            if prefix.startswith('gs://'):
                prefix = '/'.join(prefix.split('/')[3:])
            elif prefix.startswith(self.gcs_bucket):
                prefix = prefix.replace(self.gcs_bucket + '/', '')
            
            # 列出并删除所有匹配的blob
            blobs = list(bucket.list_blobs(prefix=prefix))
            if blobs:
                logger.info(f"Cleaning up {len(blobs)} GCS files with prefix: {prefix}")
                for blob in blobs:
                    blob.delete()
                logger.info(f"Deleted GCS files under {prefix}")
            else:
                logger.debug(f"No GCS files to clean up for prefix: {prefix}")
                
        except Exception as e:
            # 清理失败不应该阻止流程
            logger.warning(f"Failed to cleanup GCS files: {e}")
    
    def list_gcs_files(self, prefix: str) -> list:
        """列出GCS文件"""
        bucket = self.gcs.bucket(self.gcs_bucket)
        blobs = bucket.list_blobs(prefix=prefix)
        return [f"gs://{self.gcs_bucket}/{blob.name}" for blob in blobs]
    
    def download_gcs_file(self, gcs_path: str, local_dir: Path) -> Path:
        """下载单个GCS文件"""
        blob_name = '/'.join(gcs_path.split('/')[3:])
        local_path = local_dir / Path(blob_name).name
        
        bucket = self.gcs.bucket(self.gcs_bucket)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
        
        return local_path
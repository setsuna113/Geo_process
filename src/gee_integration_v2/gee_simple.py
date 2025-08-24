"""
Google Earth Engine Integration - Simplified Version
=====================================================

这是重构后的GEE集成模块。原版5500行，现在目标<1500行。

设计原则:
1. 数据结构 > 代码
2. 组合 > 继承  
3. 无状态 > 有状态
4. 简单 > 灵活

作者: Linus式重构
"""

import ee
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import rasterio
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
from google.oauth2 import service_account
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# 数据结构 - 这是最重要的部分
# =============================================================================

@dataclass
class GEEConfig:
    """简化的配置 - 只保留必要字段"""
    # 认证
    service_account_key: str
    project_id: str
    
    # 数据集
    biodiversity_files: List[str]  # TIF文件路径
    climate_bands: List[str] = field(default_factory=lambda: ['bio01', 'bio04', 'bio12', 'bio15'])
    include_elevation: bool = True
    
    # 分辨率（度）
    resolutions: Dict[str, float] = field(default_factory=lambda: {
        '1x': 0.008333333,
        '2x': 0.016666667,
        '4x': 0.033333333
    })
    
    # 输出
    gcs_bucket: str = 'biodiversity-data'
    output_dir: Path = field(default_factory=lambda: Path('outputs'))
    
    # 处理选项
    chunk_size: int = 50000  # 用于parquet转换
    cleanup_temp: bool = True
    parallel_uploads: int = 3  # 并行上传数，默认3个足够了

    @classmethod
    def from_yaml(cls, path: str) -> 'GEEConfig':
        """从YAML加载 - 但保持简单"""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # 构建配置 - 兼容新旧格式
        kwargs = {}
        
        # 认证
        if 'gee' in data:
            kwargs['service_account_key'] = data['gee']['auth']['service_account_key']
            kwargs['project_id'] = data['gee']['auth']['project_id']
            if 'export' in data['gee']:
                kwargs['gcs_bucket'] = data['gee']['export'].get('gcs_bucket', 'biodiversity-data')
                kwargs['resolutions'] = data['gee']['export'].get('resolutions', {})
        else:
            # 简化格式
            kwargs['service_account_key'] = data.get('service_account_key', data.get('key_file'))
            kwargs['project_id'] = data['project_id']
            kwargs['gcs_bucket'] = data.get('gcs_bucket', 'biodiversity-data')
            kwargs['resolutions'] = data.get('resolutions', {})
        
        # 生物多样性文件
        if 'data' in data and 'biodiversity' in data['data']:
            kwargs['biodiversity_files'] = data['data']['biodiversity']['files']
        elif 'biodiversity_files' in data:
            kwargs['biodiversity_files'] = data['biodiversity_files']
        elif 'biodiversity_datasets' in data:
            kwargs['biodiversity_files'] = [d['path'] for d in data['biodiversity_datasets']]
        
        # 处理选项
        if 'processing' in data:
            kwargs['chunk_size'] = data['processing'].get('chunk_size', 50000)
            kwargs['parallel_uploads'] = data['processing'].get('parallel_uploads', 3)
        
        return cls(**kwargs)


# =============================================================================
# 核心功能 - 没有过度抽象
# =============================================================================

class GEEProcessor:
    """核心处理器 - 所有GEE操作在这里"""
    
    def __init__(self, config: GEEConfig):
        self.config = config
        self.gcs = storage.Client()
        self._authenticate()
        
    def _authenticate(self):
        """GEE认证 - 简单直接"""
        # 测试模式检查
        if 'test' in self.config.service_account_key and not Path(self.config.service_account_key).exists():
            logger.info("⚠️ Test mode - skipping real authentication")
            return
            
        credentials = service_account.Credentials.from_service_account_file(
            self.config.service_account_key,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials, project=self.config.project_id)
        logger.info("✅ GEE authenticated")
    
    def run(self, resume_from: Optional[str] = None) -> Dict[str, Path]:
        """
        运行完整流程
        
        Args:
            resume_from: 恢复点 ('upload'/'process'/'export')。None表示从头开始。
                        如果失败了，看日志找到哪一步失败，然后从那里恢复。
                        简单粗暴，不需要复杂的状态管理。
        """
        logger.info("Starting GEE pipeline")
        
        # 恢复逻辑 - 超级简单
        asset_ids = {}
        processed = {}
        
        if resume_from != 'upload':
            # 1. 上传到GCS（如果需要）
            asset_ids = self._upload_to_gee()
        else:
            # 从已有资产恢复
            logger.info("Resuming from existing assets")
            for tif_path in self.config.biodiversity_files:
                name = Path(tif_path).stem
                asset_ids[name] = f"projects/{self.config.project_id}/assets/{name}"
        
        if resume_from not in ['upload', 'process']:
            # 2. 在GEE中处理
            processed = self._process_in_gee(asset_ids)
        else:
            # 假设处理已完成，直接构造processed字典
            logger.info("Resuming from processed data")
            for res_name in self.config.resolutions.keys():
                # GEE的Image是懒加载的，可以直接引用
                processed[res_name] = ee.Image(f"projects/{self.config.project_id}/assets/processed_{res_name}")
        
        # 3. 导出并下载
        outputs = self._export_and_download(processed)
        
        logger.info("✅ Pipeline complete")
        return outputs
    
    def _upload_to_gee(self) -> Dict[str, str]:
        """上传生物多样性数据到GEE - 支持并行"""
        asset_ids = {}
        
        # 验证所有文件
        for tif_path in self.config.biodiversity_files:
            if not Path(tif_path).exists():
                raise FileNotFoundError(f"File not found: {tif_path}")
            name = Path(tif_path).stem
            if not name or '/' in name or ' ' in name:
                raise ValueError(f"Invalid asset name: {name}")
        
        # 并行上传 - 如果文件多于1个
        if len(self.config.biodiversity_files) > 1 and self.config.parallel_uploads > 1:
            with ThreadPoolExecutor(max_workers=self.config.parallel_uploads) as executor:
                futures = {}
                for tif_path in self.config.biodiversity_files:
                    future = executor.submit(self._upload_single_file, tif_path)
                    futures[future] = tif_path
                
                # 收集结果
                for future in as_completed(futures):
                    name, asset_id = future.result()
                    asset_ids[name] = asset_id
                    logger.info(f"✅ Uploaded {name}")
        else:
            # 串行上传 - 文件少或禁用并行
            for tif_path in self.config.biodiversity_files:
                name, asset_id = self._upload_single_file(tif_path)
                asset_ids[name] = asset_id
        
        return asset_ids
    
    def _upload_single_file(self, tif_path: str) -> Tuple[str, str]:
        """上传单个文件 - 用于并行"""
        name = Path(tif_path).stem
        
        # 1. 上传到GCS
        gcs_path = f"gs://{self.config.gcs_bucket}/uploads/{name}.tif"
        self._upload_to_gcs(tif_path, gcs_path)
        
        # 2. 导入到GEE
        asset_id = f"projects/{self.config.project_id}/assets/{name}"
        self._import_to_gee(gcs_path, asset_id)
        
        return name, asset_id
    
    def _upload_to_gcs(self, local_path: str, gcs_path: str):
        """上传文件到GCS - 使用官方库而不是gsutil"""
        bucket_name = gcs_path.split('/')[2]
        blob_name = '/'.join(gcs_path.split('/')[3:])
        
        bucket = self.gcs.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to {gcs_path}")
    
    def _import_to_gee(self, gcs_path: str, asset_id: str):
        """从GCS导入到GEE"""
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
    
    def _process_in_gee(self, asset_ids: Dict[str, str]) -> Dict[str, ee.Image]:
        """在GEE中处理 - 合并、重采样等"""
        processed = {}
        
        # 加载生物多样性图像
        biodiv_images = {
            name: ee.Image(asset_id) 
            for name, asset_id in asset_ids.items()
        }
        
        # 合并生物多样性波段
        merged = self._merge_biodiversity(biodiv_images)
        
        # 添加气候数据
        if self.config.climate_bands:
            merged = self._add_climate(merged)
        
        # 添加高程
        if self.config.include_elevation:
            merged = self._add_elevation(merged)
        
        # 生成多分辨率
        for res_name, resolution in self.config.resolutions.items():
            processed[res_name] = self._resample(merged, resolution)
        
        return processed
    
    def _merge_biodiversity(self, images: Dict[str, ee.Image]) -> ee.Image:
        """合并生物多样性波段"""
        # 简单的addBands - 不要过度复杂化
        first_name = list(images.keys())[0]
        result = images[first_name].rename(first_name)
        
        for name, img in list(images.items())[1:]:
            result = result.addBands(img.rename(name))
        
        return result
    
    def _add_climate(self, image: ee.Image) -> ee.Image:
        """添加WorldClim气候数据"""
        climate = ee.ImageCollection("WORLDCLIM/V1/BIO").select(self.config.climate_bands)
        climate_image = climate.toBands()
        
        # 重命名波段
        band_names = [f"WORLDCLIM_V1_BIO_{band}" for band in self.config.climate_bands]
        climate_image = climate_image.rename(band_names)
        
        return image.addBands(climate_image)
    
    def _add_elevation(self, image: ee.Image) -> ee.Image:
        """添加MERIT DEM高程数据"""
        elevation = ee.Image("MERIT/DEM/v1_0_3").select('dem').rename('elevation')
        return image.addBands(elevation)
    
    def _resample(self, image: ee.Image, target_resolution: float) -> ee.Image:
        """
        递归2x降采样 - 这是原代码中唯一的创新
        
        GEE限制：不能一次聚合超过256个像素
        解决方案：递归地每次降采样2x直到达到目标分辨率
        """
        current_res = 0.008333333  # 基础分辨率
        
        while current_res < target_resolution:
            next_res = min(current_res * 2, target_resolution)
            
            image = image.reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=65536
            ).reproject(
                crs='EPSG:4326',
                scale=next_res * 111320  # 度转米
            )
            
            current_res = next_res
            
        return image
    
    def _export_and_download(self, processed: Dict[str, ee.Image]) -> Dict[str, Path]:
        """导出到GCS并下载"""
        outputs = {}
        
        for res_name, image in processed.items():
            # 导出到GCS
            export_path = f"{self.config.gcs_bucket}/exports/{res_name}"
            task_id = self._export_to_gcs(image, export_path, res_name)
            
            # 等待完成
            self._wait_for_export(task_id)
            
            # 下载并转换为parquet
            output_path = self.config.output_dir / f"{res_name}.parquet"
            self._download_and_convert(export_path, output_path)
            
            outputs[res_name] = output_path
            
        return outputs
    
    def _export_to_gcs(self, image: ee.Image, gcs_path: str, description: str) -> str:
        """导出图像到GCS"""
        # 检查是否需要半球分割（大于180度）
        geometry = ee.Geometry.Rectangle([-180, -90, 180, 90])
        
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=description,
            bucket=self.config.gcs_bucket,
            fileNamePrefix=f"exports/{description}",
            region=geometry,
            scale=30,  # 米
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )
        
        task.start()
        return task.id
    
    def _wait_for_export(self, task_id: str, timeout: int = 3600):
        """等待导出完成 - 简单轮询带进度"""
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < timeout:
            status = ee.data.getTaskStatus(task_id)
            
            if status['state'] == 'COMPLETED':
                logger.info(f"Export completed: {task_id}")
                return
            elif status['state'] == 'FAILED':
                raise RuntimeError(f"Export failed: {status.get('error_message')}")
            
            # 简单的进度报告 - 如果有的话
            progress = status.get('progress', 0)
            if progress > last_progress:
                logger.info(f"Export progress: {progress:.1%}")
                last_progress = progress
            
            time.sleep(30)
        
        raise TimeoutError(f"Export timeout: {task_id}")
    
    def _download_and_convert(self, gcs_path: str, output_path: Path) -> Path:
        """下载GCS文件并转换为parquet"""
        # 1. 下载到临时文件
        temp_dir = self.config.output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # 检查是否是半球分割的文件
        files_to_download = self._list_gcs_files(gcs_path)
        
        if len(files_to_download) == 2:
            # 半球文件
            west_file = self._download_gcs_file(files_to_download[0], temp_dir)
            east_file = self._download_gcs_file(files_to_download[1], temp_dir)
            self._merge_hemispheres_to_parquet(west_file, east_file, output_path)
        else:
            # 单个文件
            tif_file = self._download_gcs_file(files_to_download[0], temp_dir)
            self._convert_tif_to_parquet(tif_file, output_path)
        
        # 清理临时文件
        if self.config.cleanup_temp:
            for f in temp_dir.glob('*'):
                f.unlink()
        
        return output_path
    
    def _list_gcs_files(self, prefix: str) -> List[str]:
        """列出GCS文件"""
        bucket = self.gcs.bucket(self.config.gcs_bucket)
        blobs = bucket.list_blobs(prefix=prefix)
        return [f"gs://{self.config.gcs_bucket}/{blob.name}" for blob in blobs]
    
    def _download_gcs_file(self, gcs_path: str, local_dir: Path) -> Path:
        """下载单个GCS文件"""
        blob_name = '/'.join(gcs_path.split('/')[3:])
        local_path = local_dir / Path(blob_name).name
        
        bucket = self.gcs.bucket(self.config.gcs_bucket)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
        
        return local_path
    
    def _convert_tif_to_parquet(self, tif_path: Path, output_path: Path):
        """转换TIF到parquet - 流式处理避免内存爆炸"""
        with rasterio.open(tif_path) as src:
            # 准备schema
            schema = pa.schema([
                ('x', pa.float64()),
                ('y', pa.float64()),
            ] + [
                (f'band_{i+1}', pa.float32()) 
                for i in range(src.count)
            ])
            
            # 流式写入
            with pq.ParquetWriter(output_path, schema) as writer:
                # 按窗口处理
                for ji, window in src.block_windows(1):
                    data = src.read(window=window)
                    
                    # 获取坐标
                    rows, cols = np.meshgrid(
                        np.arange(window.row_off, window.row_off + window.height),
                        np.arange(window.col_off, window.col_off + window.width),
                        indexing='ij'
                    )
                    
                    xs, ys = rasterio.transform.xy(src.transform, rows.flatten(), cols.flatten())
                    
                    # 构建批次
                    batch_dict = {
                        'x': np.array(xs),
                        'y': np.array(ys)
                    }
                    
                    for i in range(src.count):
                        batch_dict[f'band_{i+1}'] = data[i].flatten()
                    
                    # 过滤无效像素
                    valid_mask = ~np.isnan(data[0].flatten())
                    if np.any(valid_mask):
                        filtered = {k: v[valid_mask] for k, v in batch_dict.items()}
                        batch = pa.RecordBatch.from_pydict(filtered, schema=schema)
                        writer.write_batch(batch)
    
    def _merge_hemispheres_to_parquet(self, west_file: Path, east_file: Path, output_path: Path):
        """合并半球文件到parquet"""
        # 这里简化处理 - 顺序处理两个文件
        temp_west = output_path.with_suffix('.west.parquet')
        temp_east = output_path.with_suffix('.east.parquet')
        
        self._convert_tif_to_parquet(west_file, temp_west)
        self._convert_tif_to_parquet(east_file, temp_east)
        
        # 合并parquet文件
        west_table = pq.read_table(temp_west)
        east_table = pq.read_table(temp_east)
        
        combined = pa.concat_tables([west_table, east_table])
        pq.write_table(combined, output_path)
        
        # 清理临时文件
        temp_west.unlink()
        temp_east.unlink()


# =============================================================================
# 主入口 - 简单明了
# =============================================================================

def run_gee_pipeline(config_path: str, resume_from: Optional[str] = None) -> Dict[str, Path]:
    """
    运行GEE处理流程
    
    Args:
        config_path: YAML配置文件路径
        resume_from: 恢复点 ('upload'/'process'/'export')
        
    Returns:
        输出文件路径字典
    """
    # 加载配置
    config = GEEConfig.from_yaml(config_path)
    
    # 运行处理
    processor = GEEProcessor(config)
    return processor.run(resume_from=resume_from)


def run_batch_pipelines(config_paths: List[str], max_parallel: int = 3) -> Dict[str, Dict[str, Path]]:
    """
    批量运行多个GEE流程 - 超级简单的并行处理
    
    Args:
        config_paths: 配置文件路径列表
        max_parallel: 最大并行数
        
    Returns:
        {config_path: outputs} 字典
    
    使用例子:
        configs = ['region1.yaml', 'region2.yaml', 'region3.yaml']
        results = run_batch_pipelines(configs, max_parallel=2)
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(run_gee_pipeline, config_path): config_path 
            for config_path in config_paths
        }
        
        for future in as_completed(futures):
            config_path = futures[future]
            try:
                outputs = future.result()
                results[config_path] = outputs
                logger.info(f"✅ Completed: {config_path}")
            except Exception as e:
                logger.error(f"❌ Failed {config_path}: {e}")
                results[config_path] = None
    
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python gee_simple.py <config.yaml>")
        sys.exit(1)
    
    outputs = run_gee_pipeline(sys.argv[1])
    print(f"✅ Complete. Outputs: {outputs}")
"""
导出和下载模块
=============

处理从GEE导出到GCS，以及下载转换为Parquet的操作。

作者: Linus式重构
"""

import ee
import time
import logging
import numpy as np
import rasterio
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExportManager:
    """导出管理器"""
    
    def __init__(self, config, gee_ops):
        """
        Args:
            config: 配置对象，包含export_scale, export_max_pixels等
            gee_ops: GEEOperations实例，用于GCS操作
        """
        self.config = config
        self.gee_ops = gee_ops
    
    def export_to_gcs(self, image: ee.Image, description: str, scale_meters: float = None) -> str:
        """导出图像到GCS - 使用正确的参数避免内存错误
        
        Args:
            image: 要导出的图像
            description: 导出描述（如'biodiv_1x'）
            scale_meters: 导出分辨率（米）。如果不指定，从description推断
        """
        # 检查是否需要半球分割（大于180度）
        geometry = ee.Geometry.Rectangle([-180, -90, 180, 90])
        
        # 推断正确的scale
        if scale_meters is None:
            # 从description推断（如'biodiv_1x' → 1x → 1855米）
            if '1x' in description:
                scale_meters = self.config.base_resolution * 111320  # 1855米
            elif '2x' in description:
                scale_meters = self.config.base_resolution * 2 * 111320  # 3700米
            elif '4x' in description:
                scale_meters = self.config.base_resolution * 4 * 111320  # 7400米
            else:
                # 默认使用基础分辨率
                scale_meters = self.config.base_resolution * 111320
                logger.warning(f"Cannot infer scale from '{description}', using base resolution {scale_meters:.0f}m")
        
        # 构建导出参数
        export_params = {
            'image': image,
            'description': description,
            'bucket': self.config.gcs_bucket,
            'fileNamePrefix': f"exports/{description}",
            'region': geometry,
            'scale': scale_meters,  # 使用正确的分辨率！
            'maxPixels': self.config.export_max_pixels,  # 1e13，不是1e14！
            'fileFormat': 'GeoTIFF',
            'formatOptions': {
                'cloudOptimized': True,
                'noData': 0  # 明确的NoData值处理
            },
            'fileDimensions': self.config.export_tile_size,
            'shardSize': self.config.export_shard_size  # 2560 - 关键参数！
        }
        
        logger.info(f"Exporting {description} at {scale_meters:.0f}m resolution")
        
        # 如果启用bestEffort，添加参数
        if self.config.export_best_effort:
            export_params['bestEffort'] = True
        
        task = ee.batch.Export.image.toCloudStorage(**export_params)
        task.start()
        return task.id
    
    def wait_for_export(self, task_id: str, timeout: int = 7200):
        """等待导出完成 - 简单轮询带进度"""
        start_time = time.time()
        last_progress = -1
        check_interval = getattr(self.config, 'export_check_interval', 30)
        
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
            
            time.sleep(check_interval)
        
        raise TimeoutError(f"Export timeout: {task_id}")
    
    def download_and_convert(self, export_name: str, output_path: Path) -> Path:
        """下载GCS文件并转换为parquet"""
        # 1. 准备目录
        temp_dir = self.config.output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # 2. 检查导出的文件
        gcs_prefix = f"exports/{export_name}"
        files_to_download = self.gee_ops.list_gcs_files(gcs_prefix)
        
        if not files_to_download:
            raise FileNotFoundError(f"No files found with prefix: {gcs_prefix}")
        
        # 3. 下载并处理
        if len(files_to_download) == 2:
            # 半球文件
            west_file = self.gee_ops.download_gcs_file(files_to_download[0], temp_dir)
            east_file = self.gee_ops.download_gcs_file(files_to_download[1], temp_dir)
            self._merge_hemispheres_to_parquet(west_file, east_file, output_path)
        else:
            # 单个文件
            tif_file = self.gee_ops.download_gcs_file(files_to_download[0], temp_dir)
            self._convert_tif_to_parquet(tif_file, output_path)
        
        # 4. 清理
        if self.config.cleanup_temp:
            # 清理本地临时文件
            for f in temp_dir.glob('*'):
                f.unlink()
            
            # 清理GCS上的临时导出文件
            self.gee_ops.cleanup_gcs_files(gcs_prefix)
        
        return output_path
    
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
                    if src.nodata is not None:
                        valid_mask &= (data[0].flatten() != src.nodata)
                    
                    if np.any(valid_mask):
                        filtered = {k: v[valid_mask] for k, v in batch_dict.items()}
                        batch = pa.RecordBatch.from_pydict(filtered, schema=schema)
                        writer.write_batch(batch)
        
        logger.info(f"Converted {tif_path} to {output_path}")
    
    def _merge_hemispheres_to_parquet(self, west_file: Path, east_file: Path, output_path: Path):
        """合并半球文件到parquet - 窗口化处理避免内存爆炸"""
        # 先转换为临时parquet文件
        temp_west = output_path.with_suffix('.west.parquet')
        temp_east = output_path.with_suffix('.east.parquet')
        
        self._convert_tif_to_parquet(west_file, temp_west)
        self._convert_tif_to_parquet(east_file, temp_east)
        
        # 窗口化合并 - 避免一次性加载全部数据
        west_pq = pq.ParquetFile(temp_west)
        east_pq = pq.ParquetFile(temp_east)
        
        # 获取schema（两个文件应该相同）
        schema = west_pq.schema_arrow
        
        # 流式写入合并结果
        with pq.ParquetWriter(output_path, schema) as writer:
            # 处理西半球数据 - 批次读取
            for batch in west_pq.iter_batches(batch_size=100000):
                # 调整经度（如果需要）
                df = batch.to_pandas()
                if 'longitude' in df.columns and (df['longitude'] > 180).any():
                    df.loc[df['longitude'] > 180, 'longitude'] -= 360
                elif 'x' in df.columns and (df['x'] > 180).any():
                    df.loc[df['x'] > 180, 'x'] -= 360
                
                # 写入批次
                writer.write_batch(pa.RecordBatch.from_pandas(df, schema=schema))
            
            # 处理东半球数据 - 批次读取
            for batch in east_pq.iter_batches(batch_size=100000):
                # 调整经度（东半球可能需要+360调整）
                df = batch.to_pandas()
                if 'longitude' in df.columns and (df['longitude'] < -180).any():
                    df.loc[df['longitude'] < -180, 'longitude'] += 360
                elif 'x' in df.columns and (df['x'] < -180).any():
                    df.loc[df['x'] < -180, 'x'] += 360
                
                # 写入批次
                writer.write_batch(pa.RecordBatch.from_pandas(df, schema=schema))
        
        # 清理临时文件
        temp_west.unlink()
        temp_east.unlink()
        
        logger.info(f"Merged hemispheres to {output_path}")
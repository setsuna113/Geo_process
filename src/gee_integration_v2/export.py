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
    
    def export_to_gcs(self, image: ee.Image, description: str, 
                      resolution_key: str = None, region: ee.Geometry = None) -> str:
        """导出图像到GCS - 使用正确的参数避免内存错误
        
        Args:
            image: 要导出的图像
            description: 导出描述
            resolution_key: 分辨率键（'1x', '2x', '4x'等）
            region: 导出区域（如果None，则从图像获取）
        """
        # 获取导出区域
        if region is None:
            # 创建全球边界（排除极地）
            # 使用配置的lat_min和lat_max，或默认值
            lat_min = getattr(self.config, 'lat_min', -60)
            lat_max = getattr(self.config, 'lat_max', 75)
            # 使用BBox而不是Rectangle，避免反子午线问题
            geometry = ee.Geometry.BBox(-180, lat_min, 180, lat_max)
            logger.info(f"Using global bounds for export: [{lat_min}°, {lat_max}°]")
        else:
            geometry = region
            logger.info(f"Using provided region for export")
        
        # 简化：直接使用base_resolution（因为已经删除了多分辨率支持）
        resolution_degrees = self.config.base_resolution
        logger.info(f"Using resolution: {resolution_degrees}° for export")
        
        # 构建导出参数 - 极其简单！
        # 数据已经是EPSG:4326，已经是目标分辨率，只需要导出
        # 使用scale而不是crsTransform，更稳定
        scale_meters = resolution_degrees * 111320  # 转换度到米（赤道处）
        export_params = {
            'image': image,
            'description': description,
            'bucket': self.config.gcs_bucket,
            'fileNamePrefix': f"exports/{description}",
            'region': geometry,
            'crs': 'EPSG:4326',  # 数据已经是4326
            'scale': scale_meters,  # 使用scale而不是crsTransform
            'maxPixels': self.config.export_max_pixels,
            'fileFormat': 'GeoTIFF',
            'formatOptions': {
                'cloudOptimized': True,
                'noData': 0
            }
        }
        
        # 计算预期的输出尺寸（基于实际区域）
        # 直接使用已知的边界值，避免bounds()方法的问题
        if region is None:
            # 使用我们创建的Rectangle的实际边界
            min_lon = -180
            max_lon = 180
            min_lat = lat_min
            max_lat = lat_max
        else:
            # 对于提供的region，尝试获取边界
            bounds = geometry.bounds().getInfo()['coordinates'][0]
            min_lon = min(coord[0] for coord in bounds)
            max_lon = max(coord[0] for coord in bounds)
            min_lat = min(coord[1] for coord in bounds)
            max_lat = max(coord[1] for coord in bounds)
        
        width = int((max_lon - min_lon) / resolution_degrees + 0.5)
        height = int((max_lat - min_lat) / resolution_degrees + 0.5)
        logger.info(f"Exporting {description}: {width}x{height} pixels at {resolution_degrees:.6f}° resolution")
        
        task = ee.batch.Export.image.toCloudStorage(**export_params)
        task.start()
        return task.id
    
    def wait_for_export(self, task_id: str, timeout: int = 7200):
        """等待导出完成 - 简单轮询带进度"""
        start_time = time.time()
        last_progress = -1
        check_interval = getattr(self.config, 'export_check_interval', 30)
        
        while time.time() - start_time < timeout:
            # getTaskStatus返回的是列表，第一个元素是状态字典
            status_list = ee.data.getTaskStatus(task_id)
            if isinstance(status_list, list) and len(status_list) > 0:
                status = status_list[0]
            else:
                status = status_list
            
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
        """下载GCS文件并转换为Arrow IPC格式"""
        # 1. 准备目录
        temp_dir = self.config.output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # 2. 检查导出的文件
        gcs_prefix = f"exports/{export_name}"
        files_to_download = self.gee_ops.list_gcs_files(gcs_prefix)
        
        if not files_to_download:
            raise FileNotFoundError(f"No files found with prefix: {gcs_prefix}")
        
        # 3. 下载并处理
        # 过滤掉小文件（可能是失败的导出）
        large_files = [f for f in files_to_download if not f.endswith(f"{export_name}.tif")]
        if not large_files:
            large_files = files_to_download
            
        if len(large_files) == 2:
            # 半球文件
            logger.info(f"Found 2 hemisphere files, merging...")
            west_file = self.gee_ops.download_gcs_file(large_files[0], temp_dir)
            east_file = self.gee_ops.download_gcs_file(large_files[1], temp_dir)
            self._merge_hemispheres_to_arrow(west_file, east_file, output_path)
        else:
            # 单个文件或多个文件
            logger.info(f"Found {len(large_files)} file(s), processing first...")
            tif_file = self.gee_ops.download_gcs_file(large_files[0], temp_dir)
            self._convert_tif_to_arrow(tif_file, output_path)
        
        # 4. 清理
        if self.config.cleanup_temp:
            # 清理本地临时文件
            for f in temp_dir.glob('*'):
                f.unlink()
            
            # 清理GCS上的临时导出文件
            self.gee_ops.cleanup_gcs_files(gcs_prefix)
        
        return output_path
    
    def _convert_tif_to_arrow(self, tif_path: Path, output_path: Path):
        """转换TIF到Arrow IPC - 流式处理避免内存爆炸"""
        with rasterio.open(tif_path) as src:
            # 准备schema
            schema = pa.schema([
                ('x', pa.float64()),
                ('y', pa.float64()),
            ] + [
                (f'band_{i+1}', pa.float32()) 
                for i in range(src.count)
            ])
            
            # 流式写入Arrow IPC格式
            with pa.ipc.RecordBatchFileWriter(output_path, schema) as writer:
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
        
        logger.info(f"Converted {tif_path} to Arrow IPC: {output_path}")
    
    def _merge_hemispheres_to_arrow(self, west_file: Path, east_file: Path, output_path: Path):
        """合并半球文件到Arrow IPC - 窗口化处理避免内存爆炸"""
        # 先转换为临时Arrow文件
        temp_west = output_path.with_suffix('.west.arrow')
        temp_east = output_path.with_suffix('.east.arrow')
        
        self._convert_tif_to_arrow(west_file, temp_west)
        self._convert_tif_to_arrow(east_file, temp_east)
        
        # 窗口化合并 - 避免一次性加载全部数据
        with pa.ipc.open_file(temp_west) as west_reader:
            with pa.ipc.open_file(temp_east) as east_reader:
                # 获取schema（两个文件应该相同）
                schema = west_reader.schema
                
                # 流式写入合并结果
                with pa.ipc.RecordBatchFileWriter(output_path, schema) as writer:
                    # 处理西半球数据 - 批次读取
                    for i in range(west_reader.num_record_batches):
                        batch = west_reader.get_batch(i)
                        # 调整经度（如果需要）
                        df = batch.to_pandas()
                        if 'longitude' in df.columns and (df['longitude'] > 180).any():
                            df.loc[df['longitude'] > 180, 'longitude'] -= 360
                        elif 'x' in df.columns and (df['x'] > 180).any():
                            df.loc[df['x'] > 180, 'x'] -= 360
                        
                        # 写入批次
                        writer.write_batch(pa.RecordBatch.from_pandas(df, schema=schema))
                    
                    # 处理东半球数据 - 批次读取
                    for i in range(east_reader.num_record_batches):
                        batch = east_reader.get_batch(i)
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
        
        logger.info(f"Merged hemispheres to Arrow IPC: {output_path}")
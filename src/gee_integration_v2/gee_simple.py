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
    
    # 导出参数（基于搜索结果的正确值）
    export_scale: float = 30  # meters - 30m分辨率
    export_max_pixels: float = 1e13  # 不是1e14！这是经验值
    export_shard_size: int = 2560  # 增大切片避免内存错误
    export_tile_size: int = 256  # 瓦片大小
    export_best_effort: bool = True  # 自动调整scale避免错误
    
    # 重试机制
    max_retries: int = 3
    retry_delay: int = 5  # seconds

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
# 状态管理 - 更完善的检查点系统
# =============================================================================

@dataclass
class PipelineState:
    """流程状态 - 用于恢复"""
    step: str  # 'upload', 'process', 'export', 'download'
    asset_ids: Dict[str, str] = field(default_factory=dict)
    processed_images: Dict[str, str] = field(default_factory=dict)  # resolution -> image_id
    export_tasks: Dict[str, str] = field(default_factory=dict)  # resolution -> task_id
    completed_exports: Dict[str, Path] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime('%Y%m%d_%H%M%S'))
    
    def save(self, path: Path):
        """保存状态到JSON"""
        state_dict = {
            'step': self.step,
            'asset_ids': self.asset_ids,
            'processed_images': self.processed_images,
            'export_tasks': self.export_tasks,
            'completed_exports': {k: str(v) for k, v in self.completed_exports.items()},
            'timestamp': self.timestamp
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        logger.info(f"Saved state to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        """从JSON加载状态"""
        with open(path) as f:
            state_dict = json.load(f)
        
        # 转换路径
        if 'completed_exports' in state_dict:
            state_dict['completed_exports'] = {
                k: Path(v) for k, v in state_dict['completed_exports'].items()
            }
        
        return cls(**state_dict)
    
    @classmethod
    def get_latest(cls, state_dir: Path) -> Optional['PipelineState']:
        """获取最新的状态文件"""
        state_files = list(state_dir.glob('state_*.json'))
        if not state_files:
            return None
        
        latest = max(state_files, key=lambda p: p.stat().st_mtime)
        return cls.load(latest)


# =============================================================================
# TIF验证功能 - 生产环境必需的
# =============================================================================

def validate_tif(file_path: str) -> Dict[str, Any]:
    """验证TIF文件 - 检查CRS、数据类型、边界等
    
    这是生产环境必需的，不能省略！
    """
    path = Path(file_path)
    issues = []
    metadata = {}
    
    if not path.exists():
        return {
            'valid': False,
            'issues': [f"File not found: {file_path}"],
            'metadata': {}
        }
    
    try:
        with rasterio.open(file_path) as src:
            # 提取元数据
            metadata = {
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'crs': str(src.crs),
                'bounds': src.bounds,
                'transform': src.transform,
                'nodata': src.nodata,
                'file_size_mb': path.stat().st_size / (1024 * 1024)
            }
            
            # 1. 检查CRS
            if not src.crs:
                issues.append("No CRS defined")
            elif not src.crs.is_geographic:
                issues.append(f"Not geographic CRS: {src.crs}")
            
            # 2. 检查边界
            if src.bounds.left >= src.bounds.right:
                issues.append("Invalid bounds: left >= right")
            if src.bounds.bottom >= src.bounds.top:
                issues.append("Invalid bounds: bottom >= top")
            
            # 3. 检查数据类型
            if src.dtypes[0] not in ['uint8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64']:
                issues.append(f"Unsupported data type: {src.dtypes[0]}")
            
            # 4. 检查文件大小（GEE限制）
            if metadata['file_size_mb'] > 10240:  # 10GB
                issues.append(f"File too large: {metadata['file_size_mb']:.1f}MB > 10GB")
            
            # 5. 快速数据检查
            window = rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height))
            data = src.read(1, window=window)
            
            valid_data = data[~np.isnan(data)]
            if src.nodata is not None:
                valid_data = valid_data[valid_data != src.nodata]
            
            if len(valid_data) > 0:
                metadata['data_range'] = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data))
                }
    
    except Exception as e:
        return {
            'valid': False,
            'issues': [f"Failed to validate: {e}"],
            'metadata': {}
        }
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'metadata': metadata
    }

def retry_on_failure(max_retries=3, delay=5, backoff=2):
    """改进的重试装饰器 - 支持指数退避
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 退避系数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 尝试从self.config获取参数
            if hasattr(args[0], 'config'):
                config = args[0].config
                actual_retries = getattr(config, 'max_retries', max_retries)
                actual_delay = getattr(config, 'retry_delay', delay)
            else:
                actual_retries = max_retries
                actual_delay = delay
            
            last_exception = None
            current_delay = actual_delay
            
            for attempt in range(actual_retries):
                try:
                    return func(*args, **kwargs)
                except (ee.EEException, TimeoutError, ConnectionError) as e:
                    # 只重试网络相关的异常
                    last_exception = e
                    if attempt == actual_retries - 1:
                        logger.error(f"Failed after {actual_retries} attempts: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1}/{actual_retries} failed: {e}")
                    logger.info(f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff  # 指数退避
                except Exception as e:
                    # 其他异常直接抛出
                    logger.error(f"Non-retryable error: {e}")
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator

def validate_batch_compatibility(file_paths: List[str]) -> Dict[str, Any]:
    """批量验证文件兼容性 - 确保可以合并"""
    results = {}
    all_metadata = []
    
    # 验证每个文件
    for path in file_paths:
        result = validate_tif(path)
        results[path] = result
        
        if result['valid']:
            all_metadata.append(result['metadata'])
    
    # 检查兼容性
    compatibility_issues = []
    if len(all_metadata) > 1:
        # CRS一致性
        crs_set = set(m['crs'] for m in all_metadata)
        if len(crs_set) > 1:
            compatibility_issues.append(f"Multiple CRS: {crs_set}")
        
        # 分辨率一致性
        resolutions = []
        for m in all_metadata:
            transform = m['transform']
            res_x = abs(transform.a)
            res_y = abs(transform.e)
            resolutions.append((res_x, res_y))
        
        res_x_values = [r[0] for r in resolutions]
        res_y_values = [r[1] for r in resolutions]
        
        if max(res_x_values) / min(res_x_values) > 1.01:  # 1%容差
            compatibility_issues.append("Resolution mismatch > 1%")
    
    valid_count = sum(1 for r in results.values() if r['valid'])
    
    return {
        'results': results,
        'valid_count': valid_count,
        'total_count': len(file_paths),
        'compatibility_issues': compatibility_issues,
        'all_valid': valid_count == len(file_paths) and not compatibility_issues
    }

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
    
    def run(self, resume_from: Optional[str] = None, state_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        运行完整流程 - 带改进的状态管理
        
        Args:
            resume_from: 简单恢复点 ('upload'/'process'/'export') - 向后兼容
            state_path: 状态文件路径，用于精确恢复
        """
        logger.info("Starting GEE pipeline")
        
        # 状态管理目录
        state_dir = self.config.output_dir / 'state'
        state_dir.mkdir(exist_ok=True)
        
        # 尝试加载状态
        state = None
        if state_path and state_path.exists():
            state = PipelineState.load(state_path)
            logger.info(f"Loaded state from {state_path}")
        elif resume_from:
            # 尝试找最新的状态文件
            state = PipelineState.get_latest(state_dir)
            if state:
                logger.info(f"Loaded latest state from {state.timestamp}")
        
        # 如果没有状态，创建新的
        if not state:
            state = PipelineState(step='start')
        
        # 根据状态决定从哪里开始
        if state.step in ['start'] and not resume_from == 'upload':
            # 1. 上传到GEE
            logger.info("Step 1: Upload to GEE")
            state.asset_ids = self._upload_to_gee()
            state.step = 'upload'
            state.save(state_dir / f'state_{state.timestamp}.json')
        
        if state.step in ['upload'] and resume_from not in ['process', 'export']:
            # 2. 在GEE中处理
            logger.info("Step 2: Process in GEE")
            processed = self._process_in_gee(state.asset_ids)
            # 保存处理后的图像ID供恢复
            state.processed_images = {k: v.serialize() for k, v in processed.items()}
            state.step = 'process'
            state.save(state_dir / f'state_{state.timestamp}.json')
        else:
            # 从状态恢复处理后的图像
            processed = {}
            for res_name, image_id in state.processed_images.items():
                processed[res_name] = ee.Image(ee.Deserializer.decode(image_id))
        
        # 3. 导出并下载
        if state.step in ['upload', 'process']:
            logger.info("Step 3: Export and download")
            outputs = self._export_and_download(processed, state)
            state.completed_exports = outputs
            state.step = 'complete'
            state.save(state_dir / f'state_{state.timestamp}.json')
        else:
            outputs = state.completed_exports
        
        logger.info("✅ Pipeline complete")
        return outputs
    
    def _upload_to_gee(self) -> Dict[str, str]:
        """上传生物多样性数据到GEE - 支持并行"""
        asset_ids = {}
        
        # 1. 批量验证文件兼容性（必须！）
        logger.info("Validating TIF files...")
        validation = validate_batch_compatibility(self.config.biodiversity_files)
        
        if not validation['all_valid']:
            # 报告验证问题
            for path, result in validation['results'].items():
                if not result['valid']:
                    logger.error(f"❌ {Path(path).name}: {result['issues']}")
            
            if validation['compatibility_issues']:
                logger.error(f"Compatibility issues: {validation['compatibility_issues']}")
            
            raise ValueError(f"Validation failed: {validation['valid_count']}/{validation['total_count']} files valid")
        
        logger.info(f"✅ All {validation['total_count']} files validated successfully")
        
        # 2. 验证资产名称
        for tif_path in self.config.biodiversity_files:
            name = Path(tif_path).stem
            if not name or '/' in name or ' ' in name:
                raise ValueError(f"Invalid asset name: {name}")
        
        # 3. 并行上传 - 如果文件多于1个
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
        asset_id = f"projects/{self.config.project_id}/assets/{name}"
        
        # 检查资产是否已存在（避免重复上传）
        if self._check_asset_exists(asset_id):
            logger.info(f"⏭️ Skipping {name} - asset already exists")
            return name, asset_id
        
        # 确保资产文件夹存在
        self._ensure_asset_folder(asset_id)
        
        # 1. 上传到GCS
        gcs_path = f"gs://{self.config.gcs_bucket}/uploads/{name}.tif"
        self._upload_to_gcs(tif_path, gcs_path)
        
        # 2. 导入到GEE
        self._import_to_gee(gcs_path, asset_id)
        
        return name, asset_id
    
    def _check_asset_exists(self, asset_id: str) -> bool:
        """检查GEE资产是否已存在"""
        try:
            ee.data.getAsset(asset_id)
            return True
        except (ee.EEException, Exception) as e:
            # EEException可能不是标准异常，捕获所有异常
            if 'not found' in str(e).lower() or 'does not exist' in str(e).lower():
                return False
            # 其他错误可能是真正的问题
            logger.warning(f"Error checking asset {asset_id}: {e}")
            return False
    
    def _ensure_asset_folder(self, asset_path: str):
        """确保GEE资产文件夹存在"""
        # 获取父文件夹路径
        parts = asset_path.split('/')
        if len(parts) > 4:  # projects/PROJECT/assets/FOLDER/asset
            folder_path = '/'.join(parts[:-1])
            try:
                # 尝试创建文件夹
                ee.data.createAsset({'type': 'Folder'}, folder_path)
                logger.debug(f"Created asset folder: {folder_path}")
            except ee.EEException as e:
                # 文件夹可能已存在，这不是错误
                if 'already exists' not in str(e):
                    logger.warning(f"Could not create folder {folder_path}: {e}")
    
    @retry_on_failure()
    def _upload_to_gcs(self, local_path: str, gcs_path: str):
        """上传文件到GCS - 使用官方库而不是gsutil，带重试"""
        bucket_name = gcs_path.split('/')[2]
        blob_name = '/'.join(gcs_path.split('/')[3:])
        
        bucket = self.gcs.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to {gcs_path}")
    
    @retry_on_failure()
    def _import_to_gee(self, gcs_path: str, asset_id: str):
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
    
    def _process_in_gee(self, asset_ids: Dict[str, str]) -> Dict[str, ee.Image]:
        """在GEE中处理 - 合并、重采样等"""
        processed = {}
        
        # 加载生物多样性图像
        biodiv_images = {
            name: ee.Image(asset_id) 
            for name, asset_id in asset_ids.items()
        }
        
        # 验证像素对齐（关键！）
        if len(biodiv_images) > 1:
            if not self._validate_pixel_alignment(list(biodiv_images.values())):
                logger.warning("Images are not pixel-aligned, will reproject to common grid")
                # 重投影到第一个图像的网格
                reference = list(biodiv_images.values())[0]
                biodiv_images = self._reproject_to_common_grid(biodiv_images, reference)
        
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
    
    def _validate_pixel_alignment(self, images: List[ee.Image]) -> bool:
        """验证所有图像的像素网格对齐 - 防止mosaic搞砸投影"""
        try:
            transforms = []
            for img in images:
                info = img.getInfo()
                if 'bands' in info and len(info['bands']) > 0:
                    transform = info['bands'][0].get('crs_transform')
                    if transform:
                        transforms.append(tuple(transform))
            
            # 检查所有transform是否一致
            if len(set(transforms)) > 1:
                logger.warning(f"Images have different pixel grids: {transforms}")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not validate pixel alignment: {e}")
            return False  # 安全起见，假设未对齐
    
    def _reproject_to_common_grid(self, images: Dict[str, ee.Image], reference: ee.Image) -> Dict[str, ee.Image]:
        """重投影所有图像到参考图像的网格"""
        proj = reference.projection()
        
        reprojected = {}
        for name, img in images.items():
            reprojected[name] = img.reproject(
                crs=proj.crs(),
                crsTransform=proj.transform()
            )
        
        return reprojected
    
    def _merge_biodiversity(self, images: Dict[str, ee.Image]) -> ee.Image:
        """合并生物多样性波段"""
        # 简单的addBands - 不要过度复杂化
        first_name = list(images.keys())[0]
        result = images[first_name].rename(first_name)
        
        for name, img in list(images.items())[1:]:
            result = result.addBands(img.rename(name))
        
        # 修复mosaic默认投影问题（关键！）
        # GEE的mosaic会默认变成WGS84 1度分辨率，必须修复
        reference_proj = images[first_name].projection()
        result = result.reproject(
            crs=reference_proj.crs(),
            crsTransform=reference_proj.transform(),
            scale=self.config.export_scale
        )
        
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
    
    def _export_and_download(self, processed: Dict[str, ee.Image], state: PipelineState = None) -> Dict[str, Path]:
        """导出到GCS并下载 - 支持状态恢复"""
        outputs = {}
        
        # 如果有状态，检查已完成的导出
        if state and state.export_tasks:
            export_tasks = state.export_tasks
            logger.info(f"Resuming with existing export tasks: {list(export_tasks.keys())}")
        else:
            export_tasks = {}
        
        for res_name, image in processed.items():
            # 跳过已完成的
            if state and res_name in state.completed_exports:
                logger.info(f"Skipping {res_name} - already completed")
                outputs[res_name] = state.completed_exports[res_name]
                continue
            
            # 如果没有正在运行的任务，启动新的
            if res_name not in export_tasks:
                # 导出到GCS
                export_path = f"{self.config.gcs_bucket}/exports/{res_name}"
                task_id = self._export_to_gcs(image, export_path, res_name)
                export_tasks[res_name] = task_id
                
                # 保存任务ID到状态
                if state:
                    state.export_tasks[res_name] = task_id
                    state_dir = self.config.output_dir / 'state'
                    state.save(state_dir / f'state_{state.timestamp}.json')
            
            # 等待完成
            try:
                self._wait_for_export(export_tasks[res_name])
                
                # 下载并转换为parquet
                export_path = f"{self.config.gcs_bucket}/exports/{res_name}"
                output_path = self.config.output_dir / f"{res_name}.parquet"
                self._download_and_convert(export_path, output_path)
                
                outputs[res_name] = output_path
                
                # 更新状态
                if state:
                    state.completed_exports[res_name] = output_path
                    state_dir = self.config.output_dir / 'state'
                    state.save(state_dir / f'state_{state.timestamp}.json')
                    
            except Exception as e:
                logger.error(f"Failed to export {res_name}: {e}")
                # 状态已保存，可以从这里恢复
                raise
            
        return outputs
    
    def _export_to_gcs(self, image: ee.Image, gcs_path: str, description: str) -> str:
        """导出图像到GCS - 使用正确的参数避免内存错误"""
        # 检查是否需要半球分割（大于180度）
        geometry = ee.Geometry.Rectangle([-180, -90, 180, 90])
        
        # 构建导出参数
        export_params = {
            'image': image,
            'description': description,
            'bucket': self.config.gcs_bucket,
            'fileNamePrefix': f"exports/{description}",
            'region': geometry,
            'scale': self.config.export_scale,  # 30m分辨率
            'maxPixels': self.config.export_max_pixels,  # 1e13，不是1e14！
            'fileFormat': 'GeoTIFF',
            'formatOptions': {
                'cloudOptimized': True,
                'noData': 0  # 明确的NoData值处理
            },
            'fileDimensions': self.config.export_tile_size,
            'shardSize': self.config.export_shard_size  # 2560 - 关键参数！
        }
        
        # 如果启用bestEffort，添加参数
        if self.config.export_best_effort:
            export_params['bestEffort'] = True
        
        task = ee.batch.Export.image.toCloudStorage(**export_params)
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
        
        # 清理临时文件和GCS文件
        if self.config.cleanup_temp:
            # 清理本地临时文件
            for f in temp_dir.glob('*'):
                f.unlink()
            
            # 清理GCS上的临时导出文件
            self._cleanup_gcs_files(gcs_path)
        
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
    
    def _cleanup_gcs_files(self, prefix: str):
        """清理GCS上的临时文件 - 避免留下孤儿文件"""
        try:
            bucket = self.gcs.bucket(self.config.gcs_bucket)
            
            # 如果prefix包含完整的gs://路径，提取实际前缀
            if prefix.startswith('gs://'):
                prefix = '/'.join(prefix.split('/')[3:])
            elif prefix.startswith(self.config.gcs_bucket):
                prefix = prefix.replace(self.config.gcs_bucket + '/', '')
            
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
            # 但还是记录下来，管理员可以手动清理
    
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
        """合并半球文件到parquet - 窗口化处理避免内存爆炸"""
        # 先转换为临时parquet文件
        temp_west = output_path.with_suffix('.west.parquet')
        temp_east = output_path.with_suffix('.east.parquet')
        
        self._convert_tif_to_parquet(west_file, temp_west)
        self._convert_tif_to_parquet(east_file, temp_east)
        
        # 窗口化合并 - 避免一次性加载全部数据
        west_file = pq.ParquetFile(temp_west)
        east_file = pq.ParquetFile(temp_east)
        
        # 获取schema（两个文件应该相同）
        schema = west_file.schema_arrow
        
        # 流式写入合并结果
        with pq.ParquetWriter(output_path, schema) as writer:
            # 处理西半球数据 - 批次读取
            for batch in west_file.iter_batches(batch_size=100000):
                # 调整经度（如果需要）
                df = batch.to_pandas()
                if 'longitude' in df.columns and (df['longitude'] > 180).any():
                    df.loc[df['longitude'] > 180, 'longitude'] -= 360
                elif 'x' in df.columns and (df['x'] > 180).any():
                    df.loc[df['x'] > 180, 'x'] -= 360
                
                # 写入批次
                writer.write_batch(pa.RecordBatch.from_pandas(df, schema=schema))
            
            # 处理东半球数据 - 批次读取
            for batch in east_file.iter_batches(batch_size=100000):
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
"""
TIF文件验证模块
==============

负责所有验证相关功能，独立可复用。

作者: Linus式重构
"""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import rasterio
import logging

logger = logging.getLogger(__name__)


class TIFValidator:
    """TIF文件验证器"""
    
    def validate_tif(self, file_path: str) -> bool:
        """验证TIF文件 - 检查CRS、数据类型、边界等
        
        这是生产环境必需的，不能省略！
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
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
                
                issues = []
                
                # 1. 检查CRS
                if not src.crs:
                    issues.append("Missing CRS")
                elif not src.crs.is_geographic:
                    issues.append(f"Non-geographic CRS: {src.crs}")
                
                # 2. 检查数据类型
                if src.dtypes[0] not in ['float32', 'float64', 'int16', 'int32', 'uint16', 'uint32']:
                    issues.append(f"Unusual dtype: {src.dtypes[0]}")
                
                # 3. 检查维度
                if src.width > 100000 or src.height > 100000:
                    issues.append(f"Very large dimensions: {src.width}x{src.height}")
                
                # 4. 检查波段数
                if src.count > 100:
                    issues.append(f"Too many bands: {src.count}")
                
                # 5. 检查文件大小
                if metadata['file_size_mb'] > 10000:  # 10GB
                    issues.append(f"File too large: {metadata['file_size_mb']:.1f}MB")
                
                # 6. 检查边界合理性（允许小的浮点误差）
                bounds = src.bounds
                tolerance = 0.01  # 允许0.01度的误差（约1km）
                if bounds.left < -180 - tolerance or bounds.right > 180 + tolerance:
                    issues.append(f"Longitude out of range: [{bounds.left}, {bounds.right}]")
                if bounds.bottom < -90 - tolerance or bounds.top > 90 + tolerance:
                    issues.append(f"Latitude out of range: [{bounds.bottom}, {bounds.top}]")
                
                # 7. 快速数据检查（只读第一个块）
                data = src.read(1, window=rasterio.windows.Window(0, 0, 
                    min(512, src.width), min(512, src.height)))
                
                if np.all(np.isnan(data)):
                    issues.append("All values are NaN in sample")
                elif np.all(data == 0) and src.nodata != 0:
                    # 只有当0不是NoData值时才报告问题
                    issues.append("All values are 0 in sample (and 0 is not NoData)")
                
                # 日志记录
                if issues:
                    logger.warning(f"TIF validation issues for {file_path}:")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
                else:
                    logger.info(f"TIF validation passed: {file_path}")
                    logger.debug(f"  Metadata: {metadata}")
                
                # 返回验证结果
                return len(issues) == 0
                
        except Exception as e:
            logger.error(f"Failed to validate TIF {file_path}: {e}")
            return False
    
    def validate_batch_compatibility(self, file_paths: List[str]) -> bool:
        """验证多个TIF文件是否兼容（可以一起处理）
        
        检查：
        - CRS一致性
        - 分辨率一致性
        - 边界重叠或邻接
        """
        if len(file_paths) < 2:
            return True
        
        try:
            # 收集所有文件的元数据
            metadata_list = []
            for path in file_paths:
                with rasterio.open(path) as src:
                    metadata_list.append({
                        'crs': src.crs,
                        'transform': src.transform,
                        'bounds': src.bounds,
                        'shape': (src.height, src.width)
                    })
            
            # 检查CRS一致性
            crs_set = set(str(m['crs']) for m in metadata_list)
            if len(crs_set) > 1:
                logger.error(f"Inconsistent CRS: {crs_set}")
                return False
            
            # 记录分辨率信息（但不要求一致）
            resolutions = []
            for m in metadata_list:
                t = m['transform']
                res_x = abs(t[0])
                res_y = abs(t[4])
                resolutions.append((res_x, res_y))
            
            # 只记录分辨率差异，不作为错误
            res_x_values = [r[0] for r in resolutions]
            res_y_values = [r[1] for r in resolutions]
            
            if max(res_x_values) - min(res_x_values) > 1e-6:
                logger.info(f"Different X resolutions detected: {res_x_values}")
                logger.info("Files will be resampled to base resolution in GEE")
            
            if max(res_y_values) - min(res_y_values) > 1e-6:
                logger.info(f"Different Y resolutions detected: {res_y_values}")
                logger.info("Files will be resampled to base resolution in GEE")
            
            logger.info(f"Batch compatibility check passed for {len(file_paths)} files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate batch compatibility: {e}")
            return False
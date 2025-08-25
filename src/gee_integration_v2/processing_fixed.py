"""
数据处理模块（修复版）
====================

修复了原版本的严重问题：
1. 先对齐到基础分辨率，再合并
2. 可配置的基础分辨率
3. 正确的处理顺序

作者: Linus式重构（修复版）
"""

import ee
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理类 - 修复版"""
    
    def __init__(self, config):
        """
        Args:
            config: 包含base_resolution, climate_bands, include_elevation等配置
        """
        self.config = config
        # 使用配置的基础分辨率，而不是硬编码
        self.base_resolution = getattr(config, 'base_resolution', 0.016666666666667)  # 默认1.85km
    
    def align_to_base_resolution(self, image: ee.Image, band_name: str = None) -> ee.Image:
        """对齐图像到基础分辨率网格 - 这是关键步骤！
        
        所有数据必须先对齐到同一网格，才能正确合并。
        """
        # 转换度到米
        base_scale = self.base_resolution * 111320  # 赤道处的近似值
        
        # 创建目标投影
        target_projection = ee.Projection('EPSG:4326').atScale(base_scale)
        
        # Reproject到基础分辨率
        aligned = image.reproject(
            crs=target_projection,
            scale=base_scale
        )
        
        # 转换为Float32确保类型一致
        aligned = aligned.toFloat()
        
        # 设置波段名
        if band_name:
            aligned = aligned.rename(band_name)
        
        return aligned
    
    def merge_biodiversity(self, images: Dict[str, ee.Image]) -> ee.Image:
        """合并生物多样性波段
        
        重要：输入图像必须已经对齐到同一分辨率！
        """
        # 先对齐所有图像到基础分辨率
        aligned_images = {}
        for name, img in images.items():
            aligned_images[name] = self.align_to_base_resolution(img, name)
            logger.info(f"Aligned {name} to base resolution {self.base_resolution}")
        
        # 合并对齐后的图像
        first_name = list(aligned_images.keys())[0]
        result = aligned_images[first_name]
        
        for name, img in list(aligned_images.items())[1:]:
            result = result.addBands(img)
        
        return result
    
    def add_climate(self, image: ee.Image) -> ee.Image:
        """添加WorldClim气候数据
        
        WorldClim原生分辨率约1km，会自动resample到image的分辨率
        """
        if not self.config.climate_bands:
            return image
        
        climate = ee.ImageCollection("WORLDCLIM/V1/BIO").select(self.config.climate_bands)
        climate_image = climate.toBands()
        
        # 重命名波段
        band_names = [f"WORLDCLIM_V1_BIO_{band}" for band in self.config.climate_bands]
        climate_image = climate_image.rename(band_names)
        
        # 对齐到基础分辨率
        climate_aligned = self.align_to_base_resolution(climate_image)
        
        return image.addBands(climate_aligned)
    
    def add_elevation(self, image: ee.Image) -> ee.Image:
        """添加MERIT DEM高程数据
        
        MERIT DEM原生分辨率90m，会resample到image的分辨率
        """
        if not self.config.include_elevation:
            return image
        
        elevation = ee.Image("MERIT/DEM/v1_0_3").select('dem').rename('elevation')
        
        # 对齐到基础分辨率
        elevation_aligned = self.align_to_base_resolution(elevation, 'elevation')
        
        return image.addBands(elevation_aligned)
    
    def recursive_downsample(self, image: ee.Image, target_resolution: float) -> ee.Image:
        """
        递归2x降采样 - 绕过GEE的256像素聚合限制
        
        重要修复：
        1. 从实际的base_resolution开始，不是硬编码的值
        2. 只能降采样（target必须 >= base）
        """
        current_res = self.base_resolution
        
        # 如果目标分辨率比基础分辨率高，返回原图（不能提高分辨率）
        if target_resolution < current_res:
            logger.warning(f"Cannot upsample from {current_res} to {target_resolution}, returning original")
            return image
        
        # 递归2x降采样
        result = image
        while current_res < target_resolution:
            next_res = min(current_res * 2, target_resolution)
            
            # 计算新的scale
            new_scale = next_res * 111320
            
            # 使用reduceResolution降采样
            result = result.reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=1024  # 安全值，远低于256限制
            ).reproject(
                crs='EPSG:4326',
                scale=new_scale
            )
            
            current_res = next_res
            logger.debug(f"Downsampled to {current_res} degrees ({new_scale}m)")
        
        return result
    
    def process_images(self, biodiv_images: Dict[str, ee.Image]) -> Dict[str, ee.Image]:
        """完整的处理流程 - 修复版
        
        正确的顺序：
        1. 对齐所有数据到基础分辨率
        2. 合并生物多样性
        3. 添加气候和高程（也对齐）
        4. 生成多分辨率版本（只能降采样）
        """
        processed = {}
        
        # 步骤1：合并生物多样性（内部会对齐）
        merged = self.merge_biodiversity(biodiv_images)
        logger.info(f"Merged {len(biodiv_images)} biodiversity images at base resolution")
        
        # 步骤2：添加气候数据（已对齐）
        merged = self.add_climate(merged)
        
        # 步骤3：添加高程数据（已对齐）
        merged = self.add_elevation(merged)
        
        # 步骤4：生成多分辨率版本
        for res_name, resolution in self.config.resolutions.items():
            if resolution >= self.base_resolution:
                # 只处理降采样
                processed[res_name] = self.recursive_downsample(merged, resolution)
                logger.info(f"Generated {res_name}: {resolution} degrees")
            else:
                logger.warning(f"Skipping {res_name}: {resolution} < base {self.base_resolution}")
        
        return processed
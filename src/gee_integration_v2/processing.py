"""
数据处理模块
===========

处理GEE中的数据操作，包括对齐、合并、重采样、添加气候数据等。

关键修复：所有数据必须先对齐到base_resolution再合并！

作者: Linus式重构
"""

import ee
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理类"""
    
    def __init__(self, config):
        """
        Args:
            config: 包含base_resolution, climate_bands, include_elevation等配置
        """
        self.config = config
        # 基础分辨率 - 所有数据对齐到这里
        self.base_resolution = config.base_resolution
        self.base_scale = self.base_resolution * 111320  # 度转米（赤道）
    
    def align_to_base(self, image: ee.Image, band_name: str = None) -> ee.Image:
        """对齐图像到基础分辨率 - 核心操作！
        
        这确保所有数据在同一网格上，才能正确合并。
        """
        # 创建统一的投影
        projection = ee.Projection('EPSG:4326').atScale(self.base_scale)
        
        # Reproject到基础分辨率
        aligned = image.reproject(
            crs=projection,
            scale=self.base_scale
        )
        
        # 转换为Float32（统一数据类型）
        aligned = aligned.toFloat()
        
        # 重命名波段
        if band_name:
            aligned = aligned.rename(band_name)
        
        return aligned
    
    def downsample_2x(self, image: ee.Image) -> ee.Image:
        """降采样2倍 - 复用的核心操作
        
        这是递归降采样的基础操作。
        """
        # 使用reduceResolution降采样
        downsampled = image.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=1024  # 安全值，远低于256限制
        )
        
        # 获取当前投影并放大2倍
        current_projection = image.projection()
        new_scale = current_projection.nominalScale().multiply(2)
        
        # Reproject到新分辨率
        return downsampled.reproject(
            crs='EPSG:4326',
            scale=new_scale
        )
    
    def recursive_downsample(self, image: ee.Image, factor: int) -> ee.Image:
        """递归降采样到指定倍数
        
        Args:
            image: 输入图像（已对齐到base_resolution）
            factor: 降采样倍数，必须是2的幂（2, 4, 8...）
        
        Returns:
            降采样后的图像
        """
        # 验证factor是2的幂
        if factor <= 0 or (factor & (factor - 1)) != 0:
            raise ValueError(f"Factor must be power of 2, got {factor}")
        
        if factor == 1:
            return image
        
        # 计算需要多少次2x降采样
        steps = 0
        temp = factor
        while temp > 1:
            temp //= 2
            steps += 1
        
        # 递归应用2x降采样
        result = image
        for _ in range(steps):
            result = self.downsample_2x(result)
            logger.debug(f"Applied 2x downsample, step {_+1}/{steps}")
        
        return result
    
    def merge_biodiversity(self, images: Dict[str, ee.Image]) -> ee.Image:
        """合并生物多样性波段
        
        关键：所有图像先对齐到base_resolution！
        """
        if not images:
            raise ValueError("No biodiversity images provided")
        
        # 步骤1：对齐所有图像到基础分辨率
        aligned_images = {}
        for name, img in images.items():
            aligned = self.align_to_base(img, name)
            aligned_images[name] = aligned
            logger.info(f"Aligned {name} to base resolution {self.base_resolution} degrees")
        
        # 步骤2：合并对齐后的图像
        first_name = list(aligned_images.keys())[0]
        result = aligned_images[first_name]
        
        for name, img in list(aligned_images.items())[1:]:
            result = result.addBands(img)
        
        logger.info(f"Merged {len(images)} biodiversity bands")
        return result
    
    def add_climate(self, image: ee.Image) -> ee.Image:
        """添加WorldClim气候数据"""
        if not self.config.climate_bands:
            return image
        
        # 获取气候数据
        climate = ee.ImageCollection("WORLDCLIM/V1/BIO").select(self.config.climate_bands)
        climate_image = climate.toBands()
        
        # 重命名波段
        band_names = [f"WORLDCLIM_V1_BIO_{band}" for band in self.config.climate_bands]
        climate_image = climate_image.rename(band_names)
        
        # 对齐到基础分辨率（WorldClim原生~1km）
        climate_aligned = self.align_to_base(climate_image)
        
        logger.info(f"Added {len(self.config.climate_bands)} climate bands")
        return image.addBands(climate_aligned)
    
    def add_elevation(self, image: ee.Image) -> ee.Image:
        """添加MERIT DEM高程数据"""
        if not self.config.include_elevation:
            return image
        
        # 获取高程数据
        elevation = ee.Image("MERIT/DEM/v1_0_3").select('dem').rename('elevation')
        
        # 对齐到基础分辨率（MERIT DEM原生90m）
        elevation_aligned = self.align_to_base(elevation, 'elevation')
        
        logger.info("Added elevation band")
        return image.addBands(elevation_aligned)
    
    def process_images(self, biodiv_images: Dict[str, ee.Image]) -> Dict[str, ee.Image]:
        """完整的处理流程
        
        正确的顺序：
        1. 对齐所有生物多样性数据到base_resolution
        2. 合并生物多样性波段
        3. 添加气候数据（自动对齐）
        4. 添加高程数据（自动对齐）
        5. 生成多分辨率版本（2x, 4x递归降采样）
        """
        processed = {}
        
        # 步骤1-2：合并生物多样性（内部会对齐）
        merged = self.merge_biodiversity(biodiv_images)
        
        # 步骤3：添加气候数据
        merged = self.add_climate(merged)
        
        # 步骤4：添加高程数据
        merged = self.add_elevation(merged)
        
        # 步骤5：生成多分辨率版本
        for res_name, resolution in self.config.resolutions.items():
            # 解析分辨率名称（如 '1x', '2x', '4x'）
            if res_name.endswith('x'):
                try:
                    factor = int(res_name[:-1])
                    
                    # 验证是2的幂
                    if factor > 0 and (factor & (factor - 1)) == 0:
                        if factor == 1:
                            # 基础分辨率，直接使用
                            processed[res_name] = merged
                            logger.info(f"Generated {res_name}: base resolution")
                        else:
                            # 递归降采样
                            processed[res_name] = self.recursive_downsample(merged, factor)
                            target_res = self.base_resolution * factor
                            logger.info(f"Generated {res_name}: {target_res:.9f} degrees")
                    else:
                        logger.warning(f"Skipping {res_name}: factor must be power of 2")
                        
                except ValueError:
                    logger.warning(f"Invalid resolution key: {res_name}")
        
        return processed
from src.abstractions import get_logger
"""
数据处理模块 - 简化版
====================

处理流程：
1. 处理NoData
2. 对齐到基准分辨率  
3. 合并生物多样性数据
4. 应用Ocean mask
5. 添加外部数据集

删除了多分辨率pyramid和降采样逻辑

作者: Linus式重构
"""

import ee
import logging
from typing import Dict

logger = get_logger(__name__)


class DataProcessor:
    """数据处理类 - 简化版，只处理单一分辨率"""
    
    def __init__(self, config):
        """
        Args:
            config: 包含base_resolution和external_datasets配置
        """
        self.config = config
        self.base_resolution = config.base_resolution
        self.base_scale = self.base_resolution * 111320  # 度转米（赤道）
    
    def process_images(self, biodiv_images: Dict[str, ee.Image]) -> Dict[str, ee.Image]:
        """处理生物多样性图像 - 简化流程
        
        Args:
            biodiv_images: {name: ee.Image}字典
        
        Returns:
            {'1x': 处理后的ee.Image} - 为了兼容现有接口
        """
        # Step 1: 处理每个数据集 - NoData + 对齐
        aligned_images = []
        
        for name, image in biodiv_images.items():
            # 获取配置
            dataset_config = self._get_dataset_config(name)
            nodata_value = dataset_config.get('nodata_value', None)
            
            # 处理NoData
            if nodata_value is not None:
                valid_mask = image.neq(nodata_value)
                image = image.updateMask(valid_mask)
                logger.info(f"Applied NoData mask for {name} (value={nodata_value})")
            
            # 对齐到基准分辨率（关键步骤！）
            aligned = image.reproject(
                crs='EPSG:4326',
                scale=self.base_scale
            ).toFloat().rename(name)
            
            aligned_images.append(aligned)
            logger.info(f"Aligned {name} to base resolution ({self.base_resolution}°)")
        
        # Step 2: 合并所有生物多样性数据
        merged = aligned_images[0]
        for img in aligned_images[1:]:
            merged = merged.addBands(img)
        logger.info(f"Merged {len(aligned_images)} biodiversity datasets")
        
        # Step 3: 应用Ocean mask和纬度限制
        masked = self.apply_mask(merged)
        
        # Step 4: 添加外部数据集（climate, elevation, PET等）
        final = self.add_external_datasets(masked, self.base_scale)
        
        logger.info(f"Processing complete: {final.bandNames().size()} bands")
        
        # 返回字典格式以兼容现有接口
        return {'1x': final}
    
    def apply_mask(self, image: ee.Image) -> ee.Image:
        """应用ocean mask和纬度限制
        
        重要：保留原始mask（包括NoData处理后的mask）！
        """
        if not getattr(self.config, 'mask_ocean', False):
            logger.info("Ocean masking disabled")
            return image
        
        # 获取原始mask（包括NoData mask）
        original_mask = image.mask()
        
        # 使用MODIS land/water mask
        # 0 = land, 1 = water
        water_mask = ee.Image("MODIS/006/MOD44W/2015_01_01").select('water_mask')
        
        # 创建陆地mask（水=1，所以陆地是水!=1）
        land_mask = water_mask.neq(1)
        
        # 应用纬度限制
        lat_min = getattr(self.config, 'lat_min', -60)
        lat_max = getattr(self.config, 'lat_max', 75)
        
        if lat_min is not None or lat_max is not None:
            logger.info(f"Applying latitude bounds: [{lat_min}°, {lat_max}°]")
            
            # 创建纬度图像
            lat_img = ee.Image.pixelLonLat().select('latitude')
            
            # 创建纬度mask
            lat_mask = lat_img.gte(lat_min).And(lat_img.lte(lat_max))
            
            # 组合所有masks：原始mask AND ocean mask AND 纬度mask
            final_mask = original_mask.And(land_mask).And(lat_mask)
        else:
            # 只组合原始mask和ocean mask
            final_mask = original_mask.And(land_mask)
        
        # 应用组合后的mask
        masked = image.updateMask(final_mask)
        
        logger.info("Applied ocean mask and latitude bounds")
        return masked
    
    def add_external_datasets(self, image: ee.Image, target_scale: float) -> ee.Image:
        """添加外部数据集（climate, elevation等）
        
        使用GEE的pyramid功能自动重采样到目标分辨率
        """
        if not hasattr(self.config, 'external_datasets'):
            return image
        
        for dataset in self.config.external_datasets:
            try:
                name = dataset.get('name', 'unknown')
                
                # 获取数据源
                if 'image' in dataset:
                    # 直接是Image（如WorldClim, MERIT DEM）
                    data = ee.Image(dataset['image'])
                elif 'collection' in dataset:
                    # 是ImageCollection（如TerraClimate）
                    collection = ee.ImageCollection(dataset['collection'])
                    
                    # 日期过滤（如果需要）
                    if 'filter_dates' in dataset:
                        start_date, end_date = dataset['filter_dates']
                        collection = collection.filterDate(start_date, end_date)
                        logger.info(f"Filtered {name} to date range: {start_date} to {end_date}")
                    
                    # 应用reducer转换为单个Image
                    reducer_type = dataset.get('reducer', 'mean')
                    if reducer_type == 'mean':
                        data = collection.mean()
                    elif reducer_type == 'median':
                        data = collection.median()
                    elif reducer_type == 'max':
                        data = collection.max()
                    elif reducer_type == 'min':
                        data = collection.min()
                    elif reducer_type == 'sum':
                        data = collection.sum()
                    else:
                        data = collection.first()
                else:
                    logger.warning(f"Dataset {name} missing 'image' or 'collection' field")
                    continue
                
                # 选择波段
                if 'bands' in dataset:
                    data = data.select(dataset['bands'])
                
                # 重命名波段
                if 'rename' in dataset:
                    data = data.rename(dataset['rename'])
                elif 'rename_pattern' in dataset:
                    pattern = dataset['rename_pattern']
                    band_names = [pattern.format(band=b) for b in dataset['bands']]
                    data = data.rename(band_names)
                
                # 关键：使用GEE的pyramid - reproject到目标scale
                # GEE会自动从pyramid中选择合适的级别！
                data_at_scale = data.reproject(
                    crs='EPSG:4326',
                    scale=target_scale
                )
                
                # 确保数据类型一致（统一为Float32）
                data_at_scale = data_at_scale.toFloat()
                
                # 添加到图像
                image = image.addBands(data_at_scale)
                
                logger.info(f"Added {name} at scale {target_scale}m using GEE pyramid")
                
            except Exception as e:
                logger.error(f"Failed to add external dataset {dataset.get('name', 'unknown')}: {e}")
                continue
        
        # 检查是否需要计算地形变量
        compute_terrain = False
        for dataset in self.config.external_datasets:
            if dataset.get('name') == 'elevation':
                compute_terrain = dataset.get('compute_terrain', True)  # 默认计算
                break
        
        if compute_terrain:
            try:
                elevation = image.select('elevation')
                
                # 计算所有地形变量
                slope = ee.Terrain.slope(elevation).toFloat().rename('slope')
                aspect = ee.Terrain.aspect(elevation).toFloat()
                
                # Aspect 转换为 cos/sin
                aspect_rad = aspect.multiply(3.14159265359/180)
                aspect_cos = aspect_rad.cos().toFloat().rename('aspect_cos')
                aspect_sin = aspect_rad.sin().toFloat().rename('aspect_sin')
                
                # TPI计算 - 简单3x3窗口版本
                mean_elev = elevation.focal_mean(radius=3, units='pixels', kernelType='square')
                tpi = elevation.subtract(mean_elev).toFloat().rename('tpi')
                
                # 一次性添加所有地形波段
                terrain_bands = ee.Image.cat([slope, aspect_cos, aspect_sin, tpi])
                image = image.addBands(terrain_bands)
                
                # 使用延迟计算的方式记录信息
                terrain_band_names = ['slope', 'aspect_cos', 'aspect_sin', 'tpi']
                logger.info(f"Added {len(terrain_band_names)} terrain variables: {', '.join(terrain_band_names)}")
                
            except Exception as e:
                logger.warning(f"Failed to compute terrain variables: {e}")
        
        return image
    
    def _get_dataset_config(self, name: str) -> dict:
        """获取数据集的配置（包括reducer、nodata、resampling等）"""
        # 先检查biodiversity_datasets
        if hasattr(self.config, 'biodiversity_datasets'):
            for dataset in self.config.biodiversity_datasets:
                if dataset.get('name') == name:
                    return dataset
        
        # 再检查external_datasets
        if hasattr(self.config, 'external_datasets'):
            for dataset in self.config.external_datasets:
                if dataset.get('name') == name:
                    return dataset
        
        # 默认配置
        return {
            'reducer': 'mean',
            'resampling_method': 'bilinear',
            'nodata_value': None
        }
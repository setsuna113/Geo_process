from src.abstractions import get_logger
"""
主流程编排器
===========

整合所有模块，提供简单的接口来执行完整的GEE数据处理流程。

作者: Linus式重构
"""

import ee
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .validation import TIFValidator
from .gee_ops import GEEOperations
from .processing import DataProcessor
from .export import ExportManager
from .state import PipelineState

logger = get_logger(__name__)


@dataclass
class GEEConfig:
    """简化的配置类 - 所有参数都有合理默认值"""
    
    # GEE认证
    service_account_key: str
    project_id: str
    gcs_bucket: str
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'GEEConfig':
        """从YAML文件加载配置"""
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        
        # 提取GEE配置
        gee_config = data.get('gee', {})
        auth = gee_config.get('auth', {})
        
        # 创建配置实例
        config = cls(
            service_account_key=auth.get('service_account_key'),
            project_id=auth.get('project_id'),
            gcs_bucket=gee_config.get('upload', {}).get('bucket'),
            output_dir=Path(data.get('output_dir', 'outputs')),
            state_dir=Path(data.get('state_dir', 'state')),
            base_resolution=data.get('base_resolution', 0.016666666666667),
            external_datasets=data.get('external_datasets', [])
        )
        
        # 添加生物多样性资产和reducers
        config.biodiversity_assets = data.get('biodiversity_assets', {})
        config.biodiversity_reducers = data.get('biodiversity_reducers', {})
        config.biodiversity_upload_paths = data.get('biodiversity_upload_paths', {})
        
        return config
    
    # 输出和状态目录
    output_dir: Path = field(default_factory=lambda: Path('data/output'))
    state_dir: Path = field(default_factory=lambda: Path('data/state'))
    
    # 基础分辨率 - 所有数据对齐到这个分辨率并输出
    base_resolution: float = 0.016666666666667  # ~1.85km (1/60度)
    
    # 统一的外部数据源配置 - 可扩展！
    external_datasets: List[Dict] = field(default_factory=lambda: [
        {
            'name': 'climate',
            'image': 'WORLDCLIM/V1/BIO',  # 是Image不是Collection！
            'bands': ['bio01', 'bio12'],  # 可配置要哪些波段
            'rename_pattern': 'climate_{band}'  # 重命名模式
        },
        {
            'name': 'elevation', 
            'image': 'MERIT/DEM/v1_0_3',
            'bands': ['dem'],
            'rename': ['elevation'],  # 直接重命名
            'compute_terrain': True  # 计算地形变量（slope, aspect_cos, aspect_sin, tpi）
        }
        # 轻松添加更多数据源：
        # {'name': 'soil', 'collection': 'OpenLandMap/SOL/...', 'bands': ['ph', 'clay']}
        # {'name': 'landcover', 'image': 'MODIS/006/MCD12Q1/...', 'bands': ['LC_Type1']}
    ])
    
    # Ocean masking参数
    mask_ocean: bool = True
    lat_min: float = -60  # 排除南极
    lat_max: float = 75   # 排除北极
    
    # 导出参数 - 关键参数，避免OOM
    # export_scale现在根据每个分辨率自动计算，不再硬编码
    export_max_pixels: float = 1e13  # 不是1e14！
    export_tile_size: int = 2048
    export_shard_size: int = 2560  # 关键参数！
    export_best_effort: bool = True
    export_check_interval: int = 30  # 检查间隔
    
    # 重试参数
    max_retries: int = 3
    retry_delay: int = 5
    
    # 清理
    cleanup_temp: bool = True
    cleanup_gcs: bool = True


class GEEPipeline:
    """主流程类 - 简单清晰的接口"""
    
    def __init__(self, config: GEEConfig):
        """
        Args:
            config: GEEConfig配置对象
        """
        self.config = config
        
        # 初始化各模块
        self.validator = TIFValidator()
        self.gee_ops = GEEOperations(
            config.service_account_key,
            config.project_id, 
            config.gcs_bucket
        )
        self.processor = DataProcessor(config)
        self.exporter = ExportManager(config, self.gee_ops)
        
        # 状态管理
        self.state: Optional[PipelineState] = None
    
    def run(self, resume: bool = False) -> Dict[str, Path]:
        """
        执行完整流程 - 使用config中的biodiversity_assets
        
        Args:
            resume: 是否从上次中断处恢复
            
        Returns:
            {分辨率: parquet文件路径} 字典
        """
        # 1. 初始化或恢复状态
        if resume:
            self.state = PipelineState.get_latest(self.config.state_dir)
            if self.state:
                logger.info(f"Resuming from step: {self.state.step}")
            else:
                logger.info("No previous state found, starting fresh")
                self.state = PipelineState(step='upload')
        else:
            self.state = PipelineState(step='upload')
        
        # 2. 检查和设置GEE资产
        if self.state.step == 'upload' and not self.state.asset_ids:
            logger.info("=== Step 1: Checking GEE assets ===")
            self._setup_gee_assets()
            self.state.step = 'process'
            self._save_state()
        
        # 4. 处理数据
        if self.state.can_resume_from('process'):
            logger.info("=== Step 3: Processing data ===")
            self._process_data()
            self.state.step = 'export'
            self._save_state()
        
        # 5. 导出到GCS
        if self.state.can_resume_from('export'):
            logger.info("=== Step 4: Exporting to GCS ===")
            self._export_to_gcs()
            self.state.step = 'download'
            self._save_state()
        
        # 6. 下载并转换为Parquet
        if self.state.can_resume_from('download'):
            logger.info("=== Step 5: Downloading and converting to Parquet ===")
            self._download_and_convert()
            self.state.step = 'complete'
            self._save_state()
        
        logger.info("✅ Pipeline completed successfully!")
        return self.state.completed_exports
    
    def _setup_gee_assets(self):
        """检查GEE资产是否存在，如果存在直接使用，否则尝试上传"""
        if not self.state.asset_ids:
            self.state.asset_ids = {}
        
        # 使用配置中的biodiversity_assets
        if not hasattr(self.config, 'biodiversity_assets'):
            raise ValueError("No biodiversity_assets defined in config")
        
        import ee
        for name, asset_path in self.config.biodiversity_assets.items():
            if name not in self.state.asset_ids:
                try:
                    # 尝试加载GEE资产
                    asset = ee.Image(asset_path)
                    # 尝试获取信息以验证资产存在
                    info = asset.getInfo()
                    logger.info(f"✅ Found GEE asset: {name} at {asset_path}")
                    self.state.asset_ids[name] = asset_path
                except Exception as e:
                    # 资产不存在，检查是否有上传路径
                    if hasattr(self.config, 'biodiversity_upload_paths') and name in self.config.biodiversity_upload_paths:
                        upload_path = self.config.biodiversity_upload_paths[name]
                        logger.info(f"Asset {name} not found in GEE, uploading from {upload_path}")
                        _, asset_id = self.gee_ops.upload_single_file(str(upload_path), name)
                        self.state.asset_ids[name] = asset_id
                    else:
                        raise ValueError(f"GEE asset {name} not found at {asset_path} and no upload path provided")
                
                self._save_state()
            else:
                logger.info(f"⏭️ Using cached asset: {name}")
        
        logger.info(f"✅ All {len(self.state.asset_ids)} biodiversity assets ready")
    
    def _process_data(self):
        """处理数据"""
        if not self.state.processed_images:
            # 加载GEE图像
            biodiv_images = {}
            for name, asset_id in self.state.asset_ids.items():
                biodiv_images[name] = ee.Image(asset_id)
            
            # 处理 - 简化版返回 {'1x': image}
            processed_images = self.processor.process_images(biodiv_images)
            
            # 保存处理状态
            self.state.processed_images = {'1x': "processed"}
            # 不要保存在实例中 - 这会造成内存泄漏
            # 后续步骤应该重新获取或通过参数传递
        else:
            logger.info("⏭️ Data already processed")
    
    def _export_to_gcs(self):
        """导出到GCS"""
        if not self.state.export_tasks:
            self.state.export_tasks = {}
        
        # 获取处理后的图像（每次都重新处理，不缓存）
        biodiv_images = {}
        for name, asset_id in self.state.asset_ids.items():
            biodiv_images[name] = ee.Image(asset_id)
        images = self.processor.process_images(biodiv_images)
        
        # 导出每个分辨率
        for res, img in images.items():
            if res not in self.state.export_tasks:
                logger.info(f"Exporting {res} resolution...")
                # 传递resolution_key让export知道正确的分辨率
                task_id = self.exporter.export_to_gcs(img, f"biodiv_{res}", resolution_key=res)
                self.state.export_tasks[res] = task_id
                self._save_state()
            
            # 等待导出完成
            if res not in self.state.completed_exports:
                logger.info(f"Waiting for {res} export...")
                self.exporter.wait_for_export(self.state.export_tasks[res])
    
    def _download_and_convert(self):
        """下载并转换为Parquet"""
        if not self.state.completed_exports:
            self.state.completed_exports = {}
        
        for res in self.state.export_tasks:
            if res not in self.state.completed_exports:
                logger.info(f"Downloading {res} resolution...")
                output_path = self.config.output_dir / f"biodiv_{res}.arrow"
                self.exporter.download_and_convert(f"biodiv_{res}", output_path)
                self.state.completed_exports[res] = output_path
                self._save_state()
            else:
                logger.info(f"⏭️ {res} already downloaded")
    
    def _save_state(self):
        """保存状态"""
        if self.state:
            state_file = self.config.state_dir / f"state_{self.state.timestamp}.json"
            self.state.save(state_file)


def main():
    """命令行入口 - 最简单的使用方式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GEE Biodiversity Data Pipeline')
    parser.add_argument('--config', required=True, help='YAML config file path')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    # 从YAML加载配置
    config = GEEConfig.from_yaml(args.config)
    
    # 运行流程
    pipeline = GEEPipeline(config)
    results = pipeline.run(resume=args.resume)
    
    # 输出结果
    logger.info("=== Results ===")
    for res, path in results.items():
        logger.info(f"{res}: {path}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
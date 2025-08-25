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

logger = logging.getLogger(__name__)


@dataclass
class GEEConfig:
    """简化的配置类 - 所有参数都有合理默认值"""
    
    # GEE认证
    service_account_key: str
    project_id: str
    gcs_bucket: str
    
    # 输入输出
    input_dir: Path = field(default_factory=lambda: Path('data/input'))
    output_dir: Path = field(default_factory=lambda: Path('data/output'))
    state_dir: Path = field(default_factory=lambda: Path('data/state'))
    
    # 基础分辨率 - 所有数据先对齐到这个分辨率
    base_resolution: float = 0.016666666666667  # ~1.85km (1/60度)
    
    # 输出分辨率 - 必须 >= base_resolution
    resolutions: Dict[str, float] = field(default_factory=lambda: {
        '1x': 0.016666666666667,  # ~1.85km (基础分辨率)
        '2x': 0.033333333,        # ~3.7km  
        '4x': 0.066666667         # ~7.4km
    })
    climate_bands: List[str] = field(default_factory=lambda: ['bio01', 'bio12'])
    include_elevation: bool = True
    
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
    
    def run(self, tif_files: Dict[str, Path], resume: bool = False) -> Dict[str, Path]:
        """
        执行完整流程
        
        Args:
            tif_files: {名称: TIF文件路径} 字典
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
        
        # 2. 验证输入文件
        if self.state.step == 'upload' and not self.state.asset_ids:
            logger.info("=== Step 1: Validating TIF files ===")
            self._validate_inputs(tif_files)
        
        # 3. 上传到GEE
        if self.state.can_resume_from('upload'):
            logger.info("=== Step 2: Uploading to GEE ===")
            self._upload_to_gee(tif_files)
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
    
    def _validate_inputs(self, tif_files: Dict[str, Path]):
        """验证输入文件"""
        for name, path in tif_files.items():
            if not self.validator.validate_tif(path):
                raise ValueError(f"Invalid TIF file: {path}")
        
        # 验证批次兼容性
        paths = list(tif_files.values())
        if len(paths) > 1 and not self.validator.validate_batch_compatibility(paths):
            raise ValueError("TIF files are not compatible (different CRS or resolution)")
        
        logger.info(f"✅ Validated {len(tif_files)} TIF files")
    
    def _upload_to_gee(self, tif_files: Dict[str, Path]):
        """上传文件到GEE"""
        if not self.state.asset_ids:
            self.state.asset_ids = {}
        
        for name, path in tif_files.items():
            if name not in self.state.asset_ids:
                logger.info(f"Uploading {name}...")
                _, asset_id = self.gee_ops.upload_single_file(str(path), name)
                self.state.asset_ids[name] = asset_id
                self._save_state()  # 每个文件后保存状态
            else:
                logger.info(f"⏭️ Skipping {name} - already uploaded")
    
    def _process_data(self):
        """处理数据"""
        if not self.state.processed_images:
            # 加载GEE图像
            biodiv_images = {}
            for name, asset_id in self.state.asset_ids.items():
                biodiv_images[name] = ee.Image(asset_id)
            
            # 处理
            processed = self.processor.process_images(biodiv_images)
            
            # 保存处理结果ID（GEE返回的是Image对象，需要序列化）
            self.state.processed_images = {
                res: img.serialize() for res, img in processed.items()
            }
        else:
            logger.info("⏭️ Data already processed")
    
    def _export_to_gcs(self):
        """导出到GCS"""
        if not self.state.export_tasks:
            self.state.export_tasks = {}
        
        # 反序列化图像
        images = {}
        for res, serialized in self.state.processed_images.items():
            images[res] = ee.Deserializer.decode(serialized)
        
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
                output_path = self.config.output_dir / f"biodiv_{res}.parquet"
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
    parser.add_argument('--service-account', required=True, help='Service account key file')
    parser.add_argument('--project', required=True, help='GCP project ID')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--input-dir', default='data/input', help='Input directory with TIF files')
    parser.add_argument('--output-dir', default='data/output', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    # 配置
    config = GEEConfig(
        service_account_key=args.service_account,
        project_id=args.project,
        gcs_bucket=args.bucket,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )
    
    # 查找输入文件
    tif_files = {}
    for tif_path in config.input_dir.glob('*.tif'):
        name = tif_path.stem
        tif_files[name] = tif_path
    
    if not tif_files:
        logger.error(f"No TIF files found in {config.input_dir}")
        return
    
    logger.info(f"Found {len(tif_files)} TIF files")
    
    # 运行流程
    pipeline = GEEPipeline(config)
    results = pipeline.run(tif_files, resume=args.resume)
    
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
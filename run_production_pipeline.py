#!/usr/bin/env python3
"""
生产环境Pipeline - 最小wrapper
==============================

从配置文件加载设置并运行pipeline
"""

import sys
import logging
from pathlib import Path

# 添加src到路径
sys.path.insert(0, 'src')

from gee_integration_v2.pipeline import GEEConfig, GEEPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_production():
    """运行生产环境pipeline"""
    
    print("=" * 70)
    print("生产环境Pipeline - 从配置文件加载")
    print("=" * 70)
    
    # 从YAML文件加载配置
    config_path = 'src/gee_integration_v2/config_simple.yaml'
    print(f"Loading config from: {config_path}")
    config = GEEConfig.from_yaml(config_path)
    
    # 创建pipeline并运行
    pipeline = GEEPipeline(config, test_mode=False)
    
    # 使用配置文件中的生物多样性资产
    biodiv_assets = config.biodiversity_assets
    reducers = config.biodiversity_reducers
    
    # 运行pipeline
    result = pipeline.run(
        biodiv_assets=biodiv_assets,
        reducers=reducers,
        resume=True
    )
    
    return result

if __name__ == "__main__":
    try:
        result = run_production()
        print(f"\n✅ Pipeline完成！输出: {result}")
    except Exception as e:
        logger.error(f"Pipeline失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
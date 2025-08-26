#!/usr/bin/env python3
"""
生产环境Pipeline
===============

处理真实的生物多样性数据：
1. daru-plants-richness (MAX reducer)
2. iucn-terrestrial-richness (MAX reducer)  
3. AM_Fungi_Richness_Predicted (MEAN reducer)
4. EcM_Fungi_Richness_Predicted (MEAN reducer)

只输出1x分辨率 (base resolution)

作者: 生产环境配置
"""

import ee
import sys
import time
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
    print("生产环境Pipeline - 真实数据处理")
    print("=" * 70)
    
    # 1. 配置（简化版）
    config = GEEConfig(
        service_account_key='.gee/service-account-key.json',
        project_id='triple-carrier-441622-t1',
        gcs_bucket='biodiv-gee-yl998',
        input_dir=Path('data/production'),  # 不需要实际创建，只是占位
        output_dir=Path('outputs/production'),
        state_dir=Path('state/production'),
        base_resolution=0.016666666666667,  # ~1.85km，与mwd24数据匹配
        # 外部数据集（climate和elevation）
        external_datasets=[
            {
                'name': 'climate',
                'image': 'WORLDCLIM/V1/BIO',
                'bands': ['bio01', 'bio04', 'bio12', 'bio15'],  # 温度、温度季节性、降水、降水季节性
                'rename': ['annual_temp', 'temp_seasonality', 'annual_precip', 'precip_seasonality']
            },
            {
                'name': 'elevation',
                'image': 'MERIT/DEM/v1_0_3',
                'bands': ['dem'],
                'rename': ['elevation']
            },
            {
                'name': 'pet',
                'collection': 'IDAHO_EPSCOR/TERRACLIMATE',
                'bands': ['pet'],
                'reducer': 'mean',  # 计算多年月均值
                'filter_dates': ['2000-01-01', '2020-12-31'],  # 使用2000-2020年数据
                'rename': ['pet_mm_month']  # 单位: mm/month
            },
            # 土壤数据（SoilGrids ISRIC - 250m分辨率）
            {
                'name': 'soil_sand',
                'image': 'projects/soilgrids-isric/sand_mean',
                'bands': ['sand_0-5cm_mean'],  # 0-5cm深度的沙含量
                'rename': ['sand_content_pct']  # 单位: %
            },
            {
                'name': 'soil_carbon',
                'image': 'projects/soilgrids-isric/soc_mean',
                'bands': ['soc_0-5cm_mean'],  # 0-5cm深度的土壤有机碳
                'rename': ['soil_organic_carbon']  # 单位: g/kg
            },
            {
                'name': 'soil_ph',
                'image': 'projects/soilgrids-isric/phh2o_mean',
                'bands': ['phh2o_0-5cm_mean'],  # 0-5cm深度的pH值
                'rename': ['soil_ph']  # 单位: pH * 10
            }
        ],
        mask_ocean=True,
        lat_min=-60,
        lat_max=75,
        cleanup_temp=False,  # 保留临时文件便于检查
        cleanup_gcs=False    # 保留GCS文件便于检查
    )
    
    # 配置每个数据集的处理参数
    config.biodiversity_datasets = [
        {
            'name': 'plants_richness',
            'path': '/maps/mwd24/richness/daru-plants-richness.tif',
            'nodata_value': 0,
            'reducer': 'max'  # 物种丰富度用MAX保留峰值
        },
        {
            'name': 'terrestrial_richness', 
            'path': '/maps/mwd24/richness/iucn-terrestrial-richness.tif',
            'nodata_value': 0,
            'reducer': 'max'  # 物种丰富度用MAX保留峰值
        },
        {
            'name': 'am_fungi_richness',
            'path': '/maps/spun/AM_fungi/AM_Fungi_Richness_Predicted.tif',
            'nodata_value': -3.3999999521443642e+38,
            'reducer': 'mean'  # 预测值用MEAN平均
        },
        {
            'name': 'ecm_fungi_richness',
            'path': '/maps/spun/EcM_fungi/EcM_Fungi_Richness_Predicted.tif',
            'nodata_value': -3.3999999521443642e+38,
            'reducer': 'mean'  # 预测值用MEAN平均
        }
    ]
    
    # 2. 准备输入文件字典
    tif_files = {}
    for dataset in config.biodiversity_datasets:
        name = dataset['name']
        path = Path(dataset['path'])
        
        if not path.exists():
            print(f"❌ 数据文件不存在: {path}")
            return None
        else:
            print(f"✓ 找到数据文件: {name} ({path.stat().st_size / (1024**2):.1f} MB)")
            tif_files[name] = path
    
    print(f"\n共找到 {len(tif_files)} 个数据文件")
    
    # 3. 初始化EE
    print("\n初始化Earth Engine...")
    credentials = ee.ServiceAccountCredentials(
        email='gee-biodiversity@triple-carrier-441622-t1.iam.gserviceaccount.com',
        key_file=config.service_account_key
    )
    ee.Initialize(credentials=credentials, project=config.project_id)
    print("✓ EE初始化成功")
    
    # 4. 运行pipeline
    print("\n开始执行Pipeline...")
    print("-" * 40)
    print("处理流程：")
    print("1. 上传4个TIF文件到GEE")
    print("2. 处理NoData值")
    print("3. 生成1x分辨率（不需要降采样）")
    print("4. 合并4个数据集")
    print("5. 应用ocean mask")
    print("6. 添加climate和elevation数据")
    print("7. 导出到GCS")
    print("8. 下载并转换为Parquet")
    print("-" * 40)
    
    pipeline = GEEPipeline(config)
    
    try:
        start_time = time.time()
        
        # 运行pipeline（不恢复，从头开始）
        results = pipeline.run(tif_files, resume=False)
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"✅ Pipeline执行成功！耗时: {elapsed/60:.1f} 分钟")
        print("=" * 70)
        
        # 5. 显示结果
        print("\n输出文件:")
        for res, path in results.items():
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"  {res}: {path} ({size_mb:.2f} MB)")
            else:
                print(f"  {res}: {path} (等待下载)")
        
        # 6. 验证输出
        print("\n验证输出内容...")
        output_path = list(results.values())[0]
        
        # 检查TIF文件
        tif_path = Path(str(output_path).replace('.parquet', '.tif'))
        if tif_path.exists():
            import rasterio
            with rasterio.open(tif_path) as src:
                print(f"\n输出TIF详情:")
                print(f"  - 尺寸: {src.width} x {src.height}")
                print(f"  - 波段数: {src.count}")
                print(f"  - 分辨率: {src.res[0]:.9f}°")
                
                expected_bands = [
                    'plants_richness',
                    'terrestrial_richness', 
                    'am_fungi_richness',
                    'ecm_fungi_richness',
                    'temperature',
                    'precipitation',
                    'elevation'
                ]
                print(f"  - 预期波段 ({len(expected_bands)}): {', '.join(expected_bands)}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # 创建必要的目录
    Path('outputs/production').mkdir(parents=True, exist_ok=True)
    Path('state/production').mkdir(parents=True, exist_ok=True)
    
    # 运行生产环境pipeline
    results = run_production()
    
    if results:
        print("\n🎉 生产环境处理完成！")
        print("输出位置: outputs/production/")
        print("\n下一步：")
        print("1. 检查输出数据质量")
        print("2. 转换为分析所需的格式")
        print("3. 进行后续的生物多样性分析")
    else:
        print("\n💥 生产环境处理失败！")
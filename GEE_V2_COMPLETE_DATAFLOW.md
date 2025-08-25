# GEE V2 完整数据流（修正版）

## ⚠️ 关键修正：导出分辨率

**错误**：`scale=30` (30米) - 这会把1855米的数据上采样到30米！  
**正确**：每个分辨率使用对应的scale：
- 1x: scale=1855米
- 2x: scale=3700米  
- 4x: scale=7400米

## 完整数据流

### 1. 输入阶段
```python
tif_files = {
    'plants_richness': 'plants.tif',      # 任意分辨率
    'terrestrial_richness': 'terrestrial.tif'  # 任意分辨率
}
```

### 2. 上传到GEE
```
本地TIF → GCS临时存储 → GEE Asset
```
- 每个文件独立上传
- 支持断点续传（保存asset_ids到state）

### 3. 核心处理步骤 ⭐️

#### 3.1 对齐到基础分辨率
```python
base_resolution = 0.016666666667  # 度
base_scale = 1855  # 米

# 所有数据对齐到同一网格
plants_aligned = align_to_base(plants, 'plants_richness')
terrestrial_aligned = align_to_base(terrestrial, 'terrestrial_richness')
```

#### 3.2 合并生物多样性
```python
# 现在可以安全合并（都在同一网格）
merged = plants_aligned.addBands(terrestrial_aligned)
```

#### 3.3 添加辅助数据
```python
# WorldClim气候数据（~1km → 1.85km）
climate = WorldClim.select(['bio01', 'bio12'])
climate_aligned = align_to_base(climate)
merged = merged.addBands(climate_aligned)

# MERIT DEM高程（90m → 1.85km）
elevation = MERIT_DEM.select('dem')
elevation_aligned = align_to_base(elevation)
merged = merged.addBands(elevation_aligned)
```

### 4. 生成多分辨率

```python
processed = {
    '1x': merged,                          # 1.85km (基础)
    '2x': downsample_2x(merged),           # 3.7km (1次降采样)
    '4x': downsample_2x(downsample_2x(merged))  # 7.4km (2次降采样)
}
```

**递归2x降采样原理**：
- 每次降采样使用`reduceResolution(maxPixels=1024)`
- 绕过GEE的256像素聚合限制
- 只支持2的幂次（1, 2, 4, 8...）

### 5. 导出到GCS（正确的scale！）

```python
# 每个分辨率使用对应的scale
exports = {
    '1x': {'scale': 1855, 'size': '~2GB'},
    '2x': {'scale': 3700, 'size': '~500MB'},
    '4x': {'scale': 7400, 'size': '~125MB'}
}

for res_name, image in processed.items():
    scale = exports[res_name]['scale']
    
    export_task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f'biodiv_{res_name}',
        scale=scale,  # 正确的分辨率！不是30米！
        maxPixels=1e13,
        shardSize=2560,
        fileFormat='GeoTIFF'
    )
```

### 6. 下载并转换为Parquet

```python
# GCS → 本地GeoTIFF → Parquet
for res_name in ['1x', '2x', '4x']:
    # 下载
    gsutil cp gs://bucket/biodiv_{res_name}*.tif local/
    
    # 转换（窗口化处理）
    convert_to_parquet(f'biodiv_{res_name}.tif')
```

## 数据分辨率对照表

| 阶段 | 1x | 2x | 4x |
|-----|---|----|----|
| **处理中（度）** | 0.016666667 | 0.033333333 | 0.066666667 |
| **处理中（km）** | 1.85 | 3.7 | 7.4 |
| **导出scale（米）** | 1855 | 3700 | 7400 |
| **像素数量** | ~9M | ~2.25M | ~0.56M |
| **文件大小** | ~2GB | ~500MB | ~125MB |

## 为什么不能用固定的30米？

1. **上采样无意义**：1855米的数据导出成30米不会增加信息，只会浪费存储
2. **计算爆炸**：30米分辨率会产生3800倍的像素！(1855/30)²
3. **违背设计**：我们精心设计的多分辨率系统变得毫无意义

## Linus的判断

```text
"这就像把320x240的视频转换成4K - 除了浪费硬盘，你什么都得不到。"

正确的做法：
- 处理什么分辨率，就导出什么分辨率
- 不要假装有不存在的细节
- 实事求是
```

## 状态总结

**已修复**：
- ✅ 数据对齐逻辑
- ✅ 递归降采样
- ✅ 导出分辨率现在正确匹配处理分辨率

**V2现在真正可用了！**
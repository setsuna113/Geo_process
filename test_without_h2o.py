#\!/usr/bin/env python
"""Test ML pipeline without H2O dependency"""

import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '/home/jason/geo')

print("Testing ML pipeline components without H2O...")

# Test data loading
print("\n1. Testing data loading with spatial sampling...")
import pyarrow.parquet as pq

parquet_file = 'data/merged/biodiversity_climate_1x_production.parquet'
parquet = pq.ParquetFile(parquet_file)
total_rows = parquet.metadata.num_rows
print(f"   Total rows in file: {total_rows:,}")

# Test spatial sampling
spatial_step = 500
sampled_indices = list(range(0, total_rows, spatial_step))
print(f"   After sampling (every {spatial_step}th): {len(sampled_indices):,} rows")

# Test reading a small sample
print("\n2. Reading sample data...")
df_sample = parquet.read().to_pandas().iloc[sampled_indices[:100]]
print(f"   Sample shape: {df_sample.shape}")
print(f"   Columns: {list(df_sample.columns[:10])}...")

# Test preprocessing
print("\n3. Testing preprocessing...")
from src.machine_learning.preprocessing import ColumnStandardizer

standardizer = ColumnStandardizer()
print("   ✓ ColumnStandardizer imported")

# Test MESF components
print("\n4. Testing MESF components...")
from src.machine_learning.spatial.mesf_cache_manager import MESFCacheManager

cache_mgr = MESFCacheManager()
print(f"   ✓ MESF cache manager initialized")
print(f"   Cache directory: {cache_mgr.cache_dir}")

# Test configuration
print("\n5. Testing configuration...")
from src.machine_learning.config_loader import MLRegressionConfig

config = MLRegressionConfig(config_path='src/config/ml/ml_config.yml')
print("   ✓ Configuration loaded")

# Test method imports (without initialization)
print("\n6. Testing method imports...")
from src.machine_learning.research.methods import (
    Method1ContinuousGLM,
    Method2CategoricalGLM,
    Method3DML
)
print("   ✓ All methods imported successfully")

print("\n✅ All non-H2O components working\!")
print("\nNote: Methods 1-3 require H2O for GLM/GBM models.")
print("With only 6GB available RAM, H2O needs memory limits:")
print("  - Set max_mem_size='4G' instead of '128G'")
print("  - Use larger spatial_grid_step (e.g., 1000-2000)")

# Test reports directory
output_dir = 'outputs/ml_test_sampling500'
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory ready: {output_dir}")

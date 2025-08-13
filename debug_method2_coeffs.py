#!/usr/bin/env python3
"""
Debug Method 2 to see what coefficient names are available.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

# Create simple data
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'x': np.random.uniform(-180, 180, n),
    'y': np.random.uniform(-90, 90, n),
    'bio01': np.random.normal(15, 5, n),
    'bio04': np.random.normal(500, 100, n),
    'bio12': np.random.normal(1000, 300, n),
    'bio15': np.random.normal(50, 20, n),
    'elevation': np.random.uniform(0, 3000, n),
    'plant_richness': np.random.poisson(100, n),
    'animal_richness': np.random.poisson(50, n),
    'am_fungi_richness': np.random.poisson(30, n),
    'ecm_fungi_richness': np.random.poisson(20, n),
    'am_fungi_coeff_var': np.random.uniform(0.2, 0.6, n),
    'ecm_fungi_coeff_var': np.random.uniform(0.2, 0.6, n)
})

sample_path = '/tmp/debug_method2.parquet'
data.to_parquet(sample_path)

import h2o
h2o.init(max_mem_size='4G', nthreads=2, min_mem_size='2G')

from src.machine_learning.ml_orchestrator import MLOrchestrator

orchestrator = MLOrchestrator(config_path='src/config/ml/ml_config.yml')
orchestrator.load_and_preprocess_data(sample_path)

# Initialize Method 2 and fit just Model 3a
from src.machine_learning.methods.method2_categorical_refactored import Method2H2Testing

method2 = orchestrator.method2
h2o_data = orchestrator.preprocessed_data

# Get ME columns
me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
print(f"Found {len(me_cols)} ME columns")

# Model 3a: Continuous interaction model
feature_cols_3a = [
    'bio01_std', 'bio04_std', 'bio12_std', 'bio15_std',
    'elevation_std', 'elevation_std_sq',
    'plants_richness_log', 'vertebrates_richness_log',
    'bio1_x_plants'  # Continuous interaction term
] + me_cols

# Create interaction term
h2o_data['bio1_x_plants'] = h2o_data['bio01_std'] * h2o_data['plants_richness_log']

# Fit model
from src.machine_learning.models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM

model = H2ONegativeBinomialGLM()
model_data = h2o_data[feature_cols_3a + ['fungi_richness']]
model.fit(model_data, y='fungi_richness', lambda_search=False, validation_fraction=0.0)

# Get all coefficients
print("\nModel 3a Coefficients:")
coeffs = model.get_all_coefficients()
for name, value in coeffs.items():
    if value is not None:
        print(f"  {name}: {value:.4f}")

# Check specific coefficient names
print("\nChecking for interaction coefficient:")
for name in coeffs.keys():
    if 'bio' in name.lower() and 'plant' in name.lower():
        print(f"  Found: {name}")
    if 'x' in name.lower():
        print(f"  Found interaction: {name}")

h2o.cluster().shutdown()
print("\nDebug completed!")
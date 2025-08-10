#\!/usr/bin/env python
"""Run Method 1 with minimal setup"""

import os
import sys
sys.path.insert(0, '/home/jason/geo')

# Set environment for smaller H2O memory
os.environ['H2O_MAX_MEM'] = '16G'

print("Starting Method 1 test with spatial sampling 500...")

from src.machine_learning.research.methods import Method1ContinuousGLM
from src.machine_learning.config_loader import MLRegressionConfig

# Load config
config = MLRegressionConfig(config_path='src/config/ml/ml_config.yml')

# Initialize Method 1
print("Initializing Method 1...")
method1 = Method1ContinuousGLM(config)

# Set up paths
input_path = 'data/merged/biodiversity_climate_1x_production.parquet'
output_dir = 'outputs/ml_test_sampling500/method1'
os.makedirs(output_dir, exist_ok=True)

try:
    print(f"Loading data with spatial sampling 500...")
    # Load data with spatial sampling
    data = method1.load_data(input_path, spatial_grid_step=500)
    print(f"Loaded {len(data)} rows after sampling")
    
    print("Preparing data...")
    prepared_data = method1.prepare_data(data)
    print(f"Prepared {len(prepared_data)} rows")
    
    print("Running Method 1 analysis...")
    results = method1.run_analysis()
    
    print("Generating reports...")
    method1.generate_reports(output_dir)
    
    print(f"\n✅ Method 1 completed successfully\!")
    print(f"Reports saved to: {output_dir}")
    
except Exception as e:
    print(f"\n❌ Error running Method 1: {e}")
    import traceback
    traceback.print_exc()

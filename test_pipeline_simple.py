#!/usr/bin/env python3
"""
Simple test of ML pipeline with synthetic data to verify reporters work.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create synthetic data
print("Creating synthetic test data...")
np.random.seed(42)
n = 1000  # Small dataset for testing

data = pd.DataFrame({
    'x': np.random.uniform(-180, 180, n),
    'y': np.random.uniform(-90, 90, n),
    'animal_richness': np.random.poisson(50, n),
    'plant_richness': np.random.poisson(100, n),
    'am_fungi_richness': np.random.poisson(30, n),
    'am_fungi_coeff_var': np.random.uniform(0.1, 1.0, n),
    'ecm_fungi_richness': np.random.poisson(20, n),
    'ecm_fungi_coeff_var': np.random.uniform(0.1, 1.0, n),
    'bio01': np.random.normal(15, 5, n),  # Mean annual temperature
    'bio04': np.random.normal(500, 100, n),  # Temperature seasonality
    'bio12': np.random.normal(1000, 300, n),  # Annual precipitation
    'bio15': np.random.normal(50, 20, n),  # Precipitation seasonality
    'elevation': np.random.uniform(0, 3000, n)
})

# Save synthetic data
sample_path = '/tmp/synthetic_biodiversity_1k.parquet'
data.to_parquet(sample_path)
print(f"Synthetic data saved to: {sample_path}")
print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Initialize H2O separately to avoid timeout
print("\nInitializing H2O...")
import h2o
h2o.init(max_mem_size='8G', nthreads=4)
print("H2O initialized successfully")

# Now run the pipeline
from src.machine_learning.ml_orchestrator import MLOrchestrator

print("\nInitializing orchestrator...")
config_path = 'src/config/ml/ml_config.yml'
orchestrator = MLOrchestrator(config_path=config_path)

print("\nLoading and preprocessing synthetic data...")
orchestrator.load_and_preprocess_data(sample_path)
print(f"Preprocessed data shape: {orchestrator.preprocessed_data.shape}")

print("\nRunning Method 1 (H0-H1 GLM Testing)...")
try:
    results = orchestrator.run_method1_h0_h1()
    print("Method 1 completed successfully!")
    
    # Print results summary
    if 'method1' in orchestrator.results:
        print("\nMethod 1 Results Summary:")
        for key in orchestrator.results['method1'].keys():
            print(f"  - {key}")
except Exception as e:
    print(f"Error running Method 1: {e}")
    import traceback
    traceback.print_exc()

# Generate reports
output_dir = Path('ml_results/method1_synthetic')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating reports in {output_dir}...")
try:
    orchestrator.generate_method_reports(output_dir)
    print("Reports generated successfully!")
except Exception as e:
    print(f"Error generating reports: {e}")
    import traceback
    traceback.print_exc()

print("\nChecking generated files...")
for file in output_dir.glob('**/*'):
    if file.is_file():
        print(f"  - {file.relative_to(output_dir)}")

# Shutdown H2O
h2o.cluster().shutdown()
print("\nH2O cluster shutdown")
print("Test completed!")
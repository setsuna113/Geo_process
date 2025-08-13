#!/usr/bin/env python3
"""
Test Method 2 (H2 Testing) with synthetic data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create synthetic data
print("Creating synthetic test data for Method 2...")
np.random.seed(42)
n = 1000

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
sample_path = '/tmp/synthetic_method2_1k.parquet'
data.to_parquet(sample_path)
print(f"Data saved to: {sample_path}")

# Initialize H2O
print("\nInitializing H2O...")
import h2o
h2o.init(max_mem_size='8G', nthreads=4, min_mem_size='4G')

from src.machine_learning.ml_orchestrator import MLOrchestrator

print("\nInitializing orchestrator...")
config_path = 'src/config/ml/ml_config.yml'
orchestrator = MLOrchestrator(config_path=config_path)

print("\nLoading and preprocessing data...")
orchestrator.load_and_preprocess_data(sample_path)
print(f"Preprocessed data shape: {orchestrator.preprocessed_data.shape}")

print("\nRunning Method 2 (H2 Testing)...")
print("Testing hypothesis: Climate-diversity interaction differs in temperate vs non-temperate regions")

try:
    results = orchestrator.run_method2_h2()
    print("Method 2 completed successfully!")
    
    if 'method2' in orchestrator.results:
        print("\nMethod 2 Results Summary:")
        for key in orchestrator.results['method2'].keys():
            print(f"  - {key}")
            
        # Check H2 conclusion
        if 'h2_conclusion' in orchestrator.results['method2']:
            conclusion = orchestrator.results['method2']['h2_conclusion']
            print(f"\nH2 Hypothesis Test:")
            print(f"  Supported: {conclusion.get('h2_supported', 'N/A')}")
            print(f"  Interpretation: {conclusion.get('interpretation', 'N/A')}")
            
except Exception as e:
    print(f"Error running Method 2: {e}")
    import traceback
    traceback.print_exc()

# Generate reports
output_dir = Path('ml_results/method2_synthetic')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating reports in {output_dir}...")
try:
    orchestrator.generate_method_reports(output_dir)
    print("Reports generated successfully!")
except Exception as e:
    print(f"Error generating reports: {e}")

print("\nChecking generated files...")
for file in output_dir.glob('**/*'):
    if file.is_file():
        print(f"  - {file.relative_to(output_dir)}")

h2o.cluster().shutdown()
print("\nTest completed!")
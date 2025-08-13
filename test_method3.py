#!/usr/bin/env python3
"""
Test Method 3 (DML Causal) with synthetic data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create synthetic data with better variance for DML
print("Creating synthetic test data for Method 3...")
np.random.seed(42)
n = 2000  # More data for DML

# Create data with causal structure
data = pd.DataFrame({
    'x': np.random.uniform(-180, 180, n),
    'y': np.random.uniform(-90, 90, n),
    'bio01': np.random.normal(15, 5, n),  # Mean annual temperature
    'bio04': np.random.normal(500, 100, n),  # Temperature seasonality
    'bio12': np.random.normal(1000, 300, n),  # Annual precipitation
    'bio15': np.random.normal(50, 20, n),  # Precipitation seasonality
    'elevation': np.random.uniform(0, 3000, n)
})

# Create biodiversity with some causal relationships
data['plant_richness'] = (
    50 + 
    2 * data['bio01'] + 
    0.01 * data['bio12'] + 
    np.random.normal(0, 20, n)
).clip(1, None).astype(int)

data['animal_richness'] = (
    30 + 
    0.5 * data['plant_richness'] + 
    1.5 * data['bio01'] + 
    np.random.normal(0, 15, n)
).clip(1, None).astype(int)

# Create fungi with causal effect from plants
data['am_fungi_richness'] = (
    20 + 
    0.3 * data['plant_richness'] +  # Causal effect
    0.2 * data['animal_richness'] +
    1.0 * data['bio01'] + 
    np.random.normal(0, 10, n)
).clip(1, None).astype(int)

data['ecm_fungi_richness'] = (
    15 + 
    0.2 * data['plant_richness'] + 
    np.random.normal(0, 8, n)
).clip(1, None).astype(int)

data['am_fungi_coeff_var'] = np.random.uniform(0.1, 1.0, n)
data['ecm_fungi_coeff_var'] = np.random.uniform(0.1, 1.0, n)

# Save synthetic data
sample_path = '/tmp/synthetic_method3_2k.parquet'
data.to_parquet(sample_path)
print(f"Data saved to: {sample_path}")
print(f"Data shape: {data.shape}")

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

print("\nRunning Method 3 (H0-H1 Causal with DML)...")
print("Testing causal effect of biodiversity on fungi richness")

try:
    results = orchestrator.run_method3_dml()
    print("Method 3 completed successfully!")
    
    if 'method3' in orchestrator.results:
        print("\nMethod 3 Results Summary:")
        for key in orchestrator.results['method3'].keys():
            print(f"  - {key}")
            
        # Check causal effects
        if 'dml_results' in orchestrator.results['method3']:
            dml = orchestrator.results['method3']['dml_results']
            if 'treatment_effects' in dml:
                print("\nCausal Treatment Effects:")
                for treatment, effect in dml['treatment_effects'].items():
                    print(f"  {treatment}: {effect.get('ate', 'N/A')}")
                    
except Exception as e:
    print(f"Error running Method 3: {e}")
    import traceback
    traceback.print_exc()

# Generate reports
output_dir = Path('ml_results/method3_synthetic')
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
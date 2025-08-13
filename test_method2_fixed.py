#!/usr/bin/env python3
"""
Test Method 2 (H2 Testing) with better synthetic data to avoid collinearity.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create synthetic data with proper variance and reduced collinearity
print("Creating improved synthetic data for Method 2...")
np.random.seed(42)
n = 2000  # More samples for stability

# Create base climate variables with realistic correlations
data = pd.DataFrame({
    'x': np.random.uniform(-180, 180, n),
    'y': np.random.uniform(-90, 90, n),
})

# Create climate variables with some correlation but not perfect
data['bio01'] = np.random.normal(15, 5, n)  # Mean annual temperature
data['bio04'] = 500 + 10 * data['bio01'] + np.random.normal(0, 200, n)  # Temperature seasonality (correlated with temp)
data['bio12'] = 1000 - 20 * data['bio01'] + np.random.normal(0, 400, n)  # Annual precipitation (inversely correlated)
data['bio15'] = np.random.normal(50, 20, n)  # Precipitation seasonality (independent)
data['elevation'] = np.random.uniform(0, 3000, n)

# Create biodiversity variables with realistic relationships
# Plants affected by temperature and precipitation
data['plant_richness'] = np.maximum(1, 
    100 + 
    2 * data['bio01'] + 
    0.01 * data['bio12'] - 
    0.01 * data['elevation'] +
    np.random.poisson(20, n)
).astype(int)

# Animals affected by plants and climate
data['animal_richness'] = np.maximum(1,
    50 + 
    0.3 * data['plant_richness'] + 
    1.5 * data['bio01'] +
    np.random.poisson(15, n)
).astype(int)

# Create fungi richness with temperate interaction effect (key for H2)
# In temperate regions (high bio04), biodiversity effect is stronger
is_temperate = (data['bio04'] > 600).astype(float)

# Base fungi richness
fungi_base = 30 + 1.0 * data['bio01'] + 0.005 * data['bio12']

# Biodiversity effect that varies by temperate status (H2 hypothesis)
biodiversity_effect_temperate = 0.5 * data['plant_richness'] + 0.3 * data['animal_richness']
biodiversity_effect_tropical = 0.2 * data['plant_richness'] + 0.1 * data['animal_richness']

data['am_fungi_richness'] = np.maximum(1,
    fungi_base + 
    is_temperate * biodiversity_effect_temperate + 
    (1 - is_temperate) * biodiversity_effect_tropical +
    np.random.normal(0, 10, n)
).astype(int)

data['ecm_fungi_richness'] = np.maximum(1,
    20 + 
    0.2 * data['plant_richness'] + 
    0.1 * data['elevation'] / 100 +
    np.random.poisson(10, n)
).astype(int)

# Add coefficient of variation
data['am_fungi_coeff_var'] = np.random.uniform(0.1, 0.8, n)
data['ecm_fungi_coeff_var'] = np.random.uniform(0.1, 0.8, n)

# Add some noise variables to ensure we have enough features
for i in range(5):
    data[f'noise_{i}'] = np.random.normal(0, 1, n)

# Save synthetic data
sample_path = '/tmp/synthetic_method2_fixed.parquet'
data.to_parquet(sample_path)
print(f"Data saved to: {sample_path}")
print(f"Data shape: {data.shape}")
print(f"Temperate regions: {is_temperate.sum()}/{n} ({100*is_temperate.mean():.1f}%)")

# Check for multicollinearity
print("\nChecking correlations to avoid multicollinearity:")
climate_vars = ['bio01', 'bio04', 'bio12', 'bio15', 'elevation']
corr_matrix = data[climate_vars].corr()
print("Climate variable correlations:")
print(corr_matrix.round(2))

# Initialize H2O with proper settings
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
            
        # Check for contrast results
        if 'contrasts' in orchestrator.results['method2']:
            print(f"\nContrast Analysis Results:")
            contrasts = orchestrator.results['method2']['contrasts']
            for name, result in contrasts.items():
                print(f"  {name}: estimate={result.get('estimate', 'N/A')}, p={result.get('p_value', 'N/A')}")
                
except Exception as e:
    print(f"Error running Method 2: {e}")
    import traceback
    traceback.print_exc()

# Generate reports
output_dir = Path('ml_results/method2_fixed')
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
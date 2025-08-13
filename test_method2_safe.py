#!/usr/bin/env python3
"""
Test Method 2 with safer GLM settings to avoid numerical issues.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Create simpler synthetic data without perfect collinearity
print("Creating simplified synthetic data for Method 2...")
np.random.seed(42)
n = 1500

# Independent base variables
data = pd.DataFrame({
    'x': np.random.uniform(-180, 180, n),
    'y': np.random.uniform(-90, 90, n),
    'bio01': np.random.normal(15, 5, n),
    'bio04': np.random.normal(500, 100, n),  # Independent
    'bio12': np.random.normal(1000, 300, n),  # Independent  
    'bio15': np.random.normal(50, 20, n),  # Independent
    'elevation': np.random.uniform(0, 3000, n)
})

# Create biodiversity with some noise
data['plant_richness'] = np.maximum(1, 
    np.random.poisson(100 + 2 * data['bio01'], n)
)

data['animal_richness'] = np.maximum(1,
    np.random.poisson(50 + 0.5 * data['plant_richness'], n)
)

# Simple fungi richness - key is the temperate interaction
is_temperate = (data['bio04'] > 500).astype(float)

# Create fungi with interaction effect
base_effect = 30 + 0.5 * data['bio01']
biodiversity_effect = 0.3 * data['plant_richness']

# Interaction: biodiversity effect is stronger in temperate regions
data['am_fungi_richness'] = np.maximum(1,
    base_effect + 
    biodiversity_effect * (1 + 0.5 * is_temperate) +  # Interaction term
    np.random.normal(0, 5, n)
).astype(int)

data['ecm_fungi_richness'] = np.random.poisson(20, n)
data['am_fungi_coeff_var'] = np.random.uniform(0.2, 0.6, n)
data['ecm_fungi_coeff_var'] = np.random.uniform(0.2, 0.6, n)

# Save data
sample_path = '/tmp/synthetic_method2_safe.parquet'
data.to_parquet(sample_path)
print(f"Data saved to: {sample_path}")
print(f"Data shape: {data.shape}")

# Initialize H2O
print("\nInitializing H2O...")
import h2o
h2o.init(max_mem_size='8G', nthreads=4)

# Monkey-patch the GLM to use safer settings
from src.machine_learning.models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM

# Save original fit method
original_fit = H2ONegativeBinomialGLM.fit

def safe_fit(self, X, y=None, validation_fraction=0.0, lambda_search=False, **kwargs):
    """Override fit to use safer settings for Method 2."""
    print("Using safe GLM settings: lambda_search=False, validation_fraction=0.0")
    return original_fit(self, X, y, validation_fraction=0.0, lambda_search=False, **kwargs)

# Apply monkey patch
H2ONegativeBinomialGLM.fit = safe_fit

from src.machine_learning.ml_orchestrator import MLOrchestrator

print("\nInitializing orchestrator...")
config_path = 'src/config/ml/ml_config.yml'
orchestrator = MLOrchestrator(config_path=config_path)

print("\nLoading and preprocessing data...")
orchestrator.load_and_preprocess_data(sample_path)
print(f"Preprocessed data shape: {orchestrator.preprocessed_data.shape}")

print("\nRunning Method 2 (H2 Testing)...")
try:
    results = orchestrator.run_method2_h2()
    print("Method 2 completed successfully!")
    
    if 'method2' in orchestrator.results:
        print("\nMethod 2 Results:")
        for key, value in orchestrator.results['method2'].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if not isinstance(v, dict):
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
                
except Exception as e:
    print(f"Error: {e}")
    # Try to get partial results
    if hasattr(orchestrator, 'method2'):
        method2 = orchestrator.method2
        if hasattr(method2, 'model_3a_continuous') and method2.model_3a_continuous:
            print("\nModel 3a (continuous) was fitted successfully")
        if hasattr(method2, 'model_3b_categorical') and method2.model_3b_categorical:
            print("Model 3b (categorical) was fitted successfully")

# Generate reports
output_dir = Path('ml_results/method2_safe')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating reports in {output_dir}...")
orchestrator.generate_method_reports(output_dir)

print("\nChecking generated files...")
for file in output_dir.glob('**/*'):
    if file.is_file():
        print(f"  - {file.relative_to(output_dir)}")

h2o.cluster().shutdown()
print("\nTest completed!")
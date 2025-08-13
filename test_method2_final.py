#!/usr/bin/env python3
"""
Test Method 2 with improved coefficient name handling.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Create synthetic data
print("Creating synthetic data for Method 2...")
np.random.seed(42)
n = 1000

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
})

# Create fungi with interaction effect
is_temperate = (data['bio04'] > 500).astype(float)
data['am_fungi_richness'] = np.maximum(1,
    30 + 0.5 * data['bio01'] + 
    0.3 * data['plant_richness'] * (1 + 0.5 * is_temperate) +
    np.random.normal(0, 5, n)
).astype(int)

data['ecm_fungi_richness'] = np.random.poisson(20, n)
data['am_fungi_coeff_var'] = np.random.uniform(0.2, 0.6, n)
data['ecm_fungi_coeff_var'] = np.random.uniform(0.2, 0.6, n)

sample_path = '/tmp/synthetic_method2_final.parquet'
data.to_parquet(sample_path)
print(f"Data saved to: {sample_path}")

# Initialize H2O
print("\nInitializing H2O...")
import h2o
h2o.init(max_mem_size='8G', nthreads=4)

# Monkey-patch for safe GLM settings
from src.machine_learning.models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
original_fit = H2ONegativeBinomialGLM.fit

def safe_fit(self, X, y=None, validation_fraction=0.0, lambda_search=False, **kwargs):
    """Force safe settings."""
    return original_fit(self, X, y, validation_fraction=0.0, lambda_search=False, **kwargs)

H2ONegativeBinomialGLM.fit = safe_fit

from src.machine_learning.ml_orchestrator import MLOrchestrator

print("\nInitializing orchestrator...")
orchestrator = MLOrchestrator(config_path='src/config/ml/ml_config.yml')

print("\nLoading and preprocessing data...")
orchestrator.load_and_preprocess_data(sample_path)

print("\nRunning Method 2...")
try:
    results = orchestrator.run_method2_h2()
    print("✅ Method 2 completed successfully!")
    
    # Debug coefficients
    if hasattr(orchestrator.method2, 'debug_coefficients'):
        print("\n📊 Coefficient Debug Information:")
        debug_info = orchestrator.method2.debug_coefficients()
        for model, coeffs in debug_info.items():
            print(f"\n{model}:")
            if isinstance(coeffs, list):
                for coef in coeffs:
                    print(f"  - {coef}")
            else:
                print(f"  {coeffs}")
    
    # Check results
    if 'method2' in orchestrator.results:
        print("\n📈 Method 2 Results Summary:")
        method2_results = orchestrator.results['method2']
        
        # Check Model 3a results
        if 'model_3a_results' in method2_results:
            model_3a = method2_results['model_3a_results']
            print("\nModel 3a (Continuous Interaction):")
            if 'wald_test' in model_3a:
                wald = model_3a['wald_test']
                if 'error' in wald:
                    print(f"  ⚠️ Wald Test Error: {wald['error']}")
                    if 'available_coefficients' in wald:
                        print(f"  Available coefficients: {wald['available_coefficients'][:5]}...")
                else:
                    print(f"  ✓ Wald Test: significant={wald.get('significant_at_0.05', 'N/A')}")
        
        # Check Model 3b results
        if 'model_3b_results' in method2_results:
            model_3b = method2_results['model_3b_results']
            print("\nModel 3b (Categorical Interaction):")
            if 'wald_test' in model_3b:
                wald = model_3b['wald_test']
                if 'error' in wald:
                    print(f"  ⚠️ Wald Test Error: {wald['error']}")
                else:
                    print(f"  ✓ Wald Test: significant={wald.get('significant_at_0.05', 'N/A')}")
        
        # Check H2 conclusion
        if 'h2_conclusion' in method2_results:
            conclusion = method2_results['h2_conclusion']
            print(f"\n🎯 H2 Hypothesis Test:")
            print(f"  Supported: {conclusion.get('h2_supported', 'N/A')}")
            print(f"  Interpretation: {conclusion.get('interpretation', 'N/A')}")
            
except Exception as e:
    print(f"❌ Error running Method 2: {e}")
    import traceback
    traceback.print_exc()

# Generate reports
output_dir = Path('ml_results/method2_final')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📝 Generating reports in {output_dir}...")
try:
    orchestrator.generate_method_reports(output_dir)
    print("✅ Reports generated successfully!")
except Exception as e:
    print(f"❌ Error generating reports: {e}")

print("\n📁 Checking generated files...")
for file in output_dir.glob('**/*'):
    if file.is_file():
        print(f"  - {file.relative_to(output_dir)}")

h2o.cluster().shutdown()
print("\n✅ Test completed!")
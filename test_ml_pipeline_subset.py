#!/usr/bin/env python3
"""
Test ML pipeline with a subset of data to verify reporters work.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Loading a small subset of data...")
# Load only first 100k rows for testing
df = pd.read_parquet('/scratch/yl998/biodiversity_climate_production_1x_corrected.parquet')
print(f"Full data shape: {df.shape}")

# Sample 10k rows for quick testing
df_sample = df.sample(n=10000, random_state=42)
print(f"Sample data shape: {df_sample.shape}")

# Save sample
sample_path = '/tmp/biodiversity_sample_10k.parquet'
df_sample.to_parquet(sample_path)
print(f"Sample saved to: {sample_path}")

# Now run the pipeline on the sample
from src.machine_learning.ml_orchestrator import MLOrchestrator

print("\nInitializing orchestrator...")
config_path = 'src/config/ml/ml_config.yml'
orchestrator = MLOrchestrator(config_path=config_path)

print("Loading and preprocessing sample data...")
orchestrator.load_and_preprocess_data(sample_path)

print("Running Method 1...")
results = orchestrator.run_method1_h0_h1()

print("\nMethod 1 Results:")
if 'method1' in orchestrator.results:
    for key, value in orchestrator.results['method1'].items():
        print(f"  {key}: {type(value)}")

# Generate reports
output_dir = Path('ml_results/method1_test')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nGenerating reports in {output_dir}...")
orchestrator.generate_method_reports(output_dir)

print("\nChecking generated files...")
for file in output_dir.glob('*'):
    print(f"  - {file.name}")

print("\nTest completed successfully!")
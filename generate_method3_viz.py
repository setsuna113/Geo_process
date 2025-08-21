#!/usr/bin/env python
"""Generate Method 3 DML visualizations from Julia results."""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, '.')

# Import visualization functions
from src.machine_learning.visualization.dml_magnitude_plots import visualize_dml_magnitude
from src.machine_learning.visualization.bayesian_dml_plots import visualize_bayesian_dml

def main():
    """Generate Method 3 visualizations."""
    results_dir = Path('results/dml_interpretation')
    
    # Load Julia DML results
    julia_results_file = results_dir / 'method3_dml_results.json'
    if not julia_results_file.exists():
        print(f"ERROR: Julia results not found at {julia_results_file}")
        return 1
    
    with open(julia_results_file) as f:
        julia_results = json.load(f)
    
    print(f"Loaded Julia DML results with {len(julia_results.get('treatments', {}))} treatments")
    
    # Generate magnitude visualization
    print("\nGenerating magnitude analysis visualization...")
    try:
        mag_plot = visualize_dml_magnitude(julia_results, results_dir)
        print(f"  ✓ Saved to: {mag_plot}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate Bayesian DML visualization
    print("\nGenerating Bayesian DML visualization...")
    try:
        bayes_plot = visualize_bayesian_dml(julia_results, results_dir)
        print(f"  ✓ Saved to: {bayes_plot}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nVisualization generation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
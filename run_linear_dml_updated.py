#!/usr/bin/env python
"""
Run LinearDML pipeline with updated residual saving that includes coordinates
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, '/home/yl998/Geo_process')

from src.machine_learning.doubleML.refactored_pipeline import CausalEffectPipeline

def main():
    # Load config
    config_path = "/home/yl998/Geo_process/src/config/ml/econml_pipeline_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update output directory to avoid overwriting existing results
    config['output_dir'] = "/scratch/yl998/econml_results_with_coords"
    
    print("=" * 70)
    print("RUNNING LINEAR DML WITH COORDINATE SAVING")
    print("=" * 70)
    print(f"Data path: {config['data_path']}")
    print(f"Output dir: {config['output_dir']}")
    print(f"Spatial step: {config['spatial_step']}")
    print()
    
    # Initialize pipeline
    pipeline = CausalEffectPipeline(config)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    pipeline.load_data(config['data_path'], spatial_step=config['spatial_step'])
    
    # Check available columns
    print(f"\nAvailable columns: {list(pipeline.data.columns)[:20]}...")
    
    # Run homogeneous effects model
    print("\nFitting homogeneous effects model...")
    results = pipeline.fit_homogeneous()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {config['output_dir']}/homogeneous/")
    print(f"Residuals with coordinates saved to: {config['output_dir']}/homogeneous/residuals.arrow")

if __name__ == "__main__":
    main()
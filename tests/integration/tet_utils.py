# tests/integration/test_utils.py
"""Utilities for integration testing."""

import numpy as np
import xarray as xr
from pathlib import Path
import tempfile


def create_test_raster(path: Path, shape=(100, 100), bounds=(-10, -10, 10, 10), 
                      data_type='richness'):
    """Create a test raster file."""
    if data_type == 'richness':
        data = np.random.poisson(lam=5, size=shape).astype(np.float32)
    else:
        data = np.random.randn(*shape).astype(np.float32)
    
    # Create coordinates
    x = np.linspace(bounds[0], bounds[2], shape[1])
    y = np.linspace(bounds[1], bounds[3], shape[0])
    
    # Create dataset
    ds = xr.Dataset({
        'data': xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'x': x, 'y': y},
            attrs={'units': 'count' if data_type == 'richness' else 'value'}
        )
    })
    
    # Save
    ds.to_netcdf(path)
    return path


def verify_pipeline_outputs(output_dir: Path) -> bool:
    """Verify all expected pipeline outputs exist."""
    expected_files = [
        'merged_dataset.nc',
        'final_report.json',
        'README.md'
    ]
    
    for file in expected_files:
        if not (output_dir / file).exists():
            return False
    
    # Check report structure
    import json
    with open(output_dir / 'final_report.json') as f:
        report = json.load(f)
    
    required_keys = ['experiment', 'configuration', 'datasets_processed', 
                     'quality_metrics', 'outputs']
    
    return all(key in report for key in required_keys)


def setup_test_database():
    """Setup test database for integration tests."""
    # This would create a test database with required schema
    pass


def cleanup_test_database():
    """Cleanup test database after tests."""
    pass
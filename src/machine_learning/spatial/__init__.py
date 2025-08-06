"""Spatial analysis utilities for ML regression."""

from .mesf import StandardMESF, NystromMESF, compute_mesf

# Alias for backwards compatibility (will be removed in future)
compute_mesf_optimized = compute_mesf

__all__ = [
    'StandardMESF', 'NystromMESF', 'compute_mesf', 'compute_mesf_optimized'
]
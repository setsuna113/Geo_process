from src.abstractions import get_logger
"""
Statistical orchestrator - Direct method execution.

"Abstractions are for people who can't program" - Linus Torvalds
NO adapter layers, NO special cases, DIRECT calls.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.biodiversity_analysis.statistical.config_utils import load_config
from src.biodiversity_analysis.shared.data.unified_loader import get_loader
from src.biodiversity_analysis.shared.data.data_variants import DataVariants

# Import methods directly - no adapter bullshit
from src.biodiversity_analysis.statistical.correlation_analyzer import analyze_correlation
from src.biodiversity_analysis.statistical.coverage import analyze_coverage
from src.biodiversity_analysis.statistical.overlap import analyze_overlap
from src.biodiversity_analysis.statistical.latitudinal import analyze_latitudinal
from src.biodiversity_analysis.statistical.global_map import analyze_global_maps
from src.biodiversity_analysis.statistical.trust_map import analyze_trust_maps
from src.biodiversity_analysis.statistical.biome_analysis import analyze_by_biome
from src.biodiversity_analysis.statistical.partial_correlation import analyze_partial_correlations
from src.biodiversity_analysis.statistical.distribution_transformation import analyze_distribution_transformation

logger = get_logger(__name__)

# Method registry - ONE PLACE to define what methods need
METHOD_REGISTRY = {
    'correlation': {
        'function': analyze_correlation,
        'requires': ['merged'],  # Needs merged parquet
        'data_type': 'tabular'
    },
    'coverage': {
        'function': analyze_coverage,
        'requires': ['multiband'],  # Needs multiband raster
        'data_type': 'spatial'
    },
    'overlap': {
        'function': analyze_overlap,
        'requires': ['multiband'],  # Needs multiband raster
        'data_type': 'spatial'
    },
    'latitudinal': {
        'function': analyze_latitudinal,
        'requires': ['multiband'],  # Needs multiband raster
        'data_type': 'spatial'
    },
    'global_maps': {
        'function': analyze_global_maps,
        'requires': ['multiband'],  # Needs multiband raster
        'data_type': 'spatial'
    },
    'trust_maps': {
        'function': analyze_trust_maps,
        'requires': ['multiband_with_cv'],  # Needs rasters with CV
        'data_type': 'spatial'
    },
    'biome': {
        'function': analyze_by_biome,
        'requires': ['merged'],  # Needs merged parquet
        'data_type': 'tabular'
    },
    'partial_correlation': {
        'function': analyze_partial_correlations,
        'requires': ['merged'],  # Needs merged parquet
        'data_type': 'tabular'
    },
    'distribution_transformation': {
        'function': analyze_distribution_transformation,
        'requires': ['merged'],  # Needs merged parquet
        'data_type': 'tabular'
    }
}


def run_analysis(
    methods: List[str],
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    auto_variants: bool = False
) -> Dict[str, Any]:
    """
    Run statistical analysis methods.
    
    LINUS: "Talk is cheap. Show me the code."
    
    Args:
        methods: List of method names to run
        config_path: Path to config file (optional)
        output_dir: Output directory  
        config: Direct config dict (overrides file)
        auto_variants: Generate data variants automatically
        
    Returns:
        Results from all methods
    """
    # Load config - prefer direct dict over file
    if config is None and config_path is not None:
        config = load_config(config_path)
    elif config is None:
        config = {}
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(config.get('output_dir', 'outputs/statistical'))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data loader - it handles ALL data loading
    loader = get_loader()
    
    results = {}
    
    for method in methods:
        logger.info(f"Running {method}...")
        
        try:
            # Get method specification
            method_spec = METHOD_REGISTRY[method]
            
            # Add current method to config for loader
            config_with_method = config.copy() if config else {}
            config_with_method['current_method'] = method
            
            # Load data based on declared requirements - sampling happens IN THE LOADER
            data = loader.load_datasets(
                datasets=method_spec['requires'],
                data_type=method_spec['data_type'],
                config=config_with_method
            )
            
            # Check if method should handle its own variants
            if auto_variants and _supports_variants(data):
                result = _run_with_variants(method, data, output_dir, config)
            else:
                result = _run_single(method, data, output_dir, config)
            
            results[method] = result
            
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            results[method] = dict(
                success=False,
                method=method,
                error=str(e)
            )
    
    # Cleanup
    loader.clear_cache()
    
    # Log sampling info - SIMPLE
    spatial_step = config.get('spatial_grid_step', 1)
    if spatial_step > 1:
        logger.info(f"Sampling: Every {spatial_step}th row (1/{spatial_step**2} of data)")
    else:
        logger.info("Sampling: Using all data")
    
    # Summary
    _log_summary(results)
    
    return results


def _run_single(
    method: str,
    data: Any,
    output_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single method."""
    method_spec = METHOD_REGISTRY[method]
    method_func = method_spec['function']
    method_output = output_dir / method
    method_output.mkdir(parents=True, exist_ok=True)
    
    # Direct call - no adapters
    return method_func(data, method_output, **config)


def _run_with_variants(
    method: str,
    data: Any,
    output_dir: Path,  
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run method with data variants."""
    if not isinstance(data, DataVariants):
        # Not a variants object, run normally
        return _run_single(method, data, output_dir, config)
    
    results = {}
    method_spec = METHOD_REGISTRY[method]
    method_func = method_spec['function']
    
    for variant_name, variant_data in data.items():
        logger.info(f"Running {method} on variant: {variant_name}")
        variant_output = output_dir / method / variant_name
        variant_output.mkdir(parents=True, exist_ok=True)
        
        try:
            results[variant_name] = method_func(
                variant_data, variant_output, **config
            )
        except Exception as e:
            logger.error(f"Variant {variant_name} failed: {e}")
            results[variant_name] = {'error': str(e)}
    
    return {
        'success': True,
        'method': method,
        'variants': results
    }


def _supports_variants(data: Any) -> bool:
    """Check if data supports variant processing."""
    return isinstance(data, DataVariants)


def _log_summary(results: Dict[str, Any]) -> None:
    """Log analysis summary."""
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*60)
    
    for method, result in results.items():
        if isinstance(result, dict):
            if result.get('success', False):
                logger.info(f"✓ {method}: SUCCESS")
                if 'variants' in result:
                    for variant in result['variants']:
                        logger.info(f"  - {variant}: Completed")
            else:
                logger.info(f"✗ {method}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"? {method}: Unknown result type")
    
    logger.info("="*60)
from src.abstractions import get_logger
"""
Statistical Analysis Orchestrator - Linus style refactored.

"Simplicity is the ultimate sophistication."
NO adapter layers, NO special cases, DIRECT calls.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.biodiversity_analysis.statistical.config_utils import load_config
from src.biodiversity_analysis.shared.data.unified_loader import get_loader
from src.biodiversity_analysis.shared.data.data_variants import DataVariants
from src.biodiversity_analysis.shared.data.sampling_strategy import SamplingStrategy
from src.biodiversity_analysis.statistical.result_types import MethodResult

# Import methods directly - no adapter bullshit
from src.biodiversity_analysis.statistical.correlation_analyzer import analyze_correlation
from src.biodiversity_analysis.statistical.coverage import analyze_coverage
from src.biodiversity_analysis.statistical.overlap import analyze_overlap
from src.biodiversity_analysis.statistical.latitudinal import analyze_latitudinal
from src.biodiversity_analysis.statistical.global_map import analyze_global_maps
from src.biodiversity_analysis.statistical.trust_map import analyze_trust_maps
from src.biodiversity_analysis.statistical.biome_analysis import analyze_by_biome
from src.biodiversity_analysis.statistical.partial_correlation import analyze_partial_correlations

logger = get_logger(__name__)


# Method registry with data requirements - SINGLE source of truth
METHOD_REGISTRY = {
    'correlation': {
        'func': analyze_correlation,
        'data_type': 'tabular',
        'requires': ['biodiversity', 'climate']
    },
    'coverage': {
        'func': analyze_coverage,
        'data_type': 'spatial',
        'requires': ['biodiversity']
    },
    'overlap': {
        'func': analyze_overlap,
        'data_type': 'spatial',
        'requires': ['biodiversity']
    },
    'latitudinal': {
        'func': analyze_latitudinal,
        'data_type': 'spatial',
        'requires': ['biodiversity', 'coordinates']
    },
    'global_maps': {
        'func': analyze_global_maps,
        'data_type': 'spatial',
        'requires': ['biodiversity']
    },
    'trust_maps': {
        'func': analyze_trust_maps,
        'data_type': 'spatial',
        'requires': ['biodiversity', 'uncertainty']
    },
    'biome_analysis': {
        'func': analyze_by_biome,
        'data_type': 'tabular',
        'requires': ['biodiversity', 'climate', 'elevation']
    },
    'partial_correlation': {
        'func': analyze_partial_correlations,
        'data_type': 'tabular',
        'requires': ['biodiversity', 'climate']
    }
}


def run_analysis(
    config_path: str = 'src/config/stat/stat_methods.yml',
    variant: str = 'default',
    methods: Optional[List[str]] = None,
    auto_variants: bool = True
) -> Dict[str, Any]:
    """
    Run statistical analysis pipeline - SIMPLIFIED.
    
    No adapter layers, no bullshit. Load data, call method, return result.
    
    Args:
        config_path: Config file path
        variant: Data variant (ignored if auto_variants=True)
        methods: Methods to run (None = all enabled)
        auto_variants: Automatically run both fungi variants
        
    Returns:
        Results from each method
    """
    # Load configuration
    config = load_config(config_path)
    config['variant'] = variant
    
    # Setup output directory
    output_dir = _get_output_dir(config, variant)
    
    # Get methods to run
    methods = _get_methods(config, methods)
    
    # Load data manager
    loader = get_loader()
    
    # Execute pipeline - DIRECT calls
    results = {}
    
    for method in methods:
        logger.info(f"Running {method}...")
        
        try:
            # Get method specification
            method_spec = METHOD_REGISTRY[method]
            
            # Load data based on declared requirements
            data = loader.load_datasets(
                datasets=method_spec['requires'],
                data_type=method_spec['data_type'],
                config=config
            )
            data = SamplingStrategy.apply_sampling(data, config)
            
            # Check if method should handle its own variants
            if auto_variants and _supports_variants(data):
                result = _run_with_variants(method, data, output_dir, config)
            else:
                result = _run_single(method, data, output_dir, config)
            
            results[method] = result
            
        except Exception as e:
            logger.error(f"Failed {method}: {e}", exc_info=True)
            results[method] = MethodResult(
                success=False,
                method=method,
                error=str(e)
            )
    
    # Cleanup
    loader.clear_cache()
    
    # Log sampling info
    sampling_info = SamplingStrategy.get_sampling_info(config)
    logger.info(f"Sampling: {sampling_info['description']}")
    
    # Summary
    _log_summary(results)
    
    return results


def _run_single(
    method: str,
    data: Any,
    output_dir: Path,
    config: Dict[str, Any],
    add_method_dir: bool = True
) -> MethodResult:
    """Run a single method - DIRECT call, no adapter.
    
    Args:
        add_method_dir: If True, append method name to output_dir.
                       Set False when calling from _run_with_variants.
    """
    method_spec = METHOD_REGISTRY[method]
    method_func = method_spec['func']
    
    # Only add method name if requested (avoid double nesting)
    if add_method_dir:
        method_output = output_dir / method
    else:
        method_output = output_dir
        
    method_config = config['methods'].get(method, {})
    
    # ALL methods return MethodResult - NO EXCEPTIONS!
    result = method_func(data, method_output, **method_config)
    
    # Enforce type - crash if wrong
    if not isinstance(result, MethodResult):
        raise TypeError(f"{method} must return MethodResult, got {type(result)}")
    
    return result


def _run_with_variants(
    method: str,
    data: Any,
    output_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, MethodResult]:
    """Run method with auto-generated fungi variants."""
    
    # Generate variants
    variants = DataVariants.generate_fungi_variants(data)
    variant_results = {}
    
    for variant_name, variant_data in variants.items():
        logger.info(f"  Running {method} with {variant_name} fungi...")
        
        # Create variant-specific output dir
        variant_suffix = DataVariants.get_variant_suffix(variant_name)
        variant_output = output_dir / f"{method}{variant_suffix}"
        logger.debug(f"[ORCHESTRATOR] variant_output = {variant_output}")
        
        # Run method - don't add method name again (add_method_dir=False)
        result = _run_single(method, variant_data, variant_output, config, add_method_dir=False)
        variant_results[variant_name] = result
    
    return variant_results


def _supports_variants(data: Any) -> bool:
    """Check if data supports fungi variants.
    
    Data structure determines capability, not type checking.
    """
    # Try to get fungi columns - method agnostic
    try:
        if hasattr(data, 'columns'):
            # DataFrame-like
            columns = data.columns
        elif hasattr(data, 'keys'):
            # Dict-like
            columns = data.keys()
        else:
            return False
        
        # Check for both fungi types
        has_am = any('am_fungi' in str(col) for col in columns)
        has_ecm = any('ecm_fungi' in str(col) for col in columns)
        return has_am and has_ecm
        
    except Exception:
        return False


def _get_output_dir(config: Dict[str, Any], variant: str) -> Path:
    """Get output directory for this run."""
    base = Path(config['settings']['output_dir'])
    
    if variant != 'default':
        output_dir = base / f'statistical_analysis_{variant}'
    else:
        output_dir = base / 'statistical_analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _get_methods(config: Dict[str, Any], 
                methods: Optional[List[str]]) -> List[str]:
    """Determine which methods to run."""
    if methods is None:
        # Get enabled methods from config
        methods = [
            name for name, cfg in config['methods'].items()
            if cfg.get('enabled', False)
        ]
    
    # Validate
    invalid = set(methods) - set(METHOD_REGISTRY.keys())
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    
    # Single log point - no duplicate logging in methods
    logger.info(f"Will run {len(methods)} methods: {methods}")
    return methods


def _log_summary(results: Dict[str, Any]) -> None:
    """Log execution summary."""
    successful = 0
    total = 0
    
    for method, result in results.items():
        if isinstance(result, dict):
            # Variant results - Dict[str, MethodResult]
            for variant_name, variant_result in result.items():
                if not isinstance(variant_result, MethodResult):
                    raise TypeError(f"Invalid result type from {method}/{variant_name}")
                total += 1
                if variant_result.success:
                    successful += 1
                    logger.info(f"  ✓ {method} ({variant_name})")
                else:
                    logger.error(f"  ✗ {method} ({variant_name}): {variant_result.error}")
        elif isinstance(result, MethodResult):
            # Single result
            total += 1
            if result.success:
                successful += 1
                logger.info(f"  ✓ {method}")
            else:
                logger.error(f"  ✗ {method}: {result.error}")
        else:
            # CRASH - no legacy support
            raise TypeError(f"{method} returned invalid type: {type(result)}")
    
    logger.info(f"Completed: {successful}/{total} successful")
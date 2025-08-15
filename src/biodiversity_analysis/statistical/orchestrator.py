"""
Statistical Analysis Orchestrator - Final Version.

"Simplicity is the ultimate sophistication."
This is what good code looks like.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.biodiversity_analysis.statistical.config_utils import load_config
from src.biodiversity_analysis.shared.data.unified_loader import get_loader
from src.biodiversity_analysis.statistical.method_adapters import call_method

logger = logging.getLogger(__name__)


# Method registry - clean and simple
AVAILABLE_METHODS = {
    'correlation',
    'coverage',
    'overlap',
    'latitudinal',
    'global_maps',
    'trust_maps',
    'biome_analysis',
    'partial_correlation'
}


def run_analysis(
    config_path: str = 'src/config/stat/stat_methods.yml',
    variant: str = 'default',
    methods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run statistical analysis pipeline.
    
    Clean. Simple. No special cases.
    
    Args:
        config_path: Config file path
        variant: Data variant
        methods: Methods to run (None = all enabled)
        
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
    
    # Execute pipeline
    results = {}
    for method in methods:
        results[method] = _run_single_method(
            method, config, output_dir, loader
        )
    
    # Cleanup
    loader.clear_cache()
    
    # Summary
    _log_summary(results)
    
    return results


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
    invalid = set(methods) - AVAILABLE_METHODS
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    
    logger.info(f"Will run {len(methods)} methods: {methods}")
    return methods


def _run_single_method(
    method: str,
    config: Dict[str, Any],
    output_dir: Path,
    loader
) -> Dict[str, Any]:
    """Run a single analysis method."""
    try:
        logger.info(f"Running {method}...")
        
        # Load data
        data = loader.load_for_method(method, config)
        
        # Get method config
        method_config = config['methods'].get(method, {})
        
        # Get output path
        method_output = output_dir / method
        
        # Execute
        result = call_method(method, data, method_output, method_config)
        
        return {
            'success': True,
            'output': str(method_output),
            'metrics': result
        }
        
    except Exception as e:
        logger.error(f"Failed {method}: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def _log_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Log execution summary."""
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    logger.info(f"Completed: {successful}/{total} successful")
    
    # Log failures
    for method, result in results.items():
        if not result['success']:
            logger.error(f"  ✗ {method}: {result['error']}")
        else:
            logger.info(f"  ✓ {method}")


# Command line interface
def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run biodiversity statistical analysis'
    )
    
    parser.add_argument(
        '--config',
        default='src/config/stat/stat_methods.yml',
        help='Config file path'
    )
    
    parser.add_argument(
        '--variant',
        default='default',
        choices=['default', '4_features', '3_features_merged'],
        help='Data variant'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=list(AVAILABLE_METHODS),
        help='Methods to run'
    )
    
    args = parser.parse_args()
    
    results = run_analysis(
        config_path=args.config,
        variant=args.variant,
        methods=args.methods
    )
    
    # Print results
    print("\nResults:")
    for method, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {method}")
        if not result['success']:
            print(f"      Error: {result['error']}")
    
    return 0 if all(r['success'] for r in results.values()) else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
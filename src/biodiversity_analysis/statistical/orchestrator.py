"""
Ultra-simple orchestrator - Linus style.

"Theory and practice sometimes clash. Theory loses. Every single time."
No classes, no inheritance, just functions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.biodiversity_analysis.statistical.config_utils import load_config

logger = logging.getLogger(__name__)


def run_analysis(
    config_path: str = 'src/config/stat/stat_methods.yml',
    variant: str = 'default',
    methods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run statistical analysis. Dead simple.
    
    Args:
        config_path: Config file path
        variant: Data variant
        methods: Methods to run (None = all enabled)
        
    Returns:
        Results from each method
    """
    # Load config - just a dict
    config = load_config(config_path)
    
    # Output directory
    output_base = Path(config['settings']['output_dir'])
    output_dir = output_base / f'statistical_analysis_{variant}' if variant != 'default' else output_base / 'statistical_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Methods to run
    if methods is None:
        methods = [name for name, cfg in config['methods'].items() if cfg.get('enabled', False)]
    
    logger.info(f"Running {len(methods)} methods to {output_dir}")
    
    # Prepare data paths
    data_paths = config['datasets'].copy()
    
    # Handle variant
    if variant == '3_features_merged' and 'am_fungi' in data_paths and 'ecm_fungi' in data_paths:
        data_paths['total_fungi'] = {
            'am': data_paths.pop('am_fungi'),
            'ecm': data_paths.pop('ecm_fungi')
        }
    
    # Remove None values
    data_paths = {k: v for k, v in data_paths.items() if v is not None}
    
    # Import and run each method
    results = {}
    for method in methods:
        try:
            logger.info(f"Running {method}...")
            method_output = output_dir / method
            
            # Direct function import
            if method == 'correlation':
                from src.biodiversity_analysis.statistical.correlation_analyzer import analyze_correlation
                result = analyze_correlation(data_paths, method_output)
                
            elif method == 'coverage':
                from src.biodiversity_analysis.statistical.coverage import analyze_coverage
                result = analyze_coverage(data_paths, method_output)
                
            elif method == 'overlap':
                from src.biodiversity_analysis.statistical.overlap import analyze_overlap
                result = analyze_overlap(data_paths, method_output)
                
            elif method == 'latitudinal':
                from src.biodiversity_analysis.statistical.latitudinal import analyze_latitudinal
                result = analyze_latitudinal(data_paths, method_output)
                
            elif method == 'global_maps':
                from src.biodiversity_analysis.statistical.global_map import analyze_global_maps
                result = analyze_global_maps(data_paths, method_output)
                
            elif method == 'trust_maps':
                from src.biodiversity_analysis.statistical.trust_map import analyze_trust_maps
                result = analyze_trust_maps(data_paths, method_output)
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[method] = {'success': True, 'output': str(method_output)}
            
        except Exception as e:
            logger.error(f"Failed {method}: {e}")
            results[method] = {'success': False, 'error': str(e)}
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    logger.info(f"Completed: {successful}/{len(methods)} successful")
    
    return results


if __name__ == '__main__':
    # Direct execution
    import sys
    methods = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_analysis(methods=methods)
    
    for method, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {method}")
#!/usr/bin/env python
"""
Statistical Analysis Runner - Linus style.

"Simplicity is the ultimate sophistication."
One script, all variants.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biodiversity_analysis.statistical.orchestrator import run_analysis


def _print_result(name: str, result):
    """Helper to print result status uniformly."""
    if hasattr(result, 'success'):
        status = "✓" if result.success else "✗"
        print(f"  {status} {name}")
        if not result.success and hasattr(result, 'error'):
            print(f"    Error: {result.error}")
    elif isinstance(result, dict) and 'success' in result:
        # Legacy dict format
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {name}")
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    else:
        print(f"  ? {name} (unknown format)")


def main():
    """Run statistical analysis with command line args."""
    parser = argparse.ArgumentParser(
        description='Run biodiversity statistical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods with 4 features (fungi separate)
  %(prog)s --variant 4_features
  
  # Run all methods with 3 features (fungi merged)
  %(prog)s --variant 3_features_merged
  
  # Run specific methods only
  %(prog)s --methods correlation coverage
  
  # Run both variants
  %(prog)s --both-variants
        """
    )
    
    parser.add_argument(
        '--variant',
        default='4_features',
        choices=['4_features', '3_features_merged', 'default'],
        help='Data variant to use (default: 4_features)'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['correlation', 'coverage', 'overlap', 'latitudinal', 'global_maps', 'trust_maps', 'biome_analysis', 'partial_correlation'],
        help='Specific methods to run (default: all enabled in config)'
    )
    
    parser.add_argument(
        '--config',
        default='src/config/stat/stat_methods.yml',
        help='Config file path'
    )
    
    parser.add_argument(
        '--both-variants',
        action='store_true',
        help='Run both 4_features and 3_features_merged (legacy)'
    )
    
    parser.add_argument(
        '--auto-variants',
        action='store_true',
        default=True,
        help='Automatically generate fungi variants (default: True)'
    )
    
    parser.add_argument(
        '--no-auto-variants',
        dest='auto_variants',
        action='store_false',
        help='Disable automatic variant generation'
    )
    
    args = parser.parse_args()
    
    # Run both variants if requested
    if args.both_variants:
        print("=" * 60)
        print("Running 4 Features Analysis (AM/ECM Separate)")
        print("=" * 60)
        
        results_4 = run_analysis(
            methods=args.methods if args.methods else [],
            config_path=args.config,
            auto_variants=False  # Legacy mode
        )
        
        print("\nResults (4 features):")
        for method, result in results_4.items():
            _print_result(method, result)
        
        print("\n" + "=" * 60)
        print("Running 3 Features Analysis (Fungi Merged)")
        print("=" * 60)
        
        results_3 = run_analysis(
            methods=args.methods if args.methods else [],
            config_path=args.config,
            auto_variants=False  # Legacy mode
        )
        
        print("\nResults (3 features):")
        for method, result in results_3.items():
            _print_result(method, result)
    
    else:
        # Run single variant
        print(f"Running analysis: variant={args.variant}")
        results = run_analysis(
            methods=args.methods if args.methods else [],
            config_path=args.config,
            auto_variants=args.auto_variants
        )
        
        print("\nResults:")
        for method, result in results.items():
            # Simplified handling - treat everything uniformly
            if isinstance(result, dict) and not hasattr(result, 'success'):
                # Has variants (dict of MethodResults)
                for variant_name, variant_result in result.items():
                    _print_result(f"{method} ({variant_name})", variant_result)
            else:
                # Single result
                _print_result(method, result)
    
    print("\nOutput directory: outputs/biodiversity_analysis/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
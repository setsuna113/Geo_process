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

from src.biodiversity_analysis.statistical import run_analysis


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
        choices=['correlation', 'coverage', 'overlap', 'latitudinal', 'global_maps', 'trust_maps'],
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
        help='Run both 4_features and 3_features_merged'
    )
    
    args = parser.parse_args()
    
    # Run both variants if requested
    if args.both_variants:
        print("=" * 60)
        print("Running 4 Features Analysis (AM/ECM Separate)")
        print("=" * 60)
        
        results_4 = run_analysis(
            config_path=args.config,
            variant='4_features',
            methods=args.methods
        )
        
        print("\nResults (4 features):")
        for method, result in results_4.items():
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {method}")
        
        print("\n" + "=" * 60)
        print("Running 3 Features Analysis (Fungi Merged)")
        print("=" * 60)
        
        results_3 = run_analysis(
            config_path=args.config,
            variant='3_features_merged',
            methods=args.methods
        )
        
        print("\nResults (3 features):")
        for method, result in results_3.items():
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {method}")
    
    else:
        # Run single variant
        print(f"Running analysis: variant={args.variant}")
        results = run_analysis(
            config_path=args.config,
            variant=args.variant,
            methods=args.methods
        )
        
        print("\nResults:")
        for method, result in results.items():
            status = "✓" if result['success'] else "✗"
            print(f"  {status} {method}")
            if not result['success']:
                print(f"    Error: {result['error']}")
    
    print("\nOutput directory: outputs/biodiversity_analysis/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
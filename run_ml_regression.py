#!/usr/bin/env python
"""Run biodiversity regression analysis.

Main entry point for hypothesis testing using GLM and DML methods.
Based on regression_analysis_decisions.md.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.machine_learning.pipelines import RegressionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('regression_analysis.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function for running regression analysis."""
    parser = argparse.ArgumentParser(
        description='Run biodiversity regression analysis for hypothesis testing'
    )
    
    # Required arguments
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to biodiversity data (parquet, csv, or pickle)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/ml/ml_config.yml',
        help='Path to configuration file (default: config/ml/ml_config.yml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/regression_results',
        help='Directory to save results (default: outputs/regression_results)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['all', 'glm-continuous', 'glm-categorical', 'dml'],
        default='all',
        help='Which method(s) to run (default: all)'
    )
    
    parser.add_argument(
        '--diagnostics-only',
        action='store_true',
        help='Run only diagnostic checks without full analysis'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("="*70)
        logger.info("BIODIVERSITY REGRESSION ANALYSIS")
        logger.info("="*70)
        logger.info(f"Data: {args.data_path}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Output: {args.output_dir}")
        logger.info(f"Method: {args.method}")
        
        # Initialize pipeline
        pipeline = RegressionPipeline(config_path=args.config)
        
        # Load data
        pipeline.load_data(args.data_path)
        
        if args.diagnostics_only:
            # Run diagnostics only
            diagnostics = pipeline.run_diagnostics_only()
            
            logger.info("\nDiagnostic Results:")
            logger.info(f"  Dispersion ratio: {diagnostics['dispersion_ratio']:.2f}")
            logger.info(f"  Recommended distribution: {diagnostics['recommended_distribution']}")
            logger.info(f"  Complete cases: {diagnostics['n_complete_cases']}")
            logger.info(f"  Dropped cases: {diagnostics['n_dropped']}")
            
            # Save diagnostics
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / 'diagnostics.json', 'w') as f:
                json.dump(diagnostics, f, indent=2)
            
        else:
            # Run full analysis
            results = pipeline.run_analysis()
            
            # Save results
            if not args.no_viz:
                pipeline.save_results(args.output_dir)
            else:
                # Save without visualizations
                output_path = Path(args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                pipeline.hypothesis_tester.save_results(
                    str(output_path / 'regression_results.json')
                )
            
            # Print summary
            if 'summary' in results:
                print("\n" + "="*60)
                print("ANALYSIS COMPLETE")
                print("="*60)
                
                summary = results['summary']
                
                # H2 consensus
                if 'h2' in summary:
                    print(f"\nH2 Temperate Mismatch Hypothesis:")
                    print(f"  Consensus: {summary['h2']['consensus']}")
                    print(f"  {summary['h2']['interpretation']}")
                
                # Conclusions
                if 'conclusions' in summary:
                    print("\nKey Findings:")
                    for conclusion in summary['conclusions']:
                        print(f"  • {conclusion}")
                
                print(f"\nResults saved to: {args.output_dir}")
        
        logger.info("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
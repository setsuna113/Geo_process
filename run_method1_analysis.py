#!/usr/bin/env python3
"""Run Method 1 analysis using Julia results from scratch directory."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.machine_learning.ml_orchestrator import MLOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run Method 1 analysis with Julia results."""
    
    # Initialize orchestrator with config path
    logger.info("Initializing ML Orchestrator...")
    config_path = Path("src/config/ml/ml_config.yml")
    orchestrator = MLOrchestrator(config_path=config_path)
    
    # Load Julia results from scratch directory
    # Use the file that contains Method 1 results
    julia_results_file = "/scratch/yl998/julia/julia_results_2025-08-20_01-48-39.json"
    
    logger.info(f"Loading Julia results from {julia_results_file}")
    results = orchestrator.load_julia_results(julia_results_file)
    
    # Run hypothesis tests (Method 1 only)
    logger.info("Running Method 1 hypothesis tests...")
    test_results = orchestrator.run_hypothesis_tests()
    
    # Generate visualizations and reports
    logger.info("Generating visualizations and reports...")
    output_dir = Path("results/method1_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations (hypothesis tests + diagnostics)
    if orchestrator.method1:
        viz_files = orchestrator.method1.generate_visualizations(output_dir)
        logger.info(f"Generated {len(viz_files)} visualization files")
        
        # Generate comprehensive report
        report = orchestrator.method1.generate_report(output_dir)
        logger.info("Generated comprehensive report")
        
        # Print summary
        print("\n" + "="*70)
        print("METHOD 1 ANALYSIS COMPLETE")
        print("="*70)
        
        if 'conclusion' in test_results.get('method1', {}):
            conclusion = test_results['method1']['conclusion']
            print(f"\nDECISION: {conclusion.get('hypothesis', 'Unknown')}")
            print(f"INTERPRETATION: {conclusion.get('interpretation', '')}")
            
            print("\nEVIDENCE SUMMARY:")
            for test_name, supports_h1 in conclusion.get('evidence', {}).items():
                result = 'Supports H1' if supports_h1 else 'Supports H0'
                print(f"  {test_name.upper()}: {result}")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Visualizations: {len(viz_files)} files")
        print(f"  - Professional report: method1_professional_report.txt")
        print(f"  - LaTeX tables: method1_latex_tables.tex")
        print(f"  - Structured results: method1_results.json")
        
    else:
        logger.error("Method 1 not properly initialized")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
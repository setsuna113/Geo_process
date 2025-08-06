#!/usr/bin/env python
"""
Quick test of the regression pipeline focusing on Method 1.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/yl998/dev/geo')

from src.machine_learning.hypothesis import HypothesisTester
from src.machine_learning.config_loader import MLRegressionConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(n_samples=100):  # Reduced samples for speed
    """Create realistic test data for regression analysis."""
    np.random.seed(42)
    
    # Create spatial coordinates
    lon = np.random.uniform(-180, 180, n_samples)
    lat = np.random.uniform(-60, 60, n_samples)
    
    # Create climate variables
    bio1 = np.random.normal(15, 5, n_samples)  # Mean temperature
    bio4 = np.random.normal(700, 200, n_samples)  # Temperature seasonality
    bio12 = np.random.normal(1000, 300, n_samples)  # Annual precipitation
    bio15 = np.random.normal(50, 15, n_samples)  # Precipitation seasonality
    elevation = np.random.normal(500, 200, n_samples)
    elevation = np.clip(elevation, 0, None)  # No negative elevation
    
    # Create biotic predictors with some correlation
    plants_richness = np.random.poisson(50 + 0.3 * bio12 / 10, n_samples)
    vertebrates_richness = np.random.poisson(30 + 0.2 * plants_richness, n_samples)
    
    # Create fungi richness with H2 temperate mismatch pattern
    # Base relationship with plants
    fungi_base = 20 + 0.4 * plants_richness + np.random.normal(0, 5, n_samples)
    
    # Add temperate mismatch effect (negative interaction)
    temperate = (bio4 > np.percentile(bio4, 75)).astype(float)
    plant_high = (plants_richness > np.percentile(plants_richness, 75)).astype(float)
    
    # H2 effect: reduced fungi in temperate regions with high plants
    h2_effect = -10 * temperate * plant_high
    
    fungi_richness = np.maximum(1, fungi_base + h2_effect + 
                                0.1 * vertebrates_richness +
                                0.05 * elevation / 100 +
                                np.random.poisson(5, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'longitude': lon,
        'latitude': lat,
        'bio1': bio1,
        'bio4': bio4,
        'bio12': bio12,
        'bio15': bio15,
        'elevation': elevation,
        'plants_richness': plants_richness.astype(int),
        'vertebrates_richness': vertebrates_richness.astype(int),
        'fungi_richness': fungi_richness.astype(int)
    })
    
    logger.info(f"Created test data with {n_samples} samples")
    logger.info(f"  Fungi richness: mean={fungi_richness.mean():.1f}, std={fungi_richness.std():.1f}")
    logger.info(f"  Plants richness: mean={plants_richness.mean():.1f}, std={plants_richness.std():.1f}")
    
    return data


def test_method1_visualizations(data, config):
    """Test Method 1 with focus on visualizations."""
    logger.info("\n" + "="*80)
    logger.info("TESTING METHOD 1: CONTINUOUS INTERACTION GLM WITH VISUALIZATIONS")
    logger.info("="*80)
    
    # Initialize hypothesis tester
    tester = HypothesisTester(config)
    
    # Prepare data
    logger.info("Preparing data...")
    tester.prepare_data(data)
    
    # Run Method 1
    logger.info("Running Method 1...")
    tester.results['method1'] = {}
    tester.results['method1']['models'] = tester.method1.fit_models()
    
    # Get interaction results
    interaction = tester.method1.get_interaction_coefficient()
    tester.results['method1']['h2_test'] = interaction
    tester.results['method1']['interaction_term'] = interaction  # Add this for visualization
    
    logger.info(f"Interaction coefficient: {interaction['coefficient']:.4f}")
    logger.info(f"P-value: {interaction['p_value']:.4f}")
    logger.info(f"Interpretation: {interaction['interpretation']}")
    
    # Run diagnostics
    logger.info("\nRunning diagnostics...")
    diagnostics = tester.method1.run_diagnostics()
    tester.results['method1']['diagnostics'] = diagnostics
    
    if 'residual_spatial_autocorrelation' in diagnostics:
        resid_spatial = diagnostics['residual_spatial_autocorrelation']
        logger.info(f"Post-MESF Moran's I: {resid_spatial['morans_i']:.4f}")
        logger.info(f"P-value: {resid_spatial['p_value']:.4f}")
    
    # Get summary for visualization
    tester.results['method1']['summary'] = tester.method1.summarize_results()
    
    # Create visualization
    logger.info("\nCreating Method 1 visualization...")
    viz_path = Path('outputs/regression_test_results/method1_quick_test.png')
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        tester.method1.create_visualization(save_path=str(viz_path))
        logger.info(f"✅ Visualization saved to {viz_path}")
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Create report
    logger.info("\nGenerating report...")
    report_dir = Path('outputs/regression_test_results/method1_report')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from src.machine_learning.reporting import RegressionReportGenerator
        
        # Prepare results for report generator
        all_results = {
            'method1': tester.results['method1'],
            'method2': {},
            'method3': {},
            'synthesis': {
                'h2': {
                    'method1_supported': interaction['supports_h2'],
                    'method2_supported': False,
                    'method3_supported': False,
                    'consensus': False,
                    'interpretation': interaction['interpretation']
                }
            }
        }
        
        viz_paths = {'method1': str(viz_path)}
        
        report_gen = RegressionReportGenerator(all_results, viz_paths)
        report_gen.generate_full_report(str(report_dir))
        
        logger.info(f"✅ Report generated in {report_dir}")
        
        # List generated files
        logger.info("\nGenerated files:")
        for file_path in report_dir.rglob('*'):
            if file_path.is_file():
                logger.info(f"  - {file_path.relative_to(report_dir)}")
                
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return tester.results['method1']


def main():
    """Run quick test."""
    # Load configuration
    logger.info("Loading configuration...")
    config = MLRegressionConfig('config/ml/ml_config.yml')
    
    # Create test data (smaller dataset for speed)
    logger.info("Creating test data...")
    data = create_test_data(n_samples=100)
    
    # Test Method 1 with visualizations
    results = test_method1_visualizations(data, config)
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETED")
    logger.info("="*80)
    
    if results:
        logger.info("\n✅ Method 1 test successful!")
        logger.info("Check outputs/regression_test_results/ for:")
        logger.info("  - method1_quick_test.png (visualization)")
        logger.info("  - method1_report/ (full report)")
    else:
        logger.info("\n❌ Test failed")


if __name__ == "__main__":
    main()
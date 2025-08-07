#!/usr/bin/env python
"""
Quick test of Methods 2 and 3 in the regression pipeline.
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


def create_test_data(n_samples=100):
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


def test_method2(data, config):
    """Test Method 2 (Categorical GLM) with visualizations."""
    logger.info("\n" + "="*80)
    logger.info("TESTING METHOD 2: CATEGORICAL GLM WITH EMMEANS")
    logger.info("="*80)
    
    # Initialize hypothesis tester
    tester = HypothesisTester(config)
    
    # Prepare data
    logger.info("Preparing data...")
    tester.prepare_data(data)
    
    # Run Method 2
    logger.info("Running Method 2...")
    tester.results['method2'] = {}
    tester.results['method2']['model'] = tester.method2.fit_model()
    
    # Get emmeans results
    logger.info("\nComputing estimated marginal means...")
    emmeans_results = tester.method2.compute_emmeans()
    tester.results['method2']['emmeans'] = emmeans_results
    tester.results['method2']['marginal_means'] = emmeans_results.get('marginal_means', {})
    
    for group, stats in emmeans_results.get('marginal_means', {}).items():
        logger.info(f"  {group}: {stats['mean']:.2f} ± {stats['se']:.2f}")
    
    # Test H2 contrast
    logger.info("\nTesting H2 contrast...")
    h2_results = tester.method2.test_h2_contrast()
    tester.results['method2']['h2_test'] = h2_results
    tester.results['method2']['h2_contrast'] = h2_results  # Add for visualization
    
    logger.info(f"Contrast estimate: {h2_results.get('contrast_estimate', 0):.4f}")
    logger.info(f"P-value: {h2_results.get('p_value', 1):.4f}")
    logger.info(f"Interpretation: {h2_results.get('interpretation', 'N/A')}")
    
    # Run diagnostics
    logger.info("\nRunning diagnostics...")
    diagnostics = tester.method2.run_diagnostics()
    tester.results['method2']['diagnostics'] = diagnostics
    
    if 'residual_spatial_autocorrelation' in diagnostics:
        resid_spatial = diagnostics['residual_spatial_autocorrelation']
        logger.info(f"Post-MESF Moran's I: {resid_spatial['morans_i']:.4f}")
        logger.info(f"P-value: {resid_spatial['p_value']:.4f}")
    
    # Create 4-panel visualization
    logger.info("\nCreating Method 2 visualization...")
    viz_path = Path('outputs/regression_test_results/method2_h2_mismatch.png')
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        tester.method2.create_visualization(save_path=str(viz_path))
        logger.info(f"✅ Visualization saved to {viz_path}")
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    return tester.results['method2']


def test_method3(data, config):
    """Test Method 3 (DML) with visualizations."""
    logger.info("\n" + "="*80)
    logger.info("TESTING METHOD 3: DOUBLE MACHINE LEARNING")
    logger.info("="*80)
    
    # Initialize hypothesis tester
    tester = HypothesisTester(config)
    
    # Prepare data
    logger.info("Preparing data...")
    tester.prepare_data(data)
    
    # Run Method 3 - reduce bootstrap iterations for speed
    logger.info("Running Method 3 (reduced bootstrap for speed)...")
    tester.method3.config.bootstrap_iterations = 10  # Reduce from 100 for speed
    
    tester.results['method3'] = {}
    hypothesis_results = tester.method3.test_all_hypotheses()
    tester.results['method3']['hypotheses'] = hypothesis_results
    
    # Add h2 specifically for visualization
    if 'h2' in hypothesis_results:
        tester.results['method3']['h2'] = hypothesis_results['h2']
    
    # Display results for each hypothesis
    for hyp in ['h0', 'h1', 'h2']:
        if hyp in hypothesis_results:
            result = hypothesis_results[hyp]
            logger.info(f"\n{hyp.upper()} Results:")
            logger.info(f"  Causal effect: {result['causal_effect']:.4f}")
            logger.info(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            logger.info(f"  P-value: {result['p_value']:.4f}")
            logger.info(f"  Interpretation: {result['interpretation']}")
    
    # Create DML effects visualization
    logger.info("\nCreating Method 3 visualization...")
    viz_path = Path('outputs/regression_test_results/method3_dml_effects.png')
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        tester.method3.create_visualization(save_path=str(viz_path))
        logger.info(f"✅ Visualization saved to {viz_path}")
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    return tester.results['method3']


def main():
    """Run tests for Methods 2 and 3."""
    # Load configuration
    logger.info("Loading configuration...")
    config = MLRegressionConfig('config/ml/ml_config.yml')
    
    # Create test data
    logger.info("Creating test data...")
    data = create_test_data(n_samples=100)
    
    # Test Method 2
    logger.info("\n" + "#"*80)
    logger.info("# TEST 1: METHOD 2 (CATEGORICAL GLM)")
    logger.info("#"*80)
    try:
        method2_results = test_method2(data, config)
        logger.info("✅ Method 2 test completed successfully")
    except Exception as e:
        logger.error(f"❌ Method 2 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Method 3
    logger.info("\n" + "#"*80)
    logger.info("# TEST 2: METHOD 3 (DML)")
    logger.info("#"*80)
    try:
        method3_results = test_method3(data, config)
        logger.info("✅ Method 3 test completed successfully")
    except Exception as e:
        logger.error(f"❌ Method 3 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info("ALL TESTS COMPLETED")
    logger.info("="*80)
    logger.info("Check outputs/regression_test_results/ for:")
    logger.info("  - method2_h2_mismatch.png (Method 2 4-panel visualization)")
    logger.info("  - method3_dml_effects.png (Method 3 causal effects)")


if __name__ == "__main__":
    main()
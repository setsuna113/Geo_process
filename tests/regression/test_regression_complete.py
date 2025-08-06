#!/usr/bin/env python
"""
Comprehensive test of the regression pipeline with all fixes:
- Quadratic elevation terms
- Post-model Moran's I checks
- Full visualization panels
- Complete reporting
"""

import pandas as pd
import numpy as np
import logging
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/yl998/dev/geo')

from src.machine_learning.hypothesis import HypothesisTester
from src.machine_learning.config_loader import MLRegressionConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(n_samples=500):
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
    logger.info(f"  Temperate regions: {temperate.sum()} ({temperate.mean()*100:.1f}%)")
    logger.info(f"  High plant regions: {plant_high.sum()} ({plant_high.mean()*100:.1f}%)")
    
    return data


def test_method1_only(data, config, output_dir):
    """Test Method 1 (Continuous GLM) only."""
    logger.info("\n" + "="*80)
    logger.info("TESTING METHOD 1: CONTINUOUS INTERACTION GLM")
    logger.info("="*80)
    
    # Initialize hypothesis tester
    tester = HypothesisTester(config)
    
    # Prepare data (this method takes the data as argument)
    logger.info("Preparing data with elevation squared term...")
    tester.prepare_data(data)
    
    # Run only Method 1
    logger.info("Running Method 1...")
    tester.results['method1'] = {}
    tester.results['method1']['models'] = tester.method1.fit_models()
    
    # Get interaction results
    interaction = tester.method1.get_interaction_coefficient()
    tester.results['method1']['h2_test'] = interaction
    logger.info(f"Interaction coefficient: {interaction['coefficient']:.4f}")
    logger.info(f"P-value: {interaction['p_value']:.4f}")
    logger.info(f"Interpretation: {interaction['interpretation']}")
    
    # Run diagnostics (includes post-MESF Moran's I)
    logger.info("\nRunning diagnostics with post-MESF check...")
    diagnostics = tester.method1.run_diagnostics()
    tester.results['method1']['diagnostics'] = diagnostics
    
    if 'residual_spatial_autocorrelation' in diagnostics:
        resid_spatial = diagnostics['residual_spatial_autocorrelation']
        logger.info(f"Post-MESF Moran's I: {resid_spatial['morans_i']:.4f}")
        logger.info(f"P-value: {resid_spatial['p_value']:.4f}")
        logger.info(f"Interpretation: {resid_spatial['interpretation']}")
    
    # Create visualization
    logger.info("\nCreating Method 1 visualization...")
    viz_path = Path(output_dir) / 'method1_only' / 'method1_interaction.png'
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    tester.method1.create_visualization(save_path=str(viz_path))
    logger.info(f"Saved visualization to {viz_path}")
    
    # Save results
    results_path = Path(output_dir) / 'method1_only' / 'results.json'
    results = tester.method1.summarize_results()
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")
    
    return results


def test_method2_only(data, config, output_dir):
    """Test Method 2 (Categorical GLM) only."""
    logger.info("\n" + "="*80)
    logger.info("TESTING METHOD 2: CATEGORICAL GLM WITH EMMEANS")
    logger.info("="*80)
    
    # Initialize hypothesis tester
    tester = HypothesisTester(config)
    
    # Prepare data
    logger.info("Preparing data with categorical thresholds (75th percentile)...")
    tester.prepare_data(data)
    
    # Run only Method 2
    logger.info("Running Method 2...")
    tester.results['method2'] = {}
    tester.results['method2']['model'] = tester.method2.fit_model()
    
    # Get emmeans results
    logger.info("\nComputing estimated marginal means...")
    emmeans_results = tester.method2.compute_emmeans()
    tester.results['method2']['emmeans'] = emmeans_results
    for group, stats in emmeans_results['marginal_means'].items():
        logger.info(f"  {group}: {stats['mean']:.2f} ± {stats['se']:.2f}")
    
    # Test H2 contrast
    logger.info("\nTesting H2 contrast...")
    h2_results = tester.method2.test_h2_contrast()
    tester.results['method2']['h2_test'] = h2_results
    logger.info(f"Contrast estimate: {h2_results['contrast_estimate']:.4f}")
    logger.info(f"P-value: {h2_results['p_value']:.4f}")
    logger.info(f"Interpretation: {h2_results['interpretation']}")
    
    # Run diagnostics
    logger.info("\nRunning diagnostics with post-MESF check...")
    diagnostics = tester.method2.run_diagnostics()
    tester.results['method2']['diagnostics'] = diagnostics
    
    if 'residual_spatial_autocorrelation' in diagnostics:
        resid_spatial = diagnostics['residual_spatial_autocorrelation']
        logger.info(f"Post-MESF Moran's I: {resid_spatial['morans_i']:.4f}")
        logger.info(f"P-value: {resid_spatial['p_value']:.4f}")
        logger.info(f"Interpretation: {resid_spatial['interpretation']}")
    
    # Create 4-panel visualization
    logger.info("\nCreating Method 2 4-panel visualization...")
    viz_path = Path(output_dir) / 'method2_only' / 'method2_h2_mismatch.png'
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    tester.method2.create_visualization(save_path=str(viz_path))
    logger.info(f"Saved 4-panel visualization to {viz_path}")
    
    # Save results
    results_path = Path(output_dir) / 'method2_only' / 'results.json'
    results = tester.method2.summarize_results()
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")
    
    return results


def test_method3_only(data, config, output_dir):
    """Test Method 3 (DML) only."""
    logger.info("\n" + "="*80)
    logger.info("TESTING METHOD 3: DOUBLE MACHINE LEARNING")
    logger.info("="*80)
    
    # Initialize hypothesis tester
    tester = HypothesisTester(config)
    
    # Prepare data
    logger.info("Preparing data for DML...")
    tester.prepare_data(data)
    
    # Run only Method 3
    logger.info("Running Method 3 for all hypotheses...")
    tester.results['method3'] = {}
    hypothesis_results = tester.method3.test_all_hypotheses()
    tester.results['method3']['hypotheses'] = hypothesis_results
    
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
    logger.info("\nCreating Method 3 causal effects visualization...")
    viz_path = Path(output_dir) / 'method3_only' / 'method3_dml_effects.png'
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    tester.method3.create_visualization(save_path=str(viz_path))
    logger.info(f"Saved visualization to {viz_path}")
    
    # Save results
    results_path = Path(output_dir) / 'method3_only' / 'results.json'
    results = tester.method3.summarize_results()
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")
    
    return results


def test_full_pipeline(data, config, output_dir):
    """Test full pipeline with all methods and synthesis."""
    logger.info("\n" + "="*80)
    logger.info("TESTING FULL PIPELINE: ALL METHODS WITH SYNTHESIS")
    logger.info("="*80)
    
    # Initialize hypothesis tester
    tester = HypothesisTester(config)
    
    # Run complete analysis (this includes prepare_data)
    logger.info("Running complete analysis with all three methods...")
    results = tester.run_all_tests(data)
    
    # Get synthesis
    logger.info("\nSynthesizing results across methods...")
    synthesis = tester.synthesize_results()
    
    # Display H2 consensus
    h2_synthesis = synthesis.get('h2', {})
    logger.info("\nH2 Temperate Mismatch Hypothesis:")
    logger.info(f"  Method 1 supports: {h2_synthesis.get('method1_supported', False)}")
    logger.info(f"  Method 2 supports: {h2_synthesis.get('method2_supported', False)}")
    logger.info(f"  Method 3 supports: {h2_synthesis.get('method3_supported', False)}")
    logger.info(f"  Consensus: {h2_synthesis.get('consensus', 'Not tested')}")
    logger.info(f"  {h2_synthesis.get('interpretation', '')}")
    
    # Generate full report with all visualizations
    logger.info("\nGenerating comprehensive report...")
    report_dir = Path(output_dir) / 'full_pipeline'
    tester.generate_full_report(str(report_dir))
    
    # List generated files
    logger.info("\nGenerated files:")
    for file_path in report_dir.rglob('*'):
        if file_path.is_file():
            logger.info(f"  - {file_path.relative_to(report_dir)}")
    
    return synthesis


def main():
    """Run all tests."""
    # Set up output directory
    output_dir = Path('outputs/regression_test_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = MLRegressionConfig('config/ml/ml_config.yml')
    
    # Create test data
    logger.info("Creating test data with H2 mismatch pattern...")
    data = create_test_data(n_samples=500)
    
    # Save test data
    data_path = output_dir / 'test_data.csv'
    data.to_csv(data_path, index=False)
    logger.info(f"Saved test data to {data_path}")
    
    try:
        # Test Method 1 only
        logger.info("\n" + "#"*80)
        logger.info("# TEST 1: METHOD 1 (CONTINUOUS GLM) ONLY")
        logger.info("#"*80)
        method1_results = test_method1_only(data, config, output_dir)
        logger.info("✅ Method 1 test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Method 1 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test Method 2 only
        logger.info("\n" + "#"*80)
        logger.info("# TEST 2: METHOD 2 (CATEGORICAL GLM) ONLY")
        logger.info("#"*80)
        method2_results = test_method2_only(data, config, output_dir)
        logger.info("✅ Method 2 test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Method 2 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test Method 3 only
        logger.info("\n" + "#"*80)
        logger.info("# TEST 3: METHOD 3 (DML) ONLY")
        logger.info("#"*80)
        method3_results = test_method3_only(data, config, output_dir)
        logger.info("✅ Method 3 test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Method 3 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test full pipeline
        logger.info("\n" + "#"*80)
        logger.info("# TEST 4: FULL PIPELINE WITH ALL METHODS")
        logger.info("#"*80)
        synthesis = test_full_pipeline(data, config, output_dir)
        logger.info("✅ Full pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info("ALL TESTS COMPLETED")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
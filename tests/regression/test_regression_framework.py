#!/usr/bin/env python
"""Test script for regression framework.

Creates synthetic data and tests all components.
"""

import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_data(n_samples=1000, seed=42):
    """Create synthetic biodiversity data for testing.
    
    Creates data with known relationships to test hypotheses:
    - H1: Positive relationship between plants and fungi
    - H2: Relationship weakens in temperate regions (high bio4)
    """
    np.random.seed(seed)
    logger.info(f"Creating synthetic data with {n_samples} samples")
    
    # Geographic coordinates (global coverage)
    longitude = np.random.uniform(-180, 180, n_samples)
    latitude = np.random.uniform(-60, 70, n_samples)
    
    # Climate variables
    bio1 = np.random.normal(15, 8, n_samples)  # Mean annual temperature
    bio4 = np.random.exponential(500, n_samples) + 100  # Temperature seasonality
    bio12 = np.random.exponential(800, n_samples) + 200  # Annual precipitation
    bio15 = np.random.uniform(10, 100, n_samples)  # Precipitation seasonality
    elevation = np.random.exponential(300, n_samples)
    
    # Normalize bio4 to create temperate indicator
    bio4_normalized = (bio4 - bio4.min()) / (bio4.max() - bio4.min())
    is_temperate = bio4_normalized > 0.7  # Top 30% are temperate
    
    # Create plant richness (influenced by climate)
    plants_effect = (
        0.5 * bio1 +  # Temperature effect
        0.3 * np.log(bio12 + 1) +  # Precipitation effect
        -0.2 * elevation / 1000 +  # Elevation effect
        np.random.normal(0, 5, n_samples)  # Noise
    )
    plants_richness = np.maximum(1, np.round(np.exp(plants_effect / 10))).astype(int)
    
    # Create vertebrate richness (similar to plants but independent)
    vertebrates_effect = (
        0.4 * bio1 +
        0.2 * np.log(bio12 + 1) +
        -0.3 * elevation / 1000 +
        np.random.normal(0, 5, n_samples)
    )
    vertebrates_richness = np.maximum(1, np.round(np.exp(vertebrates_effect / 10))).astype(int)
    
    # Create fungi richness with H1 and H2 relationships
    # Base effect from climate
    fungi_base = (
        0.3 * bio1 +
        0.4 * np.log(bio12 + 1) +
        -0.1 * elevation / 1000
    )
    
    # H1: Positive effect of plant richness
    plant_effect_on_fungi = 0.4 * np.log(plants_richness + 1)
    
    # H2: Effect weakens in temperate regions (interaction)
    temperate_modifier = np.where(is_temperate, -0.3, 0.1)
    interaction_effect = plant_effect_on_fungi * (1 + temperate_modifier)
    
    # Combine effects
    fungi_effect = (
        fungi_base +
        interaction_effect +
        0.1 * np.log(vertebrates_richness + 1) +  # Small vertebrate effect
        np.random.normal(0, 3, n_samples)  # Noise
    )
    
    # Convert to counts (Poisson-like but with overdispersion)
    fungi_lambda = np.exp(fungi_effect / 8)
    # Add overdispersion using negative binomial
    fungi_richness = np.random.negative_binomial(
        n=fungi_lambda / 2,  # Dispersion parameter
        p=0.5,  # Success probability
        size=n_samples
    )
    fungi_richness = np.maximum(1, fungi_richness)
    
    # Create DataFrame
    data = pd.DataFrame({
        'longitude': longitude,
        'latitude': latitude,
        'bio1': bio1,
        'bio4': bio4,
        'bio12': bio12,
        'bio15': bio15,
        'elevation': elevation,
        'plants_richness': plants_richness,
        'vertebrates_richness': vertebrates_richness,
        'fungi_richness': fungi_richness,
        'is_temperate': is_temperate
    })
    
    # Log summary statistics
    logger.info("\nData Summary:")
    logger.info(f"  Fungi richness: mean={fungi_richness.mean():.1f}, var={fungi_richness.var():.1f}")
    logger.info(f"  Dispersion ratio: {fungi_richness.var() / fungi_richness.mean():.2f}")
    logger.info(f"  Plants richness: mean={plants_richness.mean():.1f}")
    logger.info(f"  Temperate regions: {is_temperate.sum()} ({is_temperate.mean()*100:.1f}%)")
    logger.info(f"  Correlation (plants, fungi): {data['plants_richness'].corr(data['fungi_richness']):.3f}")
    
    # Check relationships
    temperate_data = data[data['is_temperate']]
    non_temperate_data = data[~data['is_temperate']]
    
    logger.info("\nExpected relationships:")
    logger.info(f"  Plants-fungi correlation (non-temperate): "
                f"{non_temperate_data['plants_richness'].corr(non_temperate_data['fungi_richness']):.3f}")
    logger.info(f"  Plants-fungi correlation (temperate): "
                f"{temperate_data['plants_richness'].corr(temperate_data['fungi_richness']):.3f}")
    
    return data


def test_imports():
    """Test that all modules can be imported."""
    logger.info("\n" + "="*60)
    logger.info("Testing imports...")
    logger.info("="*60)
    
    try:
        # Config
        from src.machine_learning.config_loader import MLRegressionConfig
        logger.info("✓ Config loader imported")
        
        # Spatial
        from src.machine_learning.spatial import compute_mesf
        logger.info("✓ Spatial module imported")
        
        # GLM models
        from src.machine_learning.models.glm import NegativeBinomialGLM, PoissonGLM
        logger.info("✓ GLM models imported")
        
        from src.machine_learning.models.glm import DHARMaDiagnostics
        logger.info("✓ Diagnostics imported")
        
        # Research methods
        from src.machine_learning.research.methods import (
            Method1ContinuousGLM,
            Method2CategoricalGLM,
            Method3DML
        )
        logger.info("✓ Research methods imported")
        
        # Hypothesis testing
        from src.machine_learning.hypothesis import HypothesisTester
        logger.info("✓ Hypothesis tester imported")
        
        # Pipeline
        from src.machine_learning.pipelines import RegressionPipeline
        logger.info("✓ Regression pipeline imported")
        
        logger.info("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"\n❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    logger.info("\n" + "="*60)
    logger.info("Testing configuration...")
    logger.info("="*60)
    
    try:
        from src.machine_learning.config_loader import MLRegressionConfig
        
        # Test default config
        config = MLRegressionConfig()
        logger.info("✓ Default config loaded")
        
        # Test config methods
        preprocessing = config.get_preprocessing_config()
        logger.info(f"✓ Preprocessing config: strategy={preprocessing.get('missing_data', {}).get('strategy')}")
        
        glm = config.get_glm_config()
        logger.info(f"✓ GLM config: max_iter={glm.get('max_iter')}")
        
        spatial = config.get_spatial_config()
        logger.info(f"✓ Spatial config: method={spatial.get('method')}")
        
        dml = config.get_dml_config()
        logger.info(f"✓ DML config: n_folds={dml.get('n_folds')}")
        
        # Test distribution selection
        disp_ratio = 2.5
        dist = config.get_distribution_family(disp_ratio)
        logger.info(f"✓ Distribution selection: ratio={disp_ratio:.1f} -> {dist}")
        
        logger.info("\n✅ Configuration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spatial(data):
    """Test spatial filtering."""
    logger.info("\n" + "="*60)
    logger.info("Testing spatial filtering (MESF)...")
    logger.info("="*60)
    
    try:
        from src.machine_learning.spatial import compute_mesf
        from src.machine_learning.config_loader import MLRegressionConfig
        
        config = MLRegressionConfig()
        spatial_config = config.get_spatial_config()
        
        # Test with small subset
        coords = data[['longitude', 'latitude']].values[:100]
        
        logger.info(f"Testing with {len(coords)} coordinates")
        mesf_df = compute_mesf(coords, spatial_config)
        
        logger.info(f"✓ MESF computed: {mesf_df.shape[1]} eigenvectors")
        logger.info(f"✓ Eigenvector names: {list(mesf_df.columns[:5])}...")
        
        logger.info("\n✅ Spatial filtering test passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Spatial test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method1(data):
    """Test Method 1: Continuous GLM."""
    logger.info("\n" + "="*60)
    logger.info("Testing Method 1: Continuous Interaction GLM...")
    logger.info("="*60)
    
    try:
        from src.machine_learning.research.methods import Method1ContinuousGLM
        
        # Initialize method
        method1 = Method1ContinuousGLM()
        logger.info("✓ Method 1 initialized")
        
        # Prepare data
        data_prepared = method1.prepare_data(data)
        logger.info(f"✓ Data prepared: {data_prepared.shape}")
        
        # Fit models
        results = method1.fit_models()
        logger.info("✓ Models fitted")
        
        # Get interaction coefficient
        interaction = method1.get_interaction_coefficient()
        logger.info(f"✓ Interaction coefficient: {interaction['coefficient']:.4f} "
                   f"(p={interaction['p_value']:.4f})")
        logger.info(f"  Interpretation: {interaction['interpretation']}")
        
        # Run diagnostics
        diagnostics = method1.run_diagnostics()
        logger.info(f"✓ Diagnostics complete: adequate={diagnostics['model_adequate']}")
        
        logger.info("\n✅ Method 1 test passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Method 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method2(data):
    """Test Method 2: Categorical GLM."""
    logger.info("\n" + "="*60)
    logger.info("Testing Method 2: Categorical GLM with emmeans...")
    logger.info("="*60)
    
    try:
        from src.machine_learning.research.methods import Method2CategoricalGLM
        
        # Initialize method
        method2 = Method2CategoricalGLM()
        logger.info("✓ Method 2 initialized")
        
        # Prepare data
        data_prepared = method2.prepare_data(data)
        logger.info(f"✓ Data prepared: {data_prepared.shape}")
        
        # Fit model
        model_results = method2.fit_model()
        logger.info("✓ Model fitted")
        logger.info(f"  Distribution: {model_results['distribution']}")
        logger.info(f"  Interaction p-value: {model_results['interaction_pvalue']:.4f}")
        
        # Compute emmeans
        emmeans = method2.compute_emmeans()
        logger.info("✓ Emmeans computed")
        for i, group in enumerate(emmeans['groups']):
            logger.info(f"  {group}: {emmeans['emmeans'][i]:.2f}")
        
        # Test H2 contrast
        h2_results = method2.test_h2_contrast()
        logger.info(f"✓ H2 contrast: {h2_results['contrast_estimate']:.4f} "
                   f"(p={h2_results['p_value']:.4f})")
        logger.info(f"  {h2_results['interpretation']}")
        
        logger.info("\n✅ Method 2 test passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Method 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method3(data):
    """Test Method 3: DML."""
    logger.info("\n" + "="*60)
    logger.info("Testing Method 3: Double Machine Learning...")
    logger.info("="*60)
    
    try:
        from src.machine_learning.research.methods import Method3DML
        
        # Initialize method with fewer folds for speed
        method3 = Method3DML()
        method3.n_folds = 3  # Reduce for testing
        logger.info("✓ Method 3 initialized (3-fold for testing)")
        
        # Prepare data
        data_prepared = method3.prepare_data(data)
        logger.info(f"✓ Data prepared: {data_prepared.shape}")
        
        # Test H1 hypothesis
        h1_results = method3.cross_fit_dml('h1')
        logger.info(f"✓ H1 DML complete: effect={h1_results['causal_effect']:.4f} "
                   f"(p={h1_results['p_value']:.4f})")
        logger.info(f"  {h1_results['interpretation']}")
        
        # Test H2 hypothesis
        h2_results = method3.cross_fit_dml('h2')
        logger.info(f"✓ H2 DML complete: effect={h2_results['causal_effect']:.4f} "
                   f"(p={h2_results['p_value']:.4f})")
        logger.info(f"  {h2_results['interpretation']}")
        
        logger.info("\n✅ Method 3 test passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Method 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hypothesis_tester(data):
    """Test the hypothesis tester orchestrator."""
    logger.info("\n" + "="*60)
    logger.info("Testing HypothesisTester orchestrator...")
    logger.info("="*60)
    
    try:
        from src.machine_learning.hypothesis import HypothesisTester
        
        # Initialize tester
        tester = HypothesisTester()
        logger.info("✓ HypothesisTester initialized")
        
        # Prepare data
        data_prepared = tester.prepare_data(data)
        logger.info(f"✓ Data prepared: {data_prepared.shape}")
        
        # Test Method 1
        method1_results = tester.test_method1()
        logger.info("✓ Method 1 complete")
        
        # Test Method 2
        method2_results = tester.test_method2()
        logger.info("✓ Method 2 complete")
        
        # Reduce folds for Method 3 to speed up
        tester.method3.n_folds = 2
        
        # Test Method 3
        method3_results = tester.test_method3()
        logger.info("✓ Method 3 complete")
        
        # Synthesize results
        synthesis = tester.synthesize_results()
        logger.info("✓ Results synthesized")
        
        if 'h2' in synthesis:
            logger.info(f"\nH2 Consensus: {synthesis['h2']['consensus']}")
            logger.info(f"  {synthesis['h2']['interpretation']}")
        
        logger.info("\n✅ HypothesisTester test passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ HypothesisTester test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline(data_path):
    """Test the full regression pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Testing full RegressionPipeline...")
    logger.info("="*60)
    
    try:
        from src.machine_learning.pipelines import RegressionPipeline
        
        # Initialize pipeline
        pipeline = RegressionPipeline()
        logger.info("✓ Pipeline initialized")
        
        # Load data
        pipeline.load_data(data_path)
        logger.info("✓ Data loaded")
        
        # Run diagnostics only first
        diagnostics = pipeline.run_diagnostics_only()
        logger.info(f"✓ Diagnostics: dispersion={diagnostics['dispersion_ratio']:.2f}, "
                   f"distribution={diagnostics['recommended_distribution']}")
        
        # Reduce complexity for testing
        pipeline.hypothesis_tester.method3.n_folds = 2
        
        # Run full analysis
        logger.info("\nRunning full analysis (this may take a minute)...")
        results = pipeline.run_analysis()
        logger.info("✓ Full analysis complete")
        
        # Save results
        output_dir = "test_output"
        pipeline.save_results(output_dir)
        logger.info(f"✓ Results saved to {output_dir}")
        
        logger.info("\n✅ Pipeline test passed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("REGRESSION FRAMEWORK TEST SUITE")
    logger.info("="*70)
    
    # Track results
    results = {}
    
    # Test imports
    results['imports'] = test_imports()
    if not results['imports']:
        logger.error("Cannot proceed without successful imports")
        return
    
    # Test configuration
    results['config'] = test_config()
    
    # Create synthetic data
    logger.info("\n" + "="*60)
    logger.info("Creating synthetic test data...")
    logger.info("="*60)
    data = create_synthetic_data(n_samples=500)  # Small dataset for testing
    
    # Save test data
    test_data_path = "test_data.parquet"
    data.to_parquet(test_data_path)
    logger.info(f"Test data saved to {test_data_path}")
    
    # Test individual components
    results['spatial'] = test_spatial(data)
    results['method1'] = test_method1(data)
    results['method2'] = test_method2(data)
    results['method3'] = test_method3(data)
    results['hypothesis_tester'] = test_hypothesis_tester(data)
    results['pipeline'] = test_pipeline(test_data_path)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:20s}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("\n🎉 All tests passed! The regression framework is ready to use.")
    else:
        logger.info("\n⚠️  Some tests failed. Please review the errors above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
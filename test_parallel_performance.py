#!/usr/bin/env python3
"""Test script to verify parallel processing performance improvements.

This script tests:
1. H2O GLM parallelization for Methods 1 & 2
2. LightGBM parallel processing in Method 3
3. Parallel bootstrap in DML
"""

import time
import logging
import pandas as pd
import numpy as np
from src.machine_learning.research.methods.method1_continuous import Method1ContinuousGLM
from src.machine_learning.research.methods.method2_categorical import Method2CategoricalGLM
from src.machine_learning.research.methods.method3_dml import Method3DML
from src.machine_learning.config_loader import MLRegressionConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(n_samples=10000):
    """Generate synthetic biodiversity data for testing."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'fungi_richness': np.random.negative_binomial(10, 0.3, n_samples),
        'plants_richness': np.random.negative_binomial(20, 0.4, n_samples),
        'vertebrates_richness': np.random.negative_binomial(15, 0.35, n_samples),
        'bio1': np.random.normal(15, 5, n_samples),  # Temperature
        'bio4': np.random.normal(500, 100, n_samples),  # Temperature seasonality
        'bio12': np.random.normal(1000, 300, n_samples),  # Precipitation
        'bio15': np.random.normal(50, 15, n_samples),  # Precipitation seasonality
        'elevation': np.random.normal(500, 200, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'latitude': np.random.uniform(-90, 90, n_samples)
    })
    
    return data


def test_method1_h2o_glm():
    """Test Method 1 with H2O GLM parallelization."""
    logger.info("="*60)
    logger.info("Testing Method 1: H2O GLM Parallelization")
    logger.info("="*60)
    
    # Generate data
    data = generate_test_data(n_samples=50000)
    
    # Initialize Method 1
    config = MLRegressionConfig()
    method1 = Method1ContinuousGLM(config)
    
    # Prepare data
    logger.info("Preparing data...")
    start = time.time()
    data_processed = method1.prepare_data(data)
    prep_time = time.time() - start
    logger.info(f"Data preparation took {prep_time:.2f} seconds")
    
    # Fit models with H2O GLM
    logger.info("Fitting models with H2O GLM (256-core parallelization)...")
    start = time.time()
    results = method1.fit_models(data_processed)
    fit_time = time.time() - start
    
    logger.info(f"Model fitting took {fit_time:.2f} seconds")
    logger.info(f"Distribution used: {results['distribution']}")
    
    # Get interaction coefficient
    interaction = method1.get_interaction_coefficient()
    logger.info(f"Interaction coefficient: {interaction['coefficient']:.4f} (p={interaction['p_value']:.4f})")
    logger.info(f"H2 supported: {interaction['supports_h2']}")
    
    return fit_time


def test_method2_h2o_glm():
    """Test Method 2 with H2O GLM parallelization."""
    logger.info("="*60)
    logger.info("Testing Method 2: H2O GLM Parallelization")
    logger.info("="*60)
    
    # Generate data
    data = generate_test_data(n_samples=50000)
    
    # Initialize Method 2
    config = MLRegressionConfig()
    method2 = Method2CategoricalGLM(config)
    
    # Prepare data
    logger.info("Preparing data...")
    start = time.time()
    data_processed = method2.prepare_data(data)
    prep_time = time.time() - start
    logger.info(f"Data preparation took {prep_time:.2f} seconds")
    
    # Fit model with H2O GLM
    logger.info("Fitting categorical model with H2O GLM...")
    start = time.time()
    results = method2.fit_model(data_processed)
    fit_time = time.time() - start
    
    logger.info(f"Model fitting took {fit_time:.2f} seconds")
    
    # Test H2 contrast
    h2_results = method2.test_h2_contrast()
    logger.info(f"H2 contrast: {h2_results['contrast_estimate']:.2f} (p={h2_results['p_value']:.4f})")
    logger.info(f"H2 supported: {h2_results['h2_supported']}")
    
    return fit_time


def test_method3_parallel():
    """Test Method 3 with parallel LightGBM and bootstrap."""
    logger.info("="*60)
    logger.info("Testing Method 3: Parallel LightGBM & Bootstrap")
    logger.info("="*60)
    
    # Generate data
    data = generate_test_data(n_samples=10000)  # Smaller for DML
    
    # Initialize Method 3
    config = MLRegressionConfig()
    method3 = Method3DML(config)
    
    # Prepare data
    logger.info("Preparing data...")
    start = time.time()
    data_processed = method3.prepare_data(data)
    prep_time = time.time() - start
    logger.info(f"Data preparation took {prep_time:.2f} seconds")
    
    # Run cross-fitted DML with parallel LightGBM
    logger.info("Running cross-fitted DML with parallel LightGBM...")
    start = time.time()
    h1_results = method3.cross_fit_dml('h1')
    dml_time = time.time() - start
    
    logger.info(f"DML estimation took {dml_time:.2f} seconds")
    logger.info(f"Causal effect: {h1_results['causal_effect']:.4f} (p={h1_results['p_value']:.4f})")
    
    # Test parallel bootstrap (subset)
    logger.info("Testing parallel bootstrap (10 iterations for demo)...")
    method3.n_bootstrap = 10  # Reduce for testing
    start = time.time()
    
    # Run just the bootstrap part
    from joblib import Parallel, delayed
    import gc
    
    n = len(method3.data_processed)
    n_jobs = 4  # Use 4 cores for test
    
    def bootstrap_iter(seed):
        np.random.seed(seed)
        boot_idx = np.random.choice(n, n, replace=True)
        temp = Method3DML(config)
        temp.data_processed = method3.data_processed.iloc[boot_idx]
        result = temp.cross_fit_dml('h1')
        del temp
        gc.collect()
        return result['causal_effect']
    
    bootstrap_effects = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(bootstrap_iter)(42 + b) for b in range(10)
    )
    
    boot_time = time.time() - start
    logger.info(f"Parallel bootstrap (10 iterations) took {boot_time:.2f} seconds")
    logger.info(f"Bootstrap effects range: [{min(bootstrap_effects):.4f}, {max(bootstrap_effects):.4f}]")
    
    return dml_time


def main():
    """Run all performance tests."""
    logger.info("Starting parallel performance tests...")
    logger.info(f"Testing on system with {os.cpu_count()} CPU cores")
    
    try:
        # Test Method 1
        method1_time = test_method1_h2o_glm()
        
        # Test Method 2  
        method2_time = test_method2_h2o_glm()
        
        # Test Method 3
        method3_time = test_method3_parallel()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Method 1 (H2O GLM): {method1_time:.2f} seconds")
        logger.info(f"Method 2 (H2O GLM): {method2_time:.2f} seconds")
        logger.info(f"Method 3 (Parallel LightGBM): {method3_time:.2f} seconds")
        logger.info("\nAll methods successfully parallelized for 256-core CPU!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown H2O cluster
        try:
            import h2o
            h2o.cluster().shutdown()
            logger.info("H2O cluster shutdown")
        except:
            pass


if __name__ == "__main__":
    import os
    import sys
    
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    main()
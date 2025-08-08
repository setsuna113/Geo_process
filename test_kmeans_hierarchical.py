#!/usr/bin/env python3
"""Test script for k-means hierarchical clustering implementation."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_algorithm():
    """Test the core two-stage clustering algorithm."""
    logger.info("Testing core two-stage algorithm...")
    
    from src.biodiversity_analysis.methods.kmeans_hierarchical.hierarchical_core import (
        TwoStageClusterer, VariancePreservingKMeans, BootstrapStability
    )
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # Create 3 distinct clusters
    X = np.vstack([
        np.random.randn(300, n_features) + [0, 0, 0, 0],
        np.random.randn(400, n_features) + [5, 5, 0, 0],
        np.random.randn(300, n_features) + [0, 5, 5, 5]
    ])
    
    # Test TwoStageClusterer
    clusterer = TwoStageClusterer(
        n_pre_clusters=10,
        linkage_method='ward',
        final_k=3,
        batch_size=100
    )
    
    labels = clusterer.fit_predict(X)
    
    assert len(np.unique(labels)) == 3, "Should have 3 clusters"
    assert len(labels) == n_samples, "Should have labels for all samples"
    logger.info(f"✓ Core algorithm test passed. Found {len(np.unique(labels))} clusters")
    
    # Test variance preservation
    vp_kmeans = VariancePreservingKMeans(
        variance_threshold=0.9,
        k_range=(5, 20)
    )
    
    optimal_k = vp_kmeans.determine_k(X, sample_size=500)
    assert 5 <= optimal_k <= 20, f"Optimal k should be in range, got {optimal_k}"
    logger.info(f"✓ Variance preservation test passed. Optimal k={optimal_k}")
    
    return True

def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    from src.config import MethodConfigRegistry
    from src.biodiversity_analysis.methods.kmeans_hierarchical import KMeansHierarchicalConfig
    
    # Test config loader registration
    assert MethodConfigRegistry.is_registered('kmeans_hierarchical'), "Method should be registered"
    
    # Get config
    config = MethodConfigRegistry.get_method_config('kmeans_hierarchical')
    assert 'method_params' in config, "Config should have method_params"
    assert config['method_params']['linkage_method'] == 'ward', "Default should be ward"
    
    # Test config object
    config_obj = KMeansHierarchicalConfig()
    assert config_obj.linkage_method == 'ward'
    assert config_obj.n_pre_clusters == 750
    
    logger.info("✓ Configuration test passed")
    return True

def test_analyzer():
    """Test the analyzer with small synthetic data."""
    logger.info("Testing analyzer...")
    
    from src.biodiversity_analysis.methods.kmeans_hierarchical import KMeansHierarchicalAnalyzer
    
    # Create small test dataset
    n_samples = 500
    
    # Create test data with biodiversity-like structure
    data = {
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'animal_richness': np.random.poisson(50, n_samples),
        'plant_richness': np.random.poisson(200, n_samples),
        'am_fungi_richness': np.random.poisson(20, n_samples),
        'ecm_fungi_richness': np.random.poisson(30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save as test parquet
    test_file = Path('/tmp/test_biodiversity.parquet')
    df.to_parquet(test_file)
    
    # Initialize analyzer
    analyzer = KMeansHierarchicalAnalyzer()
    
    # Test with fast preset
    try:
        result = analyzer.analyze(
            str(test_file),
            n_pre_clusters=20,  # Small for testing
            final_k_range=(2, 4),
            n_bootstrap=5,  # Few iterations for speed
            generate_dendrogram=False,  # Skip heavy viz for test
            generate_geographic_maps=False,
            generate_3d_visualization=False,
            generate_validation_plots=False
        )
        
        assert result is not None, "Should return result"
        assert hasattr(result, 'labels'), "Result should have labels"
        assert len(result.labels) == n_samples, "Should have labels for all samples"
        
        logger.info(f"✓ Analyzer test passed. Clusters: {len(np.unique(result.labels[result.labels >= 0]))}")
        
    except Exception as e:
        logger.error(f"Analyzer test failed: {e}")
        return False
    
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
    
    return True

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.biodiversity_analysis.methods.kmeans_hierarchical import (
            KMeansHierarchicalAnalyzer,
            KMeansHierarchicalAnalyzerAdapter,
            TwoStageClusterer,
            KMeansHierarchicalConfig,
            VariancePreservingKMeans,
            BootstrapStability
        )
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("Testing K-means Hierarchical Implementation")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Core Algorithm", test_core_algorithm),
        ("Configuration", test_configuration),
        ("Analyzer", test_analyzer)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n{name}:")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary:")
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        logger.info("\n✓ All tests passed!")
    else:
        logger.error("\n✗ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
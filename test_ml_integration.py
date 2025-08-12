#!/usr/bin/env python3
"""
Test script to verify ML pipeline integration with new orchestrator.
Tests import paths and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    # Check if H2O is available first
    try:
        import h2o
        print("✅ H2O is available")
    except ImportError:
        print("⚠️  H2O not available in current environment - skipping H2O-dependent tests")
        print("   (H2O is available in geo environment)")
        # Test non-H2O imports only
        return test_non_h2o_imports()
    
    try:
        from src.machine_learning.ml_orchestrator import MLOrchestrator
        print("✅ MLOrchestrator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MLOrchestrator: {e}")
        return False
    
    try:
        from src.machine_learning.config_loader import MLRegressionConfig
        print("✅ MLRegressionConfig imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MLRegressionConfig: {e}")
        return False
    
    try:
        from src.machine_learning.methods.method1_continuous_refactored import Method1H0H1GLM
        print("✅ Method1 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Method1: {e}")
        return False
    
    try:
        from src.machine_learning.methods.method2_categorical_refactored import Method2H2Testing
        print("✅ Method2 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Method2: {e}")
        return False
    
    try:
        from src.machine_learning.methods.method3_dml_refactored import Method3DML
        print("✅ Method3 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Method3: {e}")
        return False
    
    # Test hypothesis modules
    try:
        from src.machine_learning.hypothesis import (
            WaldTest, LikelihoodRatioTest, ContrastAnalysis, SimpleSlopesAnalysis
        )
        print("✅ Hypothesis testing modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import hypothesis modules: {e}")
        return False
    
    # Test evaluation modules
    try:
        from src.machine_learning.evaluation import (
            GLMComparison, DMLModelComparison
        )
        print("✅ Evaluation modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import evaluation modules: {e}")
        return False
    
    return True

def test_orchestrator_initialization():
    """Test that orchestrator can be initialized."""
    print("\nTesting orchestrator initialization...")
    
    try:
        from src.machine_learning.ml_orchestrator import MLOrchestrator
        from src.machine_learning.config_loader import MLRegressionConfig
        
        # Try with explicit config path
        config_path = Path(__file__).parent / "src" / "config" / "ml" / "ml_config.yml"
        if config_path.exists():
            orchestrator = MLOrchestrator(config_path=str(config_path))
            print("✅ Orchestrator initialized with config")
        else:
            # Try without config
            orchestrator = MLOrchestrator(config_path=None)
            print("✅ Orchestrator initialized without config")
        
        # Check that methods can be accessed
        if hasattr(orchestrator, 'method1'):
            print("✅ Method1 accessible")
        if hasattr(orchestrator, 'method2'):
            print("✅ Method2 accessible")
        if hasattr(orchestrator, 'method3'):
            print("✅ Method3 accessible")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return False

def test_reporter_integration():
    """Test that reporters are properly integrated in hypothesis tests."""
    print("\nTesting reporter integration...")
    
    try:
        from src.machine_learning.hypothesis.wald_test import WaldTest
        from src.machine_learning.hypothesis.lrt_test import LikelihoodRatioTest
        
        # Check Wald test has publication report method
        if hasattr(WaldTest, 'generate_publication_report'):
            print("✅ WaldTest has generate_publication_report method")
        else:
            print("❌ WaldTest missing generate_publication_report method")
        
        # Check Wald test has visualization method
        if hasattr(WaldTest, 'visualize_wald_test'):
            print("✅ WaldTest has visualize_wald_test method")
        else:
            print("❌ WaldTest missing visualize_wald_test method")
        
        # Check LRT has publication report method
        if hasattr(LikelihoodRatioTest, 'generate_publication_report'):
            print("✅ LikelihoodRatioTest has generate_publication_report method")
        else:
            print("❌ LikelihoodRatioTest missing generate_publication_report method")
        
        # Check LRT has visualization method
        if hasattr(LikelihoodRatioTest, 'visualize_lrt'):
            print("✅ LikelihoodRatioTest has visualize_lrt method")
        else:
            print("❌ LikelihoodRatioTest missing visualize_lrt method")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test reporter integration: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("ML PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test orchestrator
    if not test_orchestrator_initialization():
        all_passed = False
    
    # Test reporter integration
    if not test_reporter_integration():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
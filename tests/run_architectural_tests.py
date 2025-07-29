#!/usr/bin/env python3
"""
Architectural regression test runner.

This script runs all architectural tests to ensure that new patterns
and improvements maintain their design integrity over time.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_test_file(test_file: Path) -> bool:
    """Run a single test file and return success status."""
    logger.info(f"Running {test_file}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            cwd=test_file.parent.parent  # Run from project root
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {test_file.name} - PASSED")
            return True
        else:
            logger.error(f"❌ {test_file.name} - FAILED")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {test_file.name} - ERROR: {e}")
        return False


def main():
    """Run all architectural tests."""
    logger.info("🏗️  Running Architectural Regression Tests")
    
    # Find all architectural test files
    test_files = [
        Path(__file__).parent / "base" / "test_architectural_patterns.py",
        Path(__file__).parent / "database" / "test_schema_architecture.py", 
        Path(__file__).parent / "core" / "test_signal_handler_architecture.py"
    ]
    
    # Verify all test files exist
    missing_files = [f for f in test_files if not f.exists()]
    if missing_files:
        logger.error(f"Missing test files: {missing_files}")
        return 1
    
    # Run each test file
    results = []
    for test_file in test_files:
        success = run_test_file(test_file)
        results.append((test_file.name, success))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ARCHITECTURAL TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✅" if success else "❌"
        logger.info(f"{icon} {test_name:30} - {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    logger.info("="*60)
    logger.info(f"TOTAL: {passed + failed} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All architectural tests passed!")
        return 0
    else:
        logger.error(f"💥 {failed} architectural tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Simple test of the unified resampling pipeline with small datasets.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Simulate test mode
sys.modules['pytest'] = type(sys)('pytest')

def main():
    print("🧪 Testing Unified Resampling Pipeline - Simple Version")
    print("=" * 60)
    
    try:
        # Test 1: Config system
        print("\n🔍 Test 1: Configuration System")
        from src.config.config import Config
        config = Config()
        
        print(f"   Database: {config.database['database']} @ {config.database['port']}")
        print(f"   Data dir: {config.paths['data_dir']}")
        
        # Check resampling config
        resampling_config = config.get('resampling', {})
        if resampling_config:
            print(f"   Target resolution: {resampling_config.get('target_resolution', 'not set')}")
            print("   ✅ Resampling config loaded")
        
        # Test 2: Database connection
        print("\n🔍 Test 2: Database Connection")
        from src.database.connection import DatabaseManager
        db = DatabaseManager()
        
        if db.test_connection():
            print("   ✅ Database connection successful")
        else:
            print("   ❌ Database connection failed")
            return False
        
        # Test 3: Pipeline components
        print("\n🔍 Test 3: Pipeline Components")
        from src.pipelines.unified_resampling.pipeline_orchestrator import UnifiedResamplingPipeline
        from src.pipelines.unified_resampling.validation_checks import ValidationChecks
        
        validator = ValidationChecks(config)
        
        # Validate resampling config
        is_valid, error_msg = validator.validate_resampling_config()
        if not is_valid:
            print(f"   ❌ Resampling config invalid: {error_msg}")
            return False
        print("   ✅ Resampling config valid")
        
        # Validate datasets config
        is_valid, error_msg = validator.validate_datasets_config()
        if not is_valid:
            print(f"   ❌ Datasets config invalid: {error_msg}")
            return False
        print("   ✅ Datasets config valid")
        
        # Test 4: Very basic pipeline initialization
        print("\n🔍 Test 4: Pipeline Initialization")
        pipeline = UnifiedResamplingPipeline(config, db)
        print("   ✅ Pipeline initialized successfully")
        
        # Test 5: Check datasets exist
        print("\n🔍 Test 5: Dataset Files")
        data_dir = Path(config.paths['data_dir'])
        data_files = config.get('data_files', {})
        
        for key, filename in data_files.items():
            file_path = data_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {key}: {filename} ({size_mb:.1f}MB)")
            else:
                print(f"   ❌ {key}: {filename} - NOT FOUND")
                return False
        
        print("\n🎉 All basic tests passed!")
        print("\n📋 System Ready:")
        print("   ✅ Configuration system working")
        print("   ✅ Database connected")
        print("   ✅ Pipeline components available") 
        print("   ✅ Dataset files present")
        print("   ✅ Validation checks passing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to run full pipeline test!")
    else:
        print("\n💥 Fix issues before running full pipeline")
    sys.exit(0 if success else 1)
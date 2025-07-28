#!/usr/bin/env python3
"""Test script to demonstrate pipeline skip control functionality."""

import sys
sys.path.append('/home/yl998/dev/geo')

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.stages.skip_control import StageSkipController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_skip_control():
    """Test the skip control functionality."""
    
    # Initialize database
    db = DatabaseManager()
    
    # Create skip controller
    controller = StageSkipController(config, db)
    
    print("\n" + "="*60)
    print("TESTING SKIP CONTROL SYSTEM")
    print("="*60)
    
    # Test 1: Check with skip disabled (default)
    print("\n1. Testing with skip disabled (default config):")
    can_skip, reason = controller.can_skip_stage('resample', 'test_experiment')
    print(f"   Can skip resample? {can_skip}")
    print(f"   Reason: {reason}")
    
    # Test 2: Enable skip in config
    print("\n2. Enabling skip in config...")
    config.settings['pipeline'] = {
        'allow_skip_stages': True,
        'stages': {
            'data_load': {'skip_if_exists': True},
            'resample': {'skip_if_exists': True}
        },
        'data_validation': {
            'max_age_hours': 24,
            'check_source_timestamps': True
        }
    }
    
    # Test 3: Check with skip enabled but no data
    print("\n3. Testing with skip enabled but no existing data:")
    can_skip, reason = controller.can_skip_stage('resample', 'test_experiment')
    print(f"   Can skip resample? {can_skip}")
    print(f"   Reason: {reason}")
    
    # Test 4: Check if we have existing data
    from src.database.schema import schema
    existing_datasets = schema.get_resampled_datasets()
    print(f"\n4. Existing resampled datasets in DB: {len(existing_datasets)}")
    for ds in existing_datasets[:3]:  # Show first 3
        print(f"   - {ds['name']}: created at {ds['created_at']}")
    
    # Test 5: If data exists, check if we can skip
    if existing_datasets:
        print("\n5. Testing with skip enabled and existing data:")
        can_skip, reason = controller.can_skip_stage('resample', 'test_experiment')
        print(f"   Can skip resample? {can_skip}")
        print(f"   Reason: {reason}")
        
        # Log the decision
        controller.log_skip_decision('resample', can_skip, reason)
    
    # Test 6: Force fresh processing
    print("\n6. Testing with force_fresh=True:")
    can_skip, reason = controller.can_skip_stage('resample', 'test_experiment', force_fresh=True)
    print(f"   Can skip resample? {can_skip}")
    print(f"   Reason: {reason}")
    
    # Test 7: Check data_load stage
    print("\n7. Testing data_load stage:")
    can_skip, reason = controller.can_skip_stage('data_load', 'test_experiment')
    print(f"   Can skip data_load? {can_skip}")
    print(f"   Reason: {reason}")
    
    print("\n" + "="*60)
    print("Skip control test completed!")
    print("="*60)

if __name__ == "__main__":
    test_skip_control()
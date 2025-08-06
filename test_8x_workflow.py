#!/usr/bin/env python
"""Test script for 8x-only workflow"""

import sys
import traceback
from src.gee_integration.workflow.orchestrator import GEEWorkflowOrchestrator

def main():
    print("Initializing workflow for 8x-only export...")
    
    try:
        orchestrator = GEEWorkflowOrchestrator('config/gee_config.yml', 'debug_8x_only')
        print(f"✓ Orchestrator initialized")
        print(f"  Resolution outputs: {orchestrator._get_resolution_outputs()}")
        
        # Run workflow step by step
        print("\nStarting workflow steps:")
        
        # Step 1: Prerequisites
        print("\n1. Validating prerequisites...")
        orchestrator._validate_prerequisites()
        print("   ✓ Prerequisites validated")
        
        # Step 2: Authentication
        print("\n2. Authenticating with GEE...")
        orchestrator._authenticate()
        print("   ✓ Authenticated")
        
        # Step 3: Upload assets
        print("\n3. Uploading assets...")
        orchestrator._upload_assets()
        print("   ✓ Assets uploaded")
        
        # Step 4: Process in GEE
        print("\n4. Processing in GEE (recursive downsampling)...")
        orchestrator._process_in_gee()
        print("   ✓ Processing complete")
        
        # Step 5: Export
        print("\n5. Exporting from GEE...")
        orchestrator._export_from_gee()
        print("   ✓ Export complete")
        
        # Step 6: Download and convert
        print("\n6. Downloading and converting...")
        outputs = orchestrator._download_and_convert()
        print("   ✓ Download complete")
        
        print("\n✅ Workflow completed successfully!")
        print(f"Outputs: {outputs}")
        
    except Exception as e:
        print(f"\n❌ Workflow failed at step: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
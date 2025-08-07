#!/usr/bin/env python3
"""Run GEE integration workflow

This script runs the complete GEE workflow to merge biodiversity
and climate data at multiple resolutions.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gee_integration import get_workflow_orchestrator
from src.gee_integration.workflow.state_manager import WorkflowStep
from src.infrastructure.logging import get_logger
from src.config import config


def main():
    parser = argparse.ArgumentParser(
        description="Run GEE biodiversity-climate merge workflow"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/gee/gee_config.yml',
        help='Path to GEE configuration file'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Experiment name for logging'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from last completed step (default: True)'
    )
    
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Start from beginning'
    )
    
    parser.add_argument(
        '--force-step',
        type=str,
        choices=[s.value for s in WorkflowStep],
        help='Force start from specific step'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run validation only'
    )
    
    args = parser.parse_args()
    
    # Get logger
    logger = get_logger(__name__)
    
    try:
        # Create orchestrator
        GEEWorkflowOrchestrator = get_workflow_orchestrator()
        orchestrator = GEEWorkflowOrchestrator(
            config_path=args.config,
            experiment_name=args.experiment_name
        )
        
        if args.dry_run:
            print("🔍 Running validation only...")
            orchestrator._validate_prerequisites()
            print("✅ Validation passed!")
            return 0
        
        # Convert force_step string to enum
        force_step = None
        if args.force_step:
            force_step = WorkflowStep(args.force_step)
        
        # Run workflow
        outputs = orchestrator.run(
            resume=args.resume,
            force_step=force_step
        )
        
        # Print results
        print("\n📊 Final outputs:")
        for resolution, parquet_path in outputs.items():
            print(f"   {resolution}: {parquet_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
        print("Run with --resume to continue from last completed step")
        return 1
        
    except Exception as e:
        import traceback
        print(f"\n❌ Workflow failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nRun with --resume to continue from last completed step")
        return 1


if __name__ == "__main__":
    sys.exit(main())
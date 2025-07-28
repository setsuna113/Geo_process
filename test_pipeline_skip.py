#!/usr/bin/env python3
"""Test pipeline with skip control enabled to debug merge stage."""

import sys
sys.path.append('/home/yl998/dev/geo')

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineOrchestrator
from src.pipelines.stages.load_stage import DataLoadStage
from src.pipelines.stages.resample_stage import ResampleStage
from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.export_stage import ExportStage
from src.pipelines.stages.analysis_stage import AnalysisStage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline_with_skip():
    """Run pipeline with skip control enabled."""
    
    print("\n" + "="*60)
    print("RUNNING PIPELINE WITH SKIP CONTROL ENABLED")
    print("="*60)
    
    # Initialize database
    db = DatabaseManager()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config, db)
    
    # Add stages
    orchestrator.register_stage(DataLoadStage())
    orchestrator.register_stage(ResampleStage())
    orchestrator.register_stage(MergeStage())
    orchestrator.register_stage(ExportStage())
    orchestrator.register_stage(AnalysisStage())
    
    try:
        # Run pipeline
        print("\nStarting pipeline execution...")
        results = orchestrator.run_pipeline(
            experiment_name="production_run_skip_test",
            description="Testing skip control to debug merge stage"
        )
        
        print("\n✅ Pipeline completed successfully!")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_pipeline_with_skip()
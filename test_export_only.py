#!/usr/bin/env python3
"""Test script to run export and analysis stages on existing merged data."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.stages.export_stage import ExportStage
from src.pipelines.stages.analysis_stage import AnalysisStage
from src.pipelines.orchestrator import PipelineContext
from src.config import config

def run_export_and_analysis():
    """Run export and analysis on the merged dataset."""
    # Find the latest merged dataset
    output_dirs = sorted(Path("outputs").glob("*/merged_dataset.nc"), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not output_dirs:
        print("No merged datasets found!")
        return
    
    merged_file = output_dirs[0]
    output_dir = merged_file.parent
    print(f"Using merged dataset: {merged_file}")
    print(f"File size: {merged_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Create pipeline context
    context = PipelineContext()
    context.output_dir = str(output_dir)
    context.data['merged_dataset'] = str(merged_file)
    
    # Run export stage
    print("\n=== Running Export Stage ===")
    export_stage = ExportStage(config.export)
    try:
        export_result = export_stage.execute(context)
        print(f"Export completed: {export_result}")
        
        # Update context with export result
        if 'output_file' in export_result:
            context.data['export_file'] = export_result['output_file']
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run analysis stage
    print("\n=== Running Analysis Stage ===")
    analysis_stage = AnalysisStage(config.analysis)
    try:
        analysis_result = analysis_stage.execute(context)
        print(f"Analysis completed: {analysis_result}")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_export_and_analysis()
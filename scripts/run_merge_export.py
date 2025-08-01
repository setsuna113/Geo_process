#!/usr/bin/env python3
"""Run just merge and export stages for already processed data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.pipelines.orchestrator_enhanced import EnhancedPipelineOrchestrator
from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.export_stage import ExportStage
from src.config.config import Config
from src.database.connection import DatabaseManager
from src.core.signal_handler_enhanced import EnhancedSignalHandler

def main():
    """Run merge and export stages."""
    config = Config()
    db = DatabaseManager()
    signal_handler = EnhancedSignalHandler()
    
    orchestrator = EnhancedPipelineOrchestrator(config, db, signal_handler)
    
    # Only register merge and export stages
    stages = [MergeStage(), ExportStage()]
    for stage in stages:
        orchestrator.register_stage(stage)
    
    print('🚀 Starting merge and export...')
    
    # Create minimal context with dataset info
    context = {
        'datasets_loaded': {
            'plants-richness': {
                'dataset_name': 'plants-richness',
                'table_name': 'passthrough_plants_richness'
            },
            'terrestrial-richness': {
                'dataset_name': 'terrestrial-richness', 
                'table_name': 'passthrough_terrestrial_richness'
            },
            'am-fungi-richness': {
                'dataset_name': 'am-fungi-richness',
                'table_name': 'passthrough_am_fungi_richness'
            },
            'ecm-fungi-richness': {
                'dataset_name': 'ecm-fungi-richness',
                'table_name': 'passthrough_ecm_fungi_richness'
            }
        }
    }
    
    try:
        # Set initial context
        orchestrator.context = context
        
        results = orchestrator.run_pipeline(
            experiment_name='merge_export_only',
            description='Merge and export downsampled data to parquet'
        )
        print('✅ Merge and export completed!')
        
        # Print output location
        if 'export' in results and results['export'].get('outputs'):
            for output in results['export']['outputs']:
                if output.endswith('.parquet'):
                    print(f'📦 Parquet file: {output}')
                    
    except Exception as e:
        print(f'❌ Failed: {e}')
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
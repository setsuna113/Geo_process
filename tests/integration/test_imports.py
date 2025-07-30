#!/usr/bin/env python3
"""Test that all major imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")
errors = []

# Test config imports
try:
    from src.config import config
    print("✓ src.config.config (singleton)")
except Exception as e:
    errors.append(f"✗ src.config: {e}")

try:
    from src.config.config import Config
    print("✓ src.config.config.Config (class)")
except Exception as e:
    errors.append(f"✗ src.config.config.Config: {e}")

# Test logging imports
try:
    from src.infrastructure.logging import get_logger, setup_logging
    print("✓ src.infrastructure.logging")
except Exception as e:
    errors.append(f"✗ src.infrastructure.logging: {e}")

# Test database imports
try:
    from src.database.connection import DatabaseManager
    print("✓ src.database.connection.DatabaseManager")
except Exception as e:
    errors.append(f"✗ src.database.connection.DatabaseManager: {e}")

# Test processor imports
try:
    from src.processors.data_preparation.resampling_processor import ResamplingProcessor
    print("✓ src.processors.data_preparation.resampling_processor")
except Exception as e:
    errors.append(f"✗ ResamplingProcessor: {e}")

# Test pipeline imports
try:
    from src.pipelines.orchestrator import PipelineOrchestrator, PipelineContext
    print("✓ src.pipelines.orchestrator")
except Exception as e:
    errors.append(f"✗ src.pipelines.orchestrator: {e}")

# Test stage imports
try:
    from src.pipelines.stages.load_stage import DataLoadStage
    from src.pipelines.stages.resample_stage import ResampleStage
    from src.pipelines.stages.merge_stage import MergeStage
    from src.pipelines.stages.export_stage import ExportStage
    print("✓ All pipeline stages")
except Exception as e:
    errors.append(f"✗ Pipeline stages: {e}")

# Test analyzer imports
try:
    from src.spatial_analysis.som.som_trainer import SOMAnalyzer
    from src.spatial_analysis.gwpca_analyzer import GWPCAAnalyzer
    from src.spatial_analysis.maxp_regions.region_optimizer import MaxPAnalyzer
    print("✓ All analyzers")
except Exception as e:
    errors.append(f"✗ Analyzers: {e}")

# Test progress manager imports
try:
    from src.core.enhanced_progress_manager import get_enhanced_progress_manager
    print("✓ src.core.enhanced_progress_manager")
except Exception as e:
    errors.append(f"✗ Enhanced progress manager: {e}")

# Summary
print(f"\n{'='*50}")
if errors:
    print(f"Import test FAILED with {len(errors)} errors:")
    for error in errors:
        print(f"  {error}")
else:
    print("All imports PASSED!")
print(f"{'='*50}")
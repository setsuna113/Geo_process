#!/usr/bin/env python3
"""Migrate imports from abstractions to core_abstractions."""

import os
import re
from pathlib import Path

# Define import mappings
IMPORT_MAPPINGS = [
    # IAnalyzer and related
    (r'from src\.abstractions\.interfaces\.analyzer import IAnalyzer',
     'from src.core_abstractions import IAnalyzer'),
    (r'from src\.abstractions\.interfaces\.analyzer import AnalysisResult',
     'from src.core_abstractions import AnalysisResult'),
    (r'from src\.abstractions\.interfaces\.analyzer import AnalysisMetadata',
     'from src.core_abstractions import AnalysisMetadata'),
    (r'from src\.abstractions\.interfaces\.analyzer import IAnalyzer, AnalysisResult, AnalysisMetadata',
     'from src.core_abstractions import IAnalyzer, AnalysisResult, AnalysisMetadata'),
    (r'from src\.abstractions\.interfaces\.analyzer import IAnalyzer, AnalysisResult',
     'from src.core_abstractions import IAnalyzer, AnalysisResult'),
    
    # IMethodConfig
    (r'from src\.abstractions\.interfaces\.method_config import IMethodConfig',
     'from src.core_abstractions import IMethodConfig'),
    
    # Tensor types
    (r'from src\.abstractions\.types\.tensor_types import TensorData',
     'from src.core_abstractions import TensorData'),
    (r'from src\.abstractions\.types\.tensor_types import CPFactors',
     'from src.core_abstractions import CPFactors'),
    (r'from src\.abstractions\.types\.tensor_types import TensorMetrics',
     'from src.core_abstractions import TensorMetrics'),
    (r'from src\.abstractions\.types\.tensor_types import \(',
     'from src.core_abstractions import ('),
    
    # Biodiversity types
    (r'from src\.abstractions\.types\.biodiversity_types import BiodiversityData',
     'from src.core_abstractions import BiodiversityData'),
    
    # ITensorMetricsCalculator
    (r'from src\.abstractions\.interfaces\.tensor_analyzer import ITensorMetricsCalculator',
     'from src.core_abstractions import ITensorMetricsCalculator'),
]

def migrate_file(file_path: Path) -> bool:
    """Migrate imports in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        for old_import, new_import in IMPORT_MAPPINGS:
            content = re.sub(old_import, new_import, content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Run the migration."""
    src_dir = Path('src')
    
    # Find all Python files
    python_files = list(src_dir.rglob('*.py'))
    
    # Exclude abstractions directory itself
    python_files = [f for f in python_files if 'abstractions' not in str(f)]
    
    print(f"Found {len(python_files)} Python files to check")
    
    migrated_count = 0
    for file_path in python_files:
        if migrate_file(file_path):
            print(f"✓ Migrated: {file_path}")
            migrated_count += 1
    
    print(f"\nMigration complete: {migrated_count} files updated")

if __name__ == '__main__':
    main()
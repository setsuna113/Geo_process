#!/usr/bin/env python3
"""Migrate imports from core_abstractions to new abstractions module structure."""

import os
import re
from pathlib import Path

# Define import mappings
IMPORT_MAPPINGS = [
    # Replace core_abstractions with abstractions
    (r'from src\.core_abstractions import', 'from src.abstractions import'),
    (r'import src\.core_abstractions', 'import src.abstractions'),
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
    
    # Exclude the new abstractions module itself
    python_files = [f for f in python_files if 'abstractions' not in f.parts[-2:]]
    
    print(f"Found {len(python_files)} Python files to check")
    
    migrated_count = 0
    for file_path in python_files:
        if migrate_file(file_path):
            print(f"✓ Migrated: {file_path}")
            migrated_count += 1
    
    print(f"\nMigration complete: {migrated_count} files updated")

if __name__ == '__main__':
    main()
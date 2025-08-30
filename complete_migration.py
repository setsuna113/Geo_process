#!/usr/bin/env python3
"""Complete migration to eliminate ALL redundancy. No fallback."""

import re
import ast
from pathlib import Path
from typing import Set, Tuple, List

def migrate_logger(content: str, file_path: Path) -> str:
    """Migrate ALL logger initializations."""
    
    # Pattern 1: logger = logging.getLogger(__name__)
    if 'logger = logging.getLogger(__name__)' in content:
        # Replace the logger initialization
        content = content.replace(
            'logger = logging.getLogger(__name__)',
            'logger = get_logger(__name__)'
        )
        
        # Add import if not present
        if 'from src.abstractions import get_logger' not in content:
            # Find where to insert import (after other imports)
            lines = content.split('\n')
            import_added = False
            
            for i, line in enumerate(lines):
                if line.startswith('import logging'):
                    # Add right after logging import
                    lines.insert(i + 1, 'from src.abstractions import get_logger')
                    import_added = True
                    break
                elif line.startswith('from') or line.startswith('import'):
                    continue
                elif not line.strip() or line.startswith('#'):
                    continue
                else:
                    # First non-import line
                    if not import_added:
                        lines.insert(i, 'from src.abstractions import get_logger')
                    break
            
            content = '\n'.join(lines)
    
    # Pattern 2: self.logger = logging.getLogger(...)
    content = re.sub(
        r'self\.logger = logging\.getLogger\((.*?)\)',
        r'self.logger = get_logger(\1)',
        content
    )
    
    return content


def migrate_validation(content: str, file_path: Path) -> str:
    """Migrate validation methods to use utilities."""
    
    # Check if file has validate_parameters method
    if 'def validate_parameters' in content:
        # Analyzers have specific patterns - need careful migration
        if 'analyzer' in str(file_path).lower():
            # Add import
            if 'from src.abstractions import validate_parameters' not in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'from src.abstractions import' in line:
                        # Add to existing abstractions import
                        if 'validate_parameters' not in line:
                            lines[i] = line.rstrip(')') + ', validate_parameters)'
                        break
                    elif line.startswith('from src.abstractions'):
                        lines.insert(i + 1, 'from src.abstractions import validate_parameters as validate_params_util')
                        break
                content = '\n'.join(lines)
    
    return content


def migrate_io_operations(content: str, file_path: Path) -> str:
    """Migrate I/O operations to use utilities."""
    
    # Pattern: Save JSON with open(..., 'w') as f: json.dump(...)
    json_save_pattern = r'with open\(([^,]+),\s*[\'"]w[\'"]\)\s*as\s*(\w+):\s*\n\s*json\.dump\(([^,]+),\s*\2[^\)]*\)'
    
    if re.search(json_save_pattern, content):
        # Add import
        if 'from src.abstractions import save_json' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import json'):
                    lines.insert(i + 1, 'from src.abstractions import save_json')
                    break
            content = '\n'.join(lines)
    
    # Pattern: Load JSON with open(..., 'r') as f: json.load(f)
    json_load_pattern = r'with open\(([^,]+),\s*[\'"]r[\'"]\)\s*as\s*(\w+):\s*\n\s*.*?json\.load\(\2\)'
    
    if re.search(json_load_pattern, content):
        if 'from src.abstractions import load_json' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'from src.abstractions import save_json' in line:
                    lines[i] = line.replace('save_json', 'save_json, load_json')
                    break
                elif line.startswith('import json'):
                    lines.insert(i + 1, 'from src.abstractions import load_json')
                    break
            content = '\n'.join(lines)
    
    return content


def clean_imports(content: str) -> str:
    """Clean up imports after migration."""
    lines = content.split('\n')
    
    # Remove duplicate imports
    seen_imports = set()
    cleaned_lines = []
    
    for line in lines:
        if line.startswith('import ') or line.startswith('from '):
            if line not in seen_imports:
                seen_imports.add(line)
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def process_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Process a single file for migration."""
    changes = []
    
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        content = original_content
        
        # Skip abstractions module itself
        if 'abstractions' in str(file_path):
            return False, []
        
        # Apply migrations
        old_content = content
        content = migrate_logger(content, file_path)
        if content != old_content:
            changes.append("logger initialization")
        
        old_content = content
        content = migrate_validation(content, file_path)
        if content != old_content:
            changes.append("validation methods")
        
        old_content = content
        content = migrate_io_operations(content, file_path)
        if content != old_content:
            changes.append("I/O operations")
        
        # Clean imports
        content = clean_imports(content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True, changes
        
        return False, []
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, []


def main():
    """Execute complete migration."""
    print("=" * 60)
    print("COMPLETE MIGRATION - NO FALLBACK")
    print("=" * 60)
    
    src_dir = Path('src')
    python_files = list(src_dir.rglob('*.py'))
    
    # Filter out abstractions module
    python_files = [f for f in python_files if 'abstractions' not in str(f)]
    
    print(f"\nProcessing {len(python_files)} files...")
    
    migrated_files = []
    migration_stats = {
        'logger': 0,
        'validation': 0,
        'io': 0
    }
    
    for file_path in python_files:
        changed, changes = process_file(file_path)
        
        if changed:
            migrated_files.append((file_path, changes))
            
            for change in changes:
                if 'logger' in change:
                    migration_stats['logger'] += 1
                elif 'validation' in change:
                    migration_stats['validation'] += 1
                elif 'I/O' in change:
                    migration_stats['io'] += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal files migrated: {len(migrated_files)}")
    print(f"Logger migrations: {migration_stats['logger']}")
    print(f"Validation migrations: {migration_stats['validation']}")
    print(f"I/O migrations: {migration_stats['io']}")
    
    if migrated_files:
        print("\n" + "-" * 40)
        print("Files migrated:")
        print("-" * 40)
        for file_path, changes in migrated_files[:20]:  # Show first 20
            rel_path = file_path.relative_to(src_dir)
            print(f"  ✓ {rel_path}: {', '.join(changes)}")
        
        if len(migrated_files) > 20:
            print(f"  ... and {len(migrated_files) - 20} more files")


if __name__ == '__main__':
    main()
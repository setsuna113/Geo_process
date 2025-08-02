#!/usr/bin/env python3
"""Fix hardcoded paths in debug scripts by making them relative to project root."""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Project root is 3 levels up from this script
PROJECT_ROOT = Path(__file__).parent.parent


def find_hardcoded_paths(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find hardcoded paths in a file.
    
    Returns:
        List of (line_number, original_line, suggested_replacement)
    """
    hardcoded_pattern = re.compile(r'/home/yl998/dev/geo/?')
    results = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            if hardcoded_pattern.search(line):
                # Suggest replacement
                suggested = hardcoded_pattern.sub('Path(__file__).parent.parent.parent / ', line)
                # Clean up double slashes
                suggested = suggested.replace('/ /', '/')
                results.append((i, line.strip(), suggested.strip()))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return results


def fix_file(file_path: Path, dry_run: bool = True):
    """Fix hardcoded paths in a file."""
    hardcoded_paths = find_hardcoded_paths(file_path)
    
    if not hardcoded_paths:
        return
    
    print(f"\n{file_path.relative_to(PROJECT_ROOT)}:")
    
    if dry_run:
        for line_no, original, suggested in hardcoded_paths:
            print(f"  Line {line_no}:")
            print(f"    - {original}")
            print(f"    + {suggested}")
    else:
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace hardcoded paths
        content = re.sub(r'/home/yl998/dev/geo/?', 'Path(__file__).parent.parent.parent / ', content)
        content = content.replace('/ /', '/')
        
        # Check if we need to add Path import
        if 'Path(__file__)' in content and 'from pathlib import Path' not in content:
            # Add import after the first line (usually shebang or docstring)
            lines = content.split('\n')
            insert_pos = 1 if lines[0].startswith('#!') else 0
            lines.insert(insert_pos, 'from pathlib import Path')
            content = '\n'.join(lines)
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"  Fixed {len(hardcoded_paths)} hardcoded paths")


def main():
    """Main function to fix hardcoded paths in all debug scripts."""
    directories = [
        PROJECT_ROOT / 'scripts' / 'debug' / 'som',
        PROJECT_ROOT / 'scripts' / 'monitoring',
        PROJECT_ROOT / 'scripts' / 'testing',
    ]
    
    print("Scanning for hardcoded paths...")
    
    files_with_issues = []
    
    for directory in directories:
        if not directory.exists():
            continue
            
        for file_path in directory.glob('*.py'):
            if find_hardcoded_paths(file_path):
                files_with_issues.append(file_path)
    
    if not files_with_issues:
        print("No hardcoded paths found!")
        return
    
    print(f"\nFound hardcoded paths in {len(files_with_issues)} files:")
    
    # Dry run first
    for file_path in files_with_issues:
        fix_file(file_path, dry_run=True)
    
    response = input("\nFix these hardcoded paths? (y/N): ")
    if response.lower() == 'y':
        for file_path in files_with_issues:
            fix_file(file_path, dry_run=False)
        print("\nDone! All hardcoded paths have been fixed.")
    else:
        print("No changes made.")


if __name__ == '__main__':
    main()
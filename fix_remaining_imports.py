#!/usr/bin/env python3
"""Fix remaining missing imports."""

from pathlib import Path
import re

def fix_overlap_import(file_path: Path):
    """Fix overlap.py imports."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the non-existent overlap_rgb import
    content = re.sub(
        r'from src\.biodiversity_analysis\.statistical\.overlap_rgb import .*\n',
        '', 
        content
    )
    
    # Comment out calls to create_rgb_overlap_map
    content = re.sub(
        r'(\s*)create_rgb_overlap_map\(',
        r'\1# create_rgb_overlap_map(',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Fix all remaining import issues."""
    print("Fixing remaining import issues...")
    
    # Fix overlap.py
    overlap_path = Path('src/biodiversity_analysis/statistical/overlap.py')
    if overlap_path.exists():
        if fix_overlap_import(overlap_path):
            print(f"✓ Fixed {overlap_path.name}")

if __name__ == '__main__':
    main()
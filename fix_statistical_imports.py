#!/usr/bin/env python3
"""Fix missing imports in statistical modules."""

from pathlib import Path
import re

def fix_file(file_path: Path):
    """Fix imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # Remove the non-existent result_types import
    content = re.sub(
        r'from src\.biodiversity_analysis\.statistical\.result_types import .*\n',
        '', 
        content
    )
    
    # Remove the non-existent smart_visualization import
    content = re.sub(
        r'from src\.biodiversity_analysis\.statistical\.smart_visualization import .*\n',
        '',
        content
    )
    
    # Replace MethodResult with Dict[str, Any]
    content = re.sub(r'\bMethodResult\b', 'Dict[str, Any]', content)
    
    # Replace smart_imshow with plt.imshow
    content = re.sub(r'\bsmart_imshow\b', 'plt.imshow', content)
    
    if content != original:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all statistical module files."""
    stat_dir = Path('src/biodiversity_analysis/statistical')
    
    files_to_fix = [
        'correlation_analyzer.py',
        'partial_correlation.py',
        'overlap.py',
        'trust_map.py',
        'orchestrator.py',
        'latitudinal.py',
        'biome_analysis.py',
        'coverage.py',
        'global_map.py'
    ]
    
    print("Fixing statistical module imports...")
    
    for file_name in files_to_fix:
        file_path = stat_dir / file_name
        if file_path.exists():
            if fix_file(file_path):
                print(f"✓ Fixed {file_name}")
            else:
                print(f"  No changes needed for {file_name}")

if __name__ == '__main__':
    main()
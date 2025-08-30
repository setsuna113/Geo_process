#!/usr/bin/env python3
"""Fix all statistical module import issues."""

from pathlib import Path
import re

def fix_smart_visualization_imports(file_path: Path):
    """Fix smart_visualization imports."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Remove smart_visualization imports
        if 'from .smart_visualization import' in line or 'from src.biodiversity_analysis.statistical.smart_visualization' in line:
            continue
        # Replace smart_imshow with plt.imshow
        line = re.sub(r'\bsmart_imshow\b', 'plt.imshow', line)
        fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

def fix_professional_map_imports(file_path: Path):
    """Fix professional_map_visualizer imports."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Remove professional_map_visualizer imports
        if 'professional_map_visualizer' in line:
            continue
        # Comment out calls to functions from that module
        if 'create_professional_global_map' in line:
            line = '    # ' + line.lstrip() if line.strip() else line
        fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

def main():
    """Fix all statistical module files."""
    stat_dir = Path('src/biodiversity_analysis/statistical')
    
    # Files that have smart_visualization imports
    smart_viz_files = [
        'trust_map.py',
        'global_map.py',
    ]
    
    print("Fixing statistical module imports...")
    
    for file_name in smart_viz_files:
        file_path = stat_dir / file_name
        if file_path.exists():
            fix_smart_visualization_imports(file_path)
            print(f"✓ Fixed {file_name} smart_visualization imports")
    
    # Fix professional_map_visualizer
    global_map = stat_dir / 'global_map.py'
    if global_map.exists():
        fix_professional_map_imports(global_map)
        print(f"✓ Fixed global_map.py professional_map imports")

if __name__ == '__main__':
    main()
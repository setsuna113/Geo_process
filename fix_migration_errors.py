#!/usr/bin/env python3
"""Fix syntax errors from aggressive migration."""

import re
from pathlib import Path

def fix_imports(content: str) -> str:
    """Fix import statement errors."""
    lines = content.split('\n')
    fixed_lines = []
    import_section = []
    other_section = []
    
    in_imports = True
    for line in lines:
        # Skip duplicate/malformed imports
        if line.strip().startswith('from') or line.strip().startswith('import'):
            # Clean up bad imports
            if 'validate_parameters)' in line:
                line = line.replace('validate_parameters)', 'validate_parameters')
            if not line.strip() or line.strip() == 'import' or line.strip() == 'from':
                continue
            import_section.append(line)
        else:
            if in_imports and line.strip() and not line.strip().startswith('#'):
                in_imports = False
            other_section.append(line)
    
    # Deduplicate and sort imports
    seen = set()
    clean_imports = []
    for imp in import_section:
        if imp.strip() not in seen:
            seen.add(imp.strip())
            clean_imports.append(imp)
    
    # Sort with abstractions imports first
    clean_imports.sort(key=lambda x: (not 'from src.abstractions' in x, x))
    
    return '\n'.join(clean_imports + other_section)


def fix_duplicate_lines(content: str) -> str:
    """Remove duplicate 'No newline at end of file' and other duplicates."""
    lines = content.split('\n')
    fixed_lines = []
    prev_line = None
    
    for line in lines:
        # Skip duplicate "No newline" markers
        if 'No newline at end of file' in line:
            continue
        # Skip exact duplicates
        if line != prev_line or not line.strip():
            fixed_lines.append(line)
        prev_line = line
    
    return '\n'.join(fixed_lines)


def fix_docstring_placement(content: str) -> str:
    """Fix docstring that ended up in wrong place."""
    # Move docstrings to right after class/function definition
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if next line is a docstring that should be after imports
        if i + 1 < len(lines) and lines[i + 1].strip().startswith('"""') and not line.strip().startswith('class') and not line.strip().startswith('def'):
            # Check if this is the module docstring (should be at top)
            if i < 20 and 'import' in line:
                # Find where imports end
                j = i
                while j < len(lines) and (lines[j].startswith('from') or lines[j].startswith('import') or not lines[j].strip()):
                    j += 1
                # Move docstring after imports
                docstring_start = i + 1
                docstring_lines = []
                k = docstring_start
                while k < len(lines):
                    docstring_lines.append(lines[k])
                    if lines[k].strip().endswith('"""'):
                        break
                    k += 1
                
                # Skip the docstring in original position
                i = k + 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)


def fix_file(file_path: Path) -> bool:
    """Fix a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original = content
        
        # Apply fixes
        content = fix_imports(content)
        content = fix_duplicate_lines(content)
        content = fix_docstring_placement(content)
        
        # Ensure file ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
        
        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Fix all files with migration errors."""
    problem_files = [
        'src/biodiversity_analysis/methods/cpwopt/analyzer_adapter.py',
        'src/biodiversity_analysis/methods/som/analyzer.py',
        'src/biodiversity_analysis/methods/kmeans_hierarchical/hierarchical_analyzer.py',
        'src/biodiversity_analysis/shared/base_analyzer.py',
        'src/machine_learning/doubleML/refactored_pipeline.py',
        'src/biodiversity_analysis/methods/som/analyzer_adapter.py',
        'src/biodiversity_analysis/methods/kmeans_hierarchical/analyzer_adapter.py',
    ]
    
    print("Fixing migration syntax errors...")
    
    for file_str in problem_files:
        file_path = Path(file_str)
        if file_path.exists():
            if fix_file(file_path):
                print(f"✓ Fixed {file_path.name}")
            else:
                print(f"  No changes needed for {file_path.name}")


if __name__ == '__main__':
    main()
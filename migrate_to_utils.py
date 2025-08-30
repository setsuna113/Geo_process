#!/usr/bin/env python3
"""Migrate files to use new abstractions utilities."""

import re
from pathlib import Path
from typing import List, Tuple

def migrate_logger_init(content: str) -> Tuple[str, int]:
    """Replace logger = logging.getLogger(__name__) with get_logger."""
    pattern = r'logger = logging\.getLogger\(__name__\)'
    replacement = 'from src.abstractions import get_logger\nlogger = get_logger(__name__)'
    
    # Check if already has the import
    if 'from src.abstractions import get_logger' in content:
        return content, 0
    
    # Count replacements
    count = len(re.findall(pattern, content))
    
    if count > 0:
        # Replace the pattern
        content = re.sub(pattern, 'logger = get_logger(__name__)', content)
        
        # Add import at the top (after other imports)
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if in_imports and (line.startswith('import ') or line.startswith('from ')):
                import_lines.append(line)
            else:
                if in_imports and line and not line.startswith('#'):
                    in_imports = False
                    # Add our import before first non-import line
                    import_lines.append('from src.abstractions import get_logger')
                other_lines.append(line)
        
        content = '\n'.join(import_lines + other_lines)
    
    return content, count


def migrate_save_json(content: str) -> Tuple[str, int]:
    """Replace inline JSON save code with save_json utility."""
    # Pattern for common JSON save operations
    patterns = [
        # with open(..., 'w') as f: json.dump(...)
        (r'with open\(([^,]+), [\'"]w[\'"]\) as \w+:\s*\n\s*json\.dump\([^,]+, \w+(?:, indent=\d+)?\)',
         'save_json({data}, {path})'),
    ]
    
    count = 0
    for pattern, replacement in patterns:
        matches = re.findall(pattern, content)
        count += len(matches)
    
    # Would need more complex logic to properly replace
    # For now, just count occurrences
    
    return content, count


def analyze_file(file_path: Path) -> dict:
    """Analyze a file for potential migrations."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {
        'path': file_path,
        'logger_init': len(re.findall(r'logger = logging\.getLogger\(__name__\)', content)),
        'json_save': len(re.findall(r'with open\(.+[\'"]w[\'"]\) as .+json\.dump', content)),
        'json_load': len(re.findall(r'with open\(.+[\'"]r[\'"]\) as .+json\.load', content)),
        'validate_params': 'def validate_parameters' in content,
        'validate_input': 'def validate_input_data' in content,
        'save_results': 'def save_results' in content,
        'load_results': 'def load_results' in content,
    }
    
    return results


def main():
    """Analyze codebase for migration opportunities."""
    src_dir = Path('src')
    
    # Find Python files
    python_files = list(src_dir.rglob('*.py'))
    
    # Skip abstractions itself
    python_files = [f for f in python_files if 'abstractions' not in str(f)]
    
    print(f"Analyzing {len(python_files)} files for migration opportunities...\n")
    
    # Collect statistics
    total_logger = 0
    total_json_save = 0
    total_json_load = 0
    files_with_validate_params = []
    files_with_validate_input = []
    files_with_save_results = []
    
    for file_path in python_files:
        try:
            results = analyze_file(file_path)
            
            if results['logger_init'] > 0:
                total_logger += results['logger_init']
            
            if results['json_save'] > 0:
                total_json_save += results['json_save']
            
            if results['json_load'] > 0:
                total_json_load += results['json_load']
                
            if results['validate_params']:
                files_with_validate_params.append(file_path)
                
            if results['validate_input']:
                files_with_validate_input.append(file_path)
                
            if results['save_results']:
                files_with_save_results.append(file_path)
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # Print summary
    print("=" * 60)
    print("MIGRATION OPPORTUNITIES SUMMARY")
    print("=" * 60)
    print(f"\nLogger initialization: {total_logger} instances")
    print(f"JSON save operations: {total_json_save} instances")
    print(f"JSON load operations: {total_json_load} instances")
    print(f"Files with validate_parameters: {len(files_with_validate_params)}")
    print(f"Files with validate_input_data: {len(files_with_validate_input)}")
    print(f"Files with save_results: {len(files_with_save_results)}")
    
    print("\n" + "=" * 60)
    print("TOP PRIORITY FILES TO MIGRATE:")
    print("=" * 60)
    
    # Show files with validate methods
    if files_with_validate_params:
        print("\nFiles with validate_parameters to migrate:")
        for f in files_with_validate_params[:5]:
            print(f"  - {f}")
    
    if files_with_validate_input:
        print("\nFiles with validate_input_data to migrate:")
        for f in files_with_validate_input[:5]:
            print(f"  - {f}")
    
    if files_with_save_results:
        print("\nFiles with save_results to migrate:")
        for f in files_with_save_results[:5]:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
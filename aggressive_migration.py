#!/usr/bin/env python3
"""Aggressive migration - remove ALL redundant validation and I/O code."""

import re
import ast
from pathlib import Path
from typing import List, Tuple

def remove_redundant_validate_methods(content: str, file_path: Path) -> str:
    """Remove redundant validate_parameters and validate_input_data methods."""
    
    # Skip if it's a base class or abstract method
    if '@abstractmethod' in content and 'def validate_' in content:
        return content
    
    # Pattern to find validate_parameters method (entire method)
    pattern = r'(\s*)def validate_parameters\(self[^:]+:\s*[^:]+\)[^:]*:.*?(?=\n\1def|\n\1class|\Z)'
    
    if 'def validate_parameters' in content and 'analyzer' in str(file_path).lower():
        # Check if method just does basic validation
        match = re.search(pattern, content, re.DOTALL)
        if match:
            method_body = match.group(0)
            # If it's just checking required params, replace with utility
            if 'errors = []' in method_body or 'required' in method_body:
                # Replace method with wrapper using utility
                indent = match.group(1)
                replacement = f'''{indent}def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
{indent}    """Validate parameters using utility."""
{indent}    from src.abstractions import validate_parameters as validate_params
{indent}    # Define required parameters for this analyzer
{indent}    required = self.get_required_parameters() if hasattr(self, 'get_required_parameters') else []
{indent}    return validate_params(parameters, required)'''
                
                content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
    
    # Similar for validate_input_data
    pattern2 = r'(\s*)def validate_input_data\(self[^:]+:\s*[^:]+\)[^:]*:.*?(?=\n\1def|\n\1class|\Z)'
    
    if 'def validate_input_data' in content:
        match = re.search(pattern2, content, re.DOTALL)
        if match:
            method_body = match.group(0)
            if 'errors = []' in method_body or 'isinstance' in method_body:
                indent = match.group(1)
                replacement = f'''{indent}def validate_input_data(self, data: Any) -> Tuple[bool, List[str]]:
{indent}    """Validate input data using utility."""
{indent}    from src.abstractions import validate_input_data as validate_data
{indent}    return validate_data(data, expected_type=self.expected_data_type() if hasattr(self, 'expected_data_type') else None)'''
                
                content = re.sub(pattern2, replacement, content, count=1, flags=re.DOTALL)
    
    return content


def remove_redundant_save_results(content: str, file_path: Path) -> str:
    """Remove redundant save_results methods."""
    
    # Pattern to find save_results method
    pattern = r'(\s*)def save_results\(self[^:]+:\s*[^:]+\)[^:]*:.*?(?=\n\1def|\n\1class|\Z)'
    
    if 'def save_results' in content and 'json.dump' in content:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            method_body = match.group(0)
            # If it's just saving JSON, replace with utility
            if 'json.dump' in method_body or 'Path(' in method_body:
                indent = match.group(1)
                replacement = f'''{indent}def save_results(self, results: Any, output_path: Union[str, Path]) -> Path:
{indent}    """Save results using utility."""
{indent}    from src.abstractions import save_results as save_util
{indent}    return save_util(results, Path(output_path).parent, Path(output_path).stem)'''
                
                content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
    
    return content


def clean_duplicate_imports(content: str) -> str:
    """Remove duplicate imports and organize them."""
    lines = content.split('\n')
    
    # Collect all imports
    import_lines = []
    other_lines = []
    seen_imports = set()
    
    for line in lines:
        if line.strip().startswith('from ') or line.strip().startswith('import '):
            # Normalize and deduplicate
            normalized = line.strip()
            if normalized not in seen_imports:
                seen_imports.add(normalized)
                import_lines.append(line)
        else:
            other_lines.append(line)
    
    # Sort imports
    import_lines.sort(key=lambda x: (not x.startswith('from src.abstractions'), x))
    
    # Combine
    if import_lines:
        return '\n'.join(import_lines) + '\n' + '\n'.join(other_lines)
    return content


def process_file_aggressive(file_path: Path) -> Tuple[bool, List[str]]:
    """Aggressively migrate a file."""
    changes = []
    
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        content = original_content
        
        # Skip abstractions module
        if 'abstractions' in str(file_path):
            return False, []
        
        # Remove redundant methods
        old_content = content
        content = remove_redundant_validate_methods(content, file_path)
        if content != old_content:
            changes.append("validation methods")
        
        old_content = content
        content = remove_redundant_save_results(content, file_path)
        if content != old_content:
            changes.append("save_results")
        
        # Clean imports
        content = clean_duplicate_imports(content)
        
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
    """Execute aggressive migration."""
    print("=" * 60)
    print("AGGRESSIVE MIGRATION - REMOVE ALL REDUNDANCY")
    print("=" * 60)
    
    # Target files with known redundancy
    target_files = [
        'src/biodiversity_analysis/methods/som/analyzer.py',
        'src/biodiversity_analysis/methods/cpwopt/analyzer_adapter.py',
        'src/biodiversity_analysis/methods/kmeans_hierarchical/hierarchical_analyzer.py',
        'src/biodiversity_analysis/shared/base_analyzer.py',
        'src/machine_learning/doubleML/refactored_pipeline.py',
    ]
    
    for file_str in target_files:
        file_path = Path(file_str)
        if file_path.exists():
            changed, changes = process_file_aggressive(file_path)
            if changed:
                print(f"✓ {file_path.name}: {', '.join(changes)}")
            else:
                print(f"  {file_path.name}: no changes needed")


if __name__ == '__main__':
    main()
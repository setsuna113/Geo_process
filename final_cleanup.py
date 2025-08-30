#!/usr/bin/env python3
"""Final cleanup to ensure all files are syntactically correct."""

import re
from pathlib import Path

PROBLEM_FILES = {
    'src/biodiversity_analysis/methods/cpwopt/analyzer_adapter.py': {
        'bad_lines': [
            '        importance += np.abs(feature_factors[:, r]) * component_weights[r]',
            '        importance /= importance.sum()',
            '        importance = np.zeros(feature_factors.shape[0])',
        ],
        'fix': 'move_to_correct_location'
    },
    'src/biodiversity_analysis/shared/base_analyzer.py': {
        'bad_lines': [
            'Base analyzer for biodiversity analysis methods.',
        ],
        'fix': 'add_quotes'
    },
    'src/machine_learning/doubleML/refactored_pipeline.py': {
        'bad_lines': [
            '        importances = self.estimator.feature_importances_',
        ],
        'fix': 'move_to_correct_location'
    }
}

def final_cleanup(file_path: Path) -> bool:
    """Final cleanup of a file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            # Skip "No newline at end of file" markers
            if 'No newline at end of file' in line:
                continue
                
            # Fix specific problem patterns
            if str(file_path) == 'src/biodiversity_analysis/shared/base_analyzer.py':
                if line.strip() == 'Base analyzer for biodiversity analysis methods.':
                    fixed_lines.append('"""Base analyzer for biodiversity analysis methods.\n')
                    continue
                    
            if str(file_path) == 'src/biodiversity_analysis/methods/cpwopt/analyzer_adapter.py':
                # Fix the misplaced importance calculation
                if 'importance += np.abs(feature_factors[:, r])' in line:
                    fixed_lines.append('            importance += np.abs(feature_factors[:, r]) * component_weights[r]\n')
                    continue
                elif 'importance /= importance.sum()' in line:
                    fixed_lines.append('        \n')
                    fixed_lines.append('        importance /= importance.sum()\n')
                    continue
                elif 'importance = np.zeros(feature_factors.shape[0])' in line:
                    fixed_lines.append('        importance = np.zeros(feature_factors.shape[0])\n')
                    continue
                    
            if str(file_path) == 'src/machine_learning/doubleML/refactored_pipeline.py':
                if 'importances = self.estimator.feature_importances_' in line:
                    fixed_lines.append('        importances = self.estimator.feature_importances_\n')
                    continue
            
            # Fix bad import lines
            if line.strip().startswith('from') or line.strip().startswith('import'):
                # Clean up malformed imports
                if 'validate_parameters)' in line:
                    line = line.replace('validate_parameters)', 'validate_parameters')
                # Skip empty imports
                if line.strip() in ['import', 'from']:
                    continue
            
            fixed_lines.append(line)
        
        # Write back
        content = ''.join(fixed_lines)
        
        # Ensure ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
            
        with open(file_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False


def main():
    """Run final cleanup."""
    print("Running final cleanup...")
    
    files_to_clean = [
        'src/biodiversity_analysis/methods/cpwopt/analyzer_adapter.py',
        'src/biodiversity_analysis/methods/som/analyzer.py',
        'src/biodiversity_analysis/methods/kmeans_hierarchical/hierarchical_analyzer.py',
        'src/biodiversity_analysis/shared/base_analyzer.py',
        'src/machine_learning/doubleML/refactored_pipeline.py',
        'src/biodiversity_analysis/methods/som/analyzer_adapter.py',
        'src/biodiversity_analysis/methods/kmeans_hierarchical/analyzer_adapter.py',
    ]
    
    for file_str in files_to_clean:
        file_path = Path(file_str)
        if file_path.exists():
            if final_cleanup(file_path):
                print(f"✓ Cleaned {file_path.name}")


if __name__ == '__main__':
    main()
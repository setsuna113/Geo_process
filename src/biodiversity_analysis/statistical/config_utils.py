"""
Simplified configuration utilities for statistical analysis.

Following Linus's principle: "Good code has no special cases"
Just load config and resolve paths. That's it.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = 'src/config/stat/stat_methods.yml') -> Dict[str, Any]:
    """Load configuration and resolve paths.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary with resolved paths
    """
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve dataset paths to absolute
    if 'datasets' in config:
        for key, path in config['datasets'].items():
            if path and not Path(path).is_absolute():
                config['datasets'][key] = str(Path(path).resolve())
    
    # Resolve output directory
    if 'settings' in config:
        output_dir = config['settings'].get('output_dir')
        if output_dir and not Path(output_dir).is_absolute():
            config['settings']['output_dir'] = str(Path(output_dir).resolve())
    
    return config
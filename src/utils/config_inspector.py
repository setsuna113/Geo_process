"""Utility to inspect and debug method configurations.

This module provides tools to inspect configuration for analysis methods,
including default values, environment variables, and current settings.
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

from src.config.method_registry import MethodConfigRegistry


class ConfigInspector:
    """Utility to inspect and debug configurations."""
    
    @staticmethod
    def show_method_config(method: str, show_env_vars: bool = True,
                          show_defaults: bool = True) -> None:
        """Display configuration for a method.
        
        Args:
            method: Method name to inspect
            show_env_vars: Whether to show environment variables
            show_defaults: Whether to show default values
        """
        print(f"\n{'='*60}")
        print(f"Configuration for: {method.upper()}")
        print(f"{'='*60}\n")
        
        try:
            loader = MethodConfigRegistry.get_loader(method)
            
            # Show current configuration
            print("Current Configuration:")
            print("-" * 30)
            config = loader.load_config()
            print(json.dumps(config, indent=2, sort_keys=True))
            
            # Show default configuration
            if show_defaults:
                print("\nDefault Configuration:")
                print("-" * 30)
                default_config = loader.get_default_config()
                print(json.dumps(default_config, indent=2, sort_keys=True))
            
            # Show environment variables
            if show_env_vars:
                print("\nEnvironment Variables:")
                print("-" * 30)
                
                env_vars = loader.get_env_var_names()
                descriptions = loader.get_env_var_descriptions()
                
                for var in sorted(env_vars):
                    value = os.environ.get(var, '<not set>')
                    desc = descriptions.get(var, 'No description available')
                    
                    print(f"\n{var}")
                    print(f"  Description: {desc}")
                    print(f"  Current value: {value}")
        
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            import traceback
            print(f"Unexpected error: {e}")
            traceback.print_exc()
    
    @staticmethod
    def show_env_vars(method: str) -> None:
        """Show only environment variables for a method.
        
        Args:
            method: Method name
        """
        print(f"\nEnvironment variables for {method.upper()}:")
        print("=" * 60)
        
        try:
            loader = MethodConfigRegistry.get_loader(method)
            env_vars = loader.get_env_var_names()
            descriptions = loader.get_env_var_descriptions()
            
            # Group by section
            sections = {}
            for var in sorted(env_vars):
                parts = var.split('_')
                if len(parts) > 2:
                    section = parts[1]
                    if section not in sections:
                        sections[section] = []
                    sections[section].append(var)
            
            # Display by section
            for section, vars in sorted(sections.items()):
                print(f"\n{section.title()}:")
                print("-" * 30)
                
                for var in vars:
                    value = os.environ.get(var, '<not set>')
                    desc = descriptions.get(var, '')
                    
                    # Format the output nicely
                    var_display = var.replace(f"{method.upper()}_", '')
                    print(f"  {var_display:<40} = {value:<15} # {desc}")
        
        except ValueError as e:
            print(f"Error: {e}")
    
    @staticmethod
    def compare_with_defaults(method: str) -> None:
        """Show which values differ from defaults.
        
        Args:
            method: Method name
        """
        print(f"\nConfiguration differences from defaults for {method.upper()}:")
        print("=" * 60)
        
        try:
            loader = MethodConfigRegistry.get_loader(method)
            current = loader.load_config()
            defaults = loader.get_default_config()
            
            differences = ConfigInspector._find_differences(defaults, current)
            
            if not differences:
                print("No differences - using all default values")
            else:
                for path, (default_val, current_val) in differences.items():
                    print(f"\n{path}:")
                    print(f"  Default: {default_val}")
                    print(f"  Current: {current_val}")
        
        except ValueError as e:
            print(f"Error: {e}")
    
    @staticmethod
    def _find_differences(default: Dict[str, Any], current: Dict[str, Any],
                         path: str = "") -> Dict[str, tuple]:
        """Recursively find differences between configs.
        
        Args:
            default: Default configuration
            current: Current configuration
            path: Current path in config tree
            
        Returns:
            Dictionary of differences
        """
        differences = {}
        
        for key in set(default.keys()) | set(current.keys()):
            current_path = f"{path}.{key}" if path else key
            
            if key not in default:
                differences[current_path] = (None, current[key])
            elif key not in current:
                differences[current_path] = (default[key], None)
            elif isinstance(default[key], dict) and isinstance(current[key], dict):
                # Recurse into nested dicts
                nested_diff = ConfigInspector._find_differences(
                    default[key], current[key], current_path
                )
                differences.update(nested_diff)
            elif default[key] != current[key]:
                differences[current_path] = (default[key], current[key])
        
        return differences
    
    @staticmethod
    def export_config(method: str, output_file: Optional[Path] = None) -> None:
        """Export current configuration to file.
        
        Args:
            method: Method name
            output_file: Output file path (defaults to method_config.json)
        """
        if output_file is None:
            output_file = Path(f"{method}_config.json")
        
        try:
            loader = MethodConfigRegistry.get_loader(method)
            config = loader.load_config()
            
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            print(f"Configuration exported to: {output_file}")
        
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Failed to export: {e}")
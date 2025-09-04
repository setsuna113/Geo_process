"""
Data variants - Handle fungi separation (AM vs ECM).

Linus: "Special cases are bugs waiting to happen."
But biologists insist on analyzing fungi separately. Fine.
"""

from typing import Dict, List, Any
import numpy as np


class DataVariants:
    """Static methods for data variants. No state, no instance needed."""
    
    @staticmethod
    def generate_fungi_variants(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Generate fungi variants: separate and merged.
        
        Args:
            data: Dictionary with data arrays or paths
            
        Returns:
            Dict of variant_name -> data
        """
        # Check if we have fungi data
        has_am = 'am_fungi' in data
        has_ecm = 'ecm_fungi' in data
        
        if not (has_am or has_ecm):
            # No fungi data, return original as default
            return {'default': data}
        
        variants = {}
        
        # Variant 1: Separate fungi
        if has_am and has_ecm:
            separate = data.copy()
            # Keep AM and ECM separate
            variants['separate'] = separate
        
        # Variant 2: Merged fungi  
        if has_am and has_ecm:
            merged = data.copy()
            # For spatial data (paths), we can't directly merge
            # Methods must handle merging themselves
            if isinstance(data.get('am_fungi'), str):
                # It's a path - keep both, method will merge
                merged['merged_fungi'] = True  # Flag for method
            else:
                # It's array data - can merge directly
                merged['fungi'] = data['am_fungi'] + data['ecm_fungi']
                del merged['am_fungi']
                del merged['ecm_fungi']
            variants['merged'] = merged
        elif has_am:
            # Only AM
            only_am = data.copy()
            if not isinstance(data.get('am_fungi'), str):
                only_am['fungi'] = data['am_fungi']
                del only_am['am_fungi']
            variants['am_only'] = only_am
        elif has_ecm:
            # Only ECM
            only_ecm = data.copy()
            if not isinstance(data.get('ecm_fungi'), str):
                only_ecm['fungi'] = data['ecm_fungi']
                del only_ecm['ecm_fungi']
            variants['ecm_only'] = only_ecm
        
        return variants if variants else {'default': data}
    
    @staticmethod
    def supports_variants(data: Dict[str, Any]) -> bool:
        """Check if data supports fungi variants.
        
        Simple rule: if we have both AM and ECM fungi.
        """
        return 'am_fungi' in data and 'ecm_fungi' in data
    
    @staticmethod
    def get_variant_suffix(variant_name: str) -> str:
        """Get suffix for output directories based on variant name.
        
        Args:
            variant_name: Name of the variant
            
        Returns:
            Suffix string (e.g., "_merged_fungi")
        """
        if variant_name == 'separate':
            return '_separate_fungi'
        elif variant_name == 'merged':
            return '_merged_fungi'
        elif variant_name == 'am_only':
            return '_am_fungi'
        elif variant_name == 'ecm_only':
            return '_ecm_fungi'
        else:
            return ''  # Default/no variant
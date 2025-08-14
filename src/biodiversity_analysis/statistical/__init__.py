"""
Statistical analysis module - Linus style.

"Talk is cheap. Show me the code."
All functions, no classes, no inheritance.
"""

# Direct function imports
from .correlation_analyzer import analyze_correlation
from .coverage import analyze_coverage
from .overlap import analyze_overlap
from .latitudinal import analyze_latitudinal
from .global_map import analyze_global_maps
from .trust_map import analyze_trust_maps
from .orchestrator import run_analysis

__all__ = [
    'analyze_correlation',
    'analyze_coverage',
    'analyze_overlap',
    'analyze_latitudinal',
    'analyze_global_maps',
    'analyze_trust_maps',
    'run_analysis'
]
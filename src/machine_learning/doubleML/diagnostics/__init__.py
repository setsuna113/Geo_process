"""
EconML Diagnostics Module
Real diagnostics based on actual EconML public API
"""

from .core import (
    EconMLDiagnostics,
    run_full_diagnostics
)

__all__ = [
    'EconMLDiagnostics',
    'run_full_diagnostics'
]

__version__ = '1.0.0'
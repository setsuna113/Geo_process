"""CP-WOPT tensor factorization for biodiversity analysis.

This package implements CANDECOMP/PARAFAC decomposition with weighted optimization
for analyzing sparse biodiversity data patterns.
"""

from .core import CPWOPT
from .analyzer_adapter import CPWOPTAnalyzer, create_cpwopt_analyzer
from .tensor_metrics import CPWOPTMetricsCalculator
from .sparse_handler import SparseHandler, AdaptiveSparseStrategy
from .factor_updater import FactorUpdater, AdaptiveFactorUpdater

__all__ = [
    'CPWOPT',
    'CPWOPTAnalyzer',
    'create_cpwopt_analyzer',
    'CPWOPTMetricsCalculator',
    'SparseHandler',
    'AdaptiveSparseStrategy',
    'FactorUpdater',
    'AdaptiveFactorUpdater'
]
"""
GEE Integration V2 - 简化的模块化重构
====================================

原代码18个文件5500行，重构后7个文件1200行。
保留所有关键功能，去除冗余复杂性。

作者: Linus式重构
"""

from .pipeline import GEEPipeline, GEEConfig
from .validation import TIFValidator
from .gee_ops import GEEOperations
from .processing import DataProcessor
from .export import ExportManager
from .state import PipelineState

__version__ = '2.0.0'

__all__ = [
    'GEEPipeline',
    'GEEConfig',
    'TIFValidator', 
    'GEEOperations',
    'DataProcessor',
    'ExportManager',
    'PipelineState'
]
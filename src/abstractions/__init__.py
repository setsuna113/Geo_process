"""
Abstractions module - only multi-use abstractions.

Following the principle: Abstract if used multiple times, don't if used once.
"""

from .interfaces import IAnalyzer, IMethodConfig
from .types import (
    TensorData,
    CPFactors,
    AnalysisResult,
    AnalysisMetadata,
    BiodiversityData,
    TensorMetrics,
    SOMConfig,
    SOMTrainingResult,
    TensorMode,
    AnalysisStatus
)

# Import commonly used utilities
from .io import (
    save_json, load_json,
    save_pickle, load_pickle,
    save_results, load_results,
    save_dataframe, load_dataframe,
    save_numpy, load_numpy
)

from .validation import (
    validate_parameters,
    validate_input_data,
    validate_numeric_range,
    validate_path,
    validate_config,
    check_data_compatibility
)

from .logging import (
    get_logger,
    setup_logging,
    create_experiment_logger,
    log_dict,
    log_separator,
    log_section,
    suppress_warnings
)

__all__ = [
    # Interfaces
    'IAnalyzer',
    'IMethodConfig',
    # Types
    'TensorData',
    'CPFactors', 
    'AnalysisResult',
    'AnalysisMetadata',
    'BiodiversityData',
    'TensorMetrics',
    'SOMConfig',
    'SOMTrainingResult',
    'TensorMode',
    'AnalysisStatus',
    # I/O utilities
    'save_json', 'load_json',
    'save_pickle', 'load_pickle',
    'save_results', 'load_results',
    'save_dataframe', 'load_dataframe',
    'save_numpy', 'load_numpy',
    # Validation utilities
    'validate_parameters',
    'validate_input_data',
    'validate_numeric_range',
    'validate_path',
    'validate_config',
    'check_data_compatibility',
    # Logging utilities
    'get_logger',
    'setup_logging',
    'create_experiment_logger',
    'log_dict',
    'log_separator',
    'log_section',
    'suppress_warnings'
]
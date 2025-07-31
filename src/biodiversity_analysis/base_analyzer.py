"""
Base analyzer for biodiversity analysis methods.

This provides a clean base for biodiversity analyzers that:
- Load data from parquet files
- Use the biodiversity config system
- Handle spatial biodiversity data properly
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable

from src.abstractions.interfaces.analyzer import AnalysisResult, AnalysisMetadata
from src.abstractions.types.biodiversity_types import BiodiversityData
from src.config import get_biodiversity_config
from .shared.data import ParquetLoader

logger = logging.getLogger(__name__)


class BaseBiodiversityAnalyzer(ABC):
    """
    Base class for biodiversity analysis methods.
    
    Provides:
    - Parquet data loading
    - Biodiversity config integration
    - Progress tracking
    - Common preprocessing
    """
    
    def __init__(self, 
                 method_name: str,
                 version: str = "1.0.0",
                 progress_callback: Optional[Callable[[str, float], None]] = None):
        """
        Initialize biodiversity analyzer.
        
        Args:
            method_name: Name of the analysis method (e.g., 'som', 'gwpca')
            version: Version of the implementation
            progress_callback: Optional callback for progress updates
        """
        self.method_name = method_name
        self.version = version
        self.progress_callback = progress_callback
        
        # Load configuration
        config_manager = get_biodiversity_config()
        self.config = config_manager.get_method_config(method_name)
        
        # Data processing settings
        self.data_config = self.config.get('data_processing', {})
        self.spatial_config = self.config.get('spatial_validation', {})
        self.method_params = self.config.get('method_params', {})
        
        # Components
        self.data_loader = ParquetLoader()
        
        logger.info(f"Initialized {method_name} analyzer v{version}")
    
    @abstractmethod
    def analyze(self, data_path: str, **parameters) -> AnalysisResult:
        """
        Analyze biodiversity data from parquet file.
        
        Args:
            data_path: Path to parquet file
            **parameters: Method-specific parameters
            
        Returns:
            AnalysisResult with analysis outputs
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate analysis parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for this method.
        
        Returns:
            Dictionary of default parameter values
        """
        pass
    
    def load_data(self, 
                  data_path: str,
                  coordinate_cols: Optional[List[str]] = None,
                  feature_cols: Optional[List[str]] = None,
                  species_cols: Optional[List[str]] = None) -> BiodiversityData:
        """
        Load biodiversity data from parquet file.
        
        Args:
            data_path: Path to parquet file
            coordinate_cols: Coordinate column names (auto-detect if None)
            feature_cols: Feature columns to use (all if None)
            species_cols: Species column names if applicable
            
        Returns:
            BiodiversityData object
        """
        self.update_progress("Loading data", 0.1)
        
        try:
            data = self.data_loader.load(
                data_path,
                coordinate_cols=coordinate_cols,
                feature_cols=feature_cols,
                species_cols=species_cols,
                validate=True
            )
            
            logger.info(f"Loaded {data.n_samples} samples with {data.n_features} features")
            self.update_progress("Data loaded successfully", 0.2)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def preprocess_data(self, data: BiodiversityData) -> BiodiversityData:
        """
        Apply standard preprocessing based on config.
        
        Args:
            data: Input biodiversity data
            
        Returns:
            Preprocessed data
        """
        from .shared.data import FeaturePreprocessor, ZeroInflationHandler
        
        self.update_progress("Preprocessing data", 0.3)
        
        # Handle missing values
        if self.data_config.get('missing_value_strategy'):
            preprocessor = FeaturePreprocessor()
            data.features, data.feature_names = preprocessor.handle_missing_values(
                data.features,
                data.feature_names,
                strategy=self.data_config['missing_value_strategy']
            )
        
        # Handle zero inflation
        if self.data_config.get('handle_zero_inflation', False) and data.zero_inflated:
            handler = ZeroInflationHandler()
            data.features, transform_info = handler.transform_zero_inflated(
                data.features,
                method='log1p'  # Could be configurable
            )
            # Initialize metadata if None
            if data.metadata is None:
                data.metadata = {}
            data.metadata['zero_inflation_transform'] = transform_info
        
        # Remove constant features
        if self.data_config.get('remove_constant_features', True):
            preprocessor = FeaturePreprocessor()
            data.features, data.feature_names = preprocessor.remove_constant_features(
                data.features,
                data.feature_names
            )
        
        # Normalize if specified
        if self.data_config.get('normalization_method'):
            data.features, scaler_info = preprocessor.normalize_features(
                data.features,
                method=self.data_config['normalization_method']
            )
            # Initialize metadata if None
            if data.metadata is None:
                data.metadata = {}
            data.metadata['normalization'] = scaler_info
        
        self.update_progress("Preprocessing complete", 0.4)
        return data
    
    def update_progress(self, message: str, progress: float) -> None:
        """
        Update progress with optional callback.
        
        Args:
            message: Progress message
            progress: Progress value between 0 and 1
        """
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            logger.info(f"{message} ({progress*100:.0f}%)")
    
    def create_result(self,
                      success: bool,
                      data: Dict[str, Any],
                      runtime_seconds: float,
                      n_samples: int,
                      n_features: int,
                      parameters: Dict[str, Any],
                      errors: Optional[List[str]] = None,
                      warnings: Optional[List[str]] = None) -> AnalysisResult:
        """
        Create standardized analysis result.
        
        Args:
            success: Whether analysis succeeded
            data: Analysis outputs
            runtime_seconds: Total runtime
            n_samples: Number of samples analyzed
            n_features: Number of features
            parameters: Parameters used
            errors: Optional error messages
            warnings: Optional warnings
            
        Returns:
            AnalysisResult object
        """
        metadata = AnalysisMetadata(
            analysis_type=self.method_name.upper(),
            input_shape=(n_samples, n_features),
            input_bands=parameters.get('feature_names', []),
            parameters=parameters,
            processing_time=runtime_seconds,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            normalization_applied=self.data_config.get('normalization_method') is not None
        )
        
        result = AnalysisResult(
            labels=data.get('labels', None),
            metadata=metadata,
            statistics=data.get('statistics', {}),
            additional_outputs=data
        )
        
        # Note: The AnalysisResult from interfaces doesn't have errors/warnings
        # but we could add them to additional_outputs
        if errors or warnings:
            result.additional_outputs['errors'] = errors or []
            result.additional_outputs['warnings'] = warnings or []
        
        return result
    
    def measure_runtime(self, func: Callable) -> Tuple[Any, float]:
        """
        Measure runtime of a function.
        
        Args:
            func: Function to time
            
        Returns:
            Tuple of (result, runtime_seconds)
        """
        start_time = time.time()
        result = func()
        runtime = time.time() - start_time
        return result, runtime
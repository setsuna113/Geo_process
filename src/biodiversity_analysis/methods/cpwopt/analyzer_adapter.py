"""Adapter to integrate CP-WOPT with the IAnalyzer interface.

This module provides the integration layer between CP-WOPT tensor factorization
and the biodiversity analysis pipeline.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import time

from src.abstractions.interfaces.analyzer import IAnalyzer, AnalysisResult as IAnalysisResult
from src.abstractions.types import AnalysisResult, ModelMetrics
from src.abstractions.types.tensor_types import TensorData, TensorDecompositionResult
from src.biodiversity_analysis.preprocessing.tensor_preprocessing import TensorPreprocessor
from src.config.cpwopt import get_cpwopt_config
from .core import CPWOPT
from .tensor_metrics import CPWOPTMetricsCalculator, print_metrics_summary
from .minibatch_processor import create_minibatch_processor
from .sparse_handler import AdaptiveSparseStrategy
from .factor_updater import create_factor_updater
from .rank_selector import auto_select_rank, CPWOPTRankSelector

logger = logging.getLogger(__name__)


class CPWOPTAnalyzer(IAnalyzer):
    """CP-WOPT analyzer implementation following IAnalyzer interface."""
    
    def __init__(self, pipeline_config: Any, db: Any):
        """Initialize CP-WOPT analyzer.
        
        Args:
            pipeline_config: Main pipeline configuration object
            db: Database manager instance
        """
        # Store pipeline config and db
        self.pipeline_config = pipeline_config
        self.db = db
        
        # Load method-specific configuration through registry
        from src.config.method_registry import MethodConfigRegistry
        config_loader = MethodConfigRegistry.get_loader('cpwopt')
        self.config = config_loader.load_config()
        
        self.experiment_name = "cpwopt_analysis"
        
        # Initialize components
        self.model = None
        # TensorPreprocessor takes specific parameters, not a config dict
        tensor_config = self.config.get('tensor_construction', {})
        self.preprocessor = TensorPreprocessor(
            missing_value=tensor_config.get('missing_value', np.nan),
            min_observations_per_cell=tensor_config.get('min_observations_per_cell', 1),
            aggregation_method=tensor_config.get('aggregation', 'sum')
        )
        self.metrics_calculator = CPWOPTMetricsCalculator()
        self.sparse_strategy = AdaptiveSparseStrategy(
            self.config.get('sparse', {}).get('sparsity_threshold', 0.95)
        )
        
        # State tracking
        self.is_fitted = False
        self.tensor_data = None
        self.results = None
        self.output_dir = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CPWOPTAnalyzer':
        """Fit CP-WOPT model to data.
        
        Args:
            X: Input data with columns [latitude, longitude, features...]
            y: Not used (unsupervised method)
            
        Returns:
            Self
        """
        logger.info(f"Starting CP-WOPT analysis: {self.experiment_name}")
        start_time = time.time()
        
        try:
            # Validate input
            if not self._validate_input(X):
                raise ValueError("Input validation failed")
            
            # Convert to tensor format
            logger.info("Converting data to tensor format...")
            # Get tensor construction parameters from config
            tensor_config = self.config.get('tensor_construction', {})
            self.tensor_data = self.preprocessor.dataframe_to_tensor(
                X,
                lat_bins=tensor_config.get('lat_bins', 20),
                lon_bins=tensor_config.get('lon_bins', 20),
                transform=tensor_config.get('transform', 'log1p')
            )
            
            logger.info(f"Created tensor with shape: {self.tensor_data.shape}, "
                       f"sparsity: {self.tensor_data.sparsity:.1%}")
            
            # Initialize CP-WOPT model
            alg_config = self.config.get('algorithm', {})
            
            # Convert string values to proper types
            convergence_threshold = alg_config.get('convergence_threshold', 1e-6)
            if isinstance(convergence_threshold, str):
                convergence_threshold = float(convergence_threshold)
            
            # Handle rank selection
            rank_config = alg_config.get('rank', 20)
            if rank_config == 'auto':
                # Automatic rank selection
                logger.info("Performing automatic rank selection...")
                rank_selection_config = self.config.get('rank_selection', {})
                
                # Select rank
                optimal_rank = auto_select_rank(
                    self.tensor_data,
                    method=rank_selection_config.get('method', 'elbow'),
                    min_rank=rank_selection_config.get('min_rank', 2),
                    max_rank=rank_selection_config.get('max_rank', 15),
                    plot=rank_selection_config.get('plot', True),
                    output_dir=self.output_dir if hasattr(self, 'output_dir') else None
                )
                logger.info(f"Automatically selected rank: {optimal_rank}")
            else:
                optimal_rank = int(rank_config)
                logger.info(f"Using fixed rank: {optimal_rank}")
            
            self.model = CPWOPT(
                rank=optimal_rank,
                max_iterations=int(alg_config.get('max_iterations', 1000)),
                convergence_threshold=convergence_threshold,
                init_method=alg_config.get('init_method', 'random'),
                optimizer=alg_config.get('optimizer', 'als'),
                random_state=alg_config.get('random_state', None),
                verbose=True
            )
            
            # Set up mini-batch processor if needed
            if self.config.get('minibatch', {}).get('enabled', True):
                self._setup_minibatch_processing()
            
            # Set up advanced factor updates if needed
            if alg_config.get('optimizer') != 'als':
                self._setup_advanced_optimization()
            
            # Fit the model
            logger.info("Fitting CP-WOPT model...")
            self.model.fit(self.tensor_data)
            
            # Calculate comprehensive metrics
            self._calculate_metrics()
            
            # Mark as fitted
            self.is_fitted = True
            
            # Log summary
            elapsed_time = time.time() - start_time
            logger.info(f"CP-WOPT analysis completed in {elapsed_time:.2f}s")
            
            if self.model.metrics_:
                print_metrics_summary(self.model.metrics_, "CP-WOPT")
            
        except Exception as e:
            logger.error(f"Error during CP-WOPT fitting: {str(e)}")
            raise
            
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data to component space.
        
        Args:
            X: Input data
            
        Returns:
            Component activations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        # Convert to tensor
        tensor_data = self.preprocessor.dataframe_to_tensor(X)
        
        # Transform using the model
        # Note: This would need implementation in core.py
        logger.warning("Transform not fully implemented yet")
        
        # For now, return placeholder
        return np.zeros((len(X), self.model.rank))
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit model and transform data.
        
        Args:
            X: Input data
            y: Not used
            
        Returns:
            Component activations
        """
        self.fit(X, y)
        return self.transform(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Reconstruct data from learned factors.
        
        Args:
            X: Input data (not used, reconstruction is global)
            
        Returns:
            Reconstructed values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict")
        
        # Get reconstruction
        reconstructed = self.model.predict()
        
        # Flatten for compatibility with sklearn interface
        return reconstructed.flatten()
    
    def analyze(self, data, **kwargs) -> AnalysisResult:
        """Run complete CP-WOPT analysis.
        
        Args:
            data: Input data - can be DataFrame or path to parquet file
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with factor decomposition
        """
        # Handle file path input
        if isinstance(data, (str, Path)):
            logger.info(f"Loading data from file: {data}")
            data = pd.read_parquet(data)
        
        # Fit the model
        self.fit(data)
        
        # Create analysis result
        result = AnalysisResult(
            success=True,
            data={
                'method': 'cpwopt',
                'parameters': self.config,
                'metrics': self._create_model_metrics(),
                'factors': self._export_factors(),
                'tensor_shape': self.tensor_data.shape,
                'sparsity': self.tensor_data.sparsity,
                'component_interpretation': self._interpret_components()
            },
            metadata={
                'experiment_name': self.experiment_name,
                'n_iterations': self.model.n_iterations_,
                'training_time': self.model.training_time_,
                'tensor_construction': self.config.get('tensor_construction', {})
            }
        )
        
        self.results = result
        return result
    
    def save_results(self, results: AnalysisResult, exp_name: str, output_dir: Path) -> Path:
        """Save analysis results to directory.
        
        Args:
            results: Analysis results (can be self.results)
            exp_name: Experiment name (unused, for compatibility)
            output_dir: Directory to save results
            
        Returns:
            Path to the main output file
        """
        if not self.is_fitted or self.results is None:
            raise ValueError("No results to save")
        
        # Validate and secure the output path
        output_dir = self._validate_output_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        
        # Save factors
        factors_dir = output_dir / 'factors'
        factors_dir.mkdir(exist_ok=True)
        
        factor_dict = self.model.get_component_matrices()
        for mode_name, factor in factor_dict.items():
            np.save(factors_dir / f'{mode_name}_factors.npy', factor)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            metrics_dict = {
                'reconstruction_error': float(self.model.metrics_.reconstruction_error),
                'relative_error': float(self.model.metrics_.relative_error),
                'fit': float(self.model.metrics_.fit),
                'n_iterations': int(self.model.n_iterations_),
                'training_time': float(self.model.training_time_)
            }
            json.dump(metrics_dict, f, indent=2)
        
        # Save convergence history
        np.save(output_dir / 'convergence_history.npy', 
                np.array(self.model.convergence_history_))
        
        # Save configuration with security sanitization
        config_to_save = self._sanitize_config_for_export(self.config)
            
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        # Save component interpretation
        interpretation = self._interpret_components()
        with open(output_dir / 'component_interpretation.json', 'w') as f:
            json.dump(interpretation, f, indent=2)
        
        # Generate reports using reporters
        self._generate_reports(output_dir)
        
        logger.info(f"Saved CP-WOPT results to: {output_dir}")
        
        # Return path to main results file
        return output_dir / 'metrics.json'
    
    def load_results(self, input_dir: Path) -> None:
        """Load analysis results from directory.
        
        Args:
            input_dir: Directory containing saved results
        """
        input_dir = Path(input_dir)
        
        # This would need full implementation to restore model state
        logger.warning("Load results not fully implemented")
        raise NotImplementedError("Load results not yet implemented")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tensor decomposition.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get feature factor matrix
        factors = self.model.get_component_matrices()
        feature_factors = factors.get('features', factors.get('mode_2'))
        
        # Calculate importance as sum of absolute loadings weighted by component weights
        component_weights = self.model.metrics_.component_weights
        if component_weights is None:
            component_weights = np.ones(self.model.rank) / self.model.rank
        
        # Weighted sum of absolute loadings
        importance = np.zeros(feature_factors.shape[0])
        for r in range(self.model.rank):
            importance += np.abs(feature_factors[:, r]) * component_weights[r]
        
        # Normalize
        importance /= importance.sum()
        
        # Create DataFrame
        feature_names = self.tensor_data.feature_names
        if not feature_names:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def _validate_input(self, X: pd.DataFrame) -> bool:
        """Validate input data.
        
        Args:
            X: Input dataframe
            
        Returns:
            True if valid
        """
        # Accept either latitude/longitude or x/y
        spatial_cols = [['latitude', 'longitude'], ['x', 'y']]
        has_spatial = False
        
        for cols in spatial_cols:
            if all(col in X.columns for col in cols):
                has_spatial = True
                # Rename x/y to latitude/longitude for consistency
                if cols == ['x', 'y']:
                    X.rename(columns={'x': 'longitude', 'y': 'latitude'}, inplace=True)
                break
        
        if not has_spatial:
            logger.error(f"Missing required spatial columns. Need either ['latitude', 'longitude'] or ['x', 'y']")
            return False
        
        # Check for at least one feature column
        spatial_columns = ['latitude', 'longitude', 'x', 'y']
        feature_cols = [col for col in X.columns if col not in spatial_columns]
        if len(feature_cols) == 0:
            logger.error("No feature columns found")
            return False
        
        # Check data types
        if not np.issubdtype(X['latitude'].dtype, np.number):
            logger.error("Latitude must be numeric")
            return False
            
        if not np.issubdtype(X['longitude'].dtype, np.number):
            logger.error("Longitude must be numeric")
            return False
        
        return True
    
    def _setup_minibatch_processing(self):
        """Configure mini-batch processing."""
        mb_config = self.config.get('minibatch', {})
        
        # Create mini-batch processor
        processor = create_minibatch_processor(
            self.tensor_data,
            self.model.rank,
            auto_adjust=True
        )
        
        # Update processor config
        processor.config.batch_size = mb_config.get('batch_size', 10000)
        processor.config.shuffle = mb_config.get('shuffle', False)
        processor.config.drop_last = mb_config.get('drop_last', False)
        
        # Attach to model (would need modification in core.py)
        logger.info(f"Configured mini-batch processing with batch size: "
                   f"{processor.config.batch_size}")
    
    def _setup_advanced_optimization(self):
        """Configure advanced optimization methods."""
        alg_config = self.config.get('algorithm', {})
        optimizer = alg_config.get('optimizer', 'als')
        
        # Create factor updater
        updater = create_factor_updater(
            method=optimizer,
            adaptive=self.config.get('optimization', {}).get('adaptive', False)
        )
        
        logger.info(f"Using {optimizer} optimization method")
    
    def _calculate_metrics(self):
        """Calculate comprehensive metrics."""
        if self.model.factors_ is None:
            return
        
        # Basic metrics are already calculated
        # Add any additional analysis-specific metrics
        metrics = self.model.metrics_
        
        # Calculate explained variance
        metrics.explained_variance = self.metrics_calculator.explained_variance(
            self.tensor_data.tensor,
            self.model.factors_,
            self.tensor_data.weights
        )
        
        # Update model metrics
        self.model.metrics_ = metrics
    
    def _create_model_metrics(self) -> ModelMetrics:
        """Create ModelMetrics object from results."""
        if not self.model.metrics_:
            return ModelMetrics(
                fit=0.0,
                relative_error=1.0,
                iterations=0,
                convergence_history=[],
                final_error=float('inf'),
                component_weights=[]
            )
        
        return ModelMetrics(
            fit=self.model.metrics_.fit,
            relative_error=self.model.metrics_.relative_error,
            iterations=self.model.n_iterations_,
            convergence_history=self.model.convergence_history_,
            final_error=self.model.metrics_.reconstruction_error,
            component_weights=self.model.metrics_.component_weights
        )
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for CP-WOPT.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'rank': 20,
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'initialization': 'random',
            'use_minibatch': True,
            'minibatch_size': 10000,
            'regularization': 0.0,
            'random_state': None,
            'verbose': True,
            'preprocessing': {
                'lat_bins': 20,
                'lon_bins': 20,
                'aggregation': 'sum',
                'transform': 'log1p'
            }
        }
    
    def set_progress_callback(self, callback) -> None:
        """Set progress callback function.
        
        Args:
            callback: Function to call with progress updates (current, total, message)
        """
        self.progress_callback = callback
        # Also set on model if it exists
        if self.model is not None and hasattr(self.model, 'set_progress_callback'):
            self.model.set_progress_callback(callback)
    
    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for CP-WOPT analysis.
        
        Args:
            data: Input dataframe
            
        Returns:
            True if valid, False otherwise
        """
        # This is a wrapper for _validate_input
        return self._validate_input(data)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate analysis parameters.
        
        Args:
            parameters: Dictionary of parameters
            
        Returns:
            True if valid, False otherwise
        """
        # Validate rank
        rank = parameters.get('rank', 20)
        if not isinstance(rank, int) or rank < 1:
            logger.error(f"Invalid rank: {rank}. Must be positive integer.")
            return False
        
        # Validate max_iterations
        max_iter = parameters.get('max_iterations', 1000)
        if not isinstance(max_iter, int) or max_iter < 1:
            logger.error(f"Invalid max_iterations: {max_iter}. Must be positive integer.")
            return False
        
        # Validate convergence_threshold
        conv_thresh = parameters.get('convergence_threshold', 1e-6)
        if not isinstance(conv_thresh, (int, float)) or conv_thresh <= 0:
            logger.error(f"Invalid convergence_threshold: {conv_thresh}. Must be positive number.")
            return False
        
        # Validate preprocessing if provided
        if 'preprocessing' in parameters:
            prep = parameters['preprocessing']
            if not isinstance(prep, dict):
                logger.error("Preprocessing must be a dictionary")
                return False
            
            # Check lat/lon bins if provided
            for bin_param in ['lat_bins', 'lon_bins']:
                if bin_param in prep:
                    bins = prep[bin_param]
                    if not isinstance(bins, int) or bins < 2:
                        logger.error(f"Invalid {bin_param}: {bins}. Must be integer >= 2.")
                        return False
        
        return True
    
    def _export_factors(self) -> Dict[str, Any]:
        """Export factor matrices in serializable format."""
        if not self.model.factors_:
            return {}
        
        factors_dict = self.model.get_component_matrices()
        
        # Convert to lists for JSON serialization
        return {
            mode_name: factor.tolist()
            for mode_name, factor in factors_dict.items()
        }
    
    def _interpret_components(self) -> Dict[str, Any]:
        """Interpret components by finding top features."""
        if not self.model.factors_:
            return {}
        
        interpretation = {}
        
        # Get top features for each component
        n_top = self.config.get('interpretation', {}).get('top_n_features', 10)
        
        for r in range(min(self.model.rank, 10)):  # Limit to first 10 components
            try:
                top_features = self.model.get_top_features(r, n_top)
                
                # Convert to serializable format
                interpretation[f'component_{r}'] = {
                    mode: {
                        'indices': features['indices'].tolist(),
                        'loadings': features['loadings'].tolist()
                    }
                    for mode, features in top_features.items()
                }
                
                # Add interpretation if we have feature names
                if self.tensor_data.feature_names:
                    feature_idx = top_features.get('features', {}).get('indices', [])
                    if len(feature_idx) > 0:
                        top_names = [self.tensor_data.feature_names[i] for i in feature_idx]
                        interpretation[f'component_{r}']['feature_names'] = top_names
                        
            except Exception as e:
                logger.warning(f"Failed to interpret component {r}: {e}")
        
        return interpretation
    
    def _generate_reports(self, output_dir: Path) -> None:
        """Generate comprehensive reports using reporter classes.
        
        Args:
            output_dir: Directory to save reports
        """
        # Import reporters
        from .reporters import (
            FactorReporter, ReconstructionReporter, SpatialReporter,
            ComponentReporter, SummaryReporter, BiomeReporter
        )
        
        # Create decomposition result
        decomposition_result = TensorDecompositionResult(
            factors=self.model.factors_,
            metrics=self.model.metrics_,
            metadata={
                'experiment_name': self.experiment_name,
                'convergence_history': self.model.convergence_history_,
                'training_time': self.model.training_time_,
                'n_iterations': self.model.n_iterations_
            }
        )
        
        # Get reporter config
        reporter_config = self.config.get('reporting', {})
        
        # Generate factor plots
        factor_reporter = FactorReporter(output_dir, reporter_config)
        factor_reporter.report(
            decomposition_result.factors,
            self.tensor_data.shape,
            self.tensor_data.feature_names,
            self.tensor_data.coordinates
        )
        
        # Generate reconstruction analysis
        recon_reporter = ReconstructionReporter(output_dir, reporter_config)
        # Reconstruct tensor for comparison
        reconstructed = self.model.reconstruct()
        recon_reporter.report(
            self.tensor_data.tensor,  # original tensor
            reconstructed,            # reconstructed tensor
            self.tensor_data.weights, # weights
            decomposition_result.metrics  # metrics
        )
        
        # Generate spatial patterns
        # TODO: Fix spatial reporter dimension mismatch
        # spatial_reporter = SpatialReporter(output_dir, reporter_config)
        # 
        # # Get lat/lon bins from preprocessing config
        # preproc_config = self.config.get('preprocessing', {})
        # lat_bins = np.linspace(-90, 90, preproc_config.get('lat_bins', 20) + 1)
        # lon_bins = np.linspace(-180, 180, preproc_config.get('lon_bins', 20) + 1)
        # 
        # spatial_reporter.report(
        #     decomposition_result.factors,
        #     self.tensor_data,
        #     self.tensor_data.coordinates,
        #     lat_bins,
        #     lon_bins
        # )
        
        # Generate component analysis
        comp_reporter = ComponentReporter(output_dir, reporter_config)
        comp_reporter.report(
            decomposition_result.factors,
            self.tensor_data.feature_names,
            decomposition_result.metrics.component_weights,
            decomposition_result.metadata
        )
        
        # Generate biome-like visualizations
        logger.info("Creating biome reporter...")
        biome_reporter = BiomeReporter(output_dir, reporter_config)
        logger.info("Running biome report generation...")
        biome_reporter.report(
            decomposition_result.factors.factors,  # Pass the actual factor list
            self.tensor_data.shape,
            self.tensor_data.feature_names,
            self.tensor_data.coordinates,
            decomposition_result.metrics.component_weights
        )
        logger.info("Biome report generation completed")
        
        # Generate summary report
        # Note: Summary reporter is currently disabled due to interface mismatch
        # It expects reporter results, not reporter objects
        # TODO: Fix this by collecting results from each reporter
        
        # summary_reporter = SummaryReporter(output_dir, reporter_config)
        # experiment_metadata = {
        #     'experiment_name': self.experiment_name,
        #     'config': self.config,
        #     'tensor_shape': self.tensor_data.shape,
        #     'sparsity': self.tensor_data.sparsity,
        #     'feature_names': self.tensor_data.feature_names
        # }
        # summary_reporter.generate_summary_report(
        #     decomposition_result,
        #     experiment_metadata,
        #     reporter_results
        # )
        
        logger.info(f"Generated all reports in: {output_dir}")
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility.
        
        Args:
            deep: If True, return nested parameters
            
        Returns:
            Parameter dictionary
        """
        return self.config if deep else {'config': self.config}
    
    def set_params(self, **params) -> 'CPWOPTAnalyzer':
        """Set parameters for sklearn compatibility.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        if 'config' in params:
            self.config = params['config']
        else:
            # Update config with individual parameters
            for key, value in params.items():
                # Navigate nested config
                keys = key.split('__')
                current = self.config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
        
        return self
    
    def _validate_output_path(self, output_path: Path) -> Path:
        """Validate output path to prevent directory traversal attacks.
        
        Args:
            output_path: Requested output path
            
        Returns:
            Validated absolute path
            
        Raises:
            SecurityError: If path validation fails
        """
        from .exceptions import SecurityError
        
        # Convert to Path object and resolve to absolute path
        output_path = Path(output_path).resolve()
        
        # SECURITY: Check for symbolic links in the path
        current = output_path
        while current != current.parent:
            if current.exists() and current.is_symlink():
                raise SecurityError(
                    f"Symbolic link detected in path: {current}. "
                    "Symbolic links are not allowed for security reasons."
                )
            current = current.parent
        
        # Define allowed base directories with stricter whitelist
        allowed_bases = [
            Path.cwd() / "outputs",
            Path("/scratch") / os.environ.get("USER", ""),  # User-specific scratch
            Path("/tmp") / f"cpwopt_{os.getpid()}",  # Process-specific temp
        ]
        
        # Filter out non-existent bases
        allowed_bases = [base for base in allowed_bases if base.parent.exists()]
        
        # Check if the resolved path is under an allowed directory
        is_allowed = False
        for base in allowed_bases:
            try:
                # Ensure both paths are resolved to catch symlink traversals
                base_resolved = base.resolve()
                output_resolved = output_path.resolve()
                output_resolved.relative_to(base_resolved)
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            raise SecurityError(
                f"Output path '{output_path}' is not in an allowed directory. "
                f"Allowed bases: {[str(b) for b in allowed_bases]}"
            )
        
        # Check for suspicious patterns (expanded list)
        suspicious_patterns = [
            "../", "..", "~", "${", "$(", "`",  # Original patterns
            "%", "\\", "//", "..\\", ".\\",     # Windows-style and URL encoding
            "\x00", "\n", "\r", "|", ";", "&",  # Null bytes and command separators
            "file://", "ftp://", "http://"      # Protocol handlers
        ]
        path_str = str(output_path)
        for pattern in suspicious_patterns:
            if pattern in path_str:
                raise SecurityError(
                    f"Suspicious pattern '{pattern}' found in path: {path_str}"
                )
        
        # SECURITY: Validate filename characters (alphanumeric, dash, underscore, dot)
        import re
        filename = output_path.name
        if filename and not re.match(r'^[a-zA-Z0-9._\-]+$', filename):
            raise SecurityError(
                f"Invalid characters in filename: {filename}. "
                "Only alphanumeric, dash, underscore, and dot are allowed."
            )
        
        return output_path
    
    def _sanitize_config_for_export(self, config: Any) -> Dict[str, Any]:
        """Sanitize configuration for safe export.
        
        Removes sensitive information and ensures only safe data is exported.
        
        Args:
            config: Configuration to sanitize
            
        Returns:
            Sanitized configuration dictionary
        """
        # Handle different config types
        if hasattr(config, 'settings'):
            config_dict = config.settings
        elif hasattr(config, '__dict__') and not isinstance(config, dict):
            config_dict = config.__dict__
        else:
            config_dict = config
        
        # Deep copy to avoid modifying original
        import copy
        sanitized = copy.deepcopy(config_dict)
        
        # Remove sensitive keys
        sensitive_keys = [
            'password', 'token', 'key', 'secret', 'credential',
            'auth', 'private', 'api_key', 'access_key'
        ]
        
        def remove_sensitive(d: Dict[str, Any]) -> None:
            """Recursively remove sensitive keys."""
            if not isinstance(d, dict):
                return
                
            keys_to_remove = []
            for key in d:
                # Check if key contains sensitive patterns
                key_lower = str(key).lower()
                if any(sensitive in key_lower for sensitive in sensitive_keys):
                    keys_to_remove.append(key)
                # Recurse into nested dicts
                elif isinstance(d[key], dict):
                    remove_sensitive(d[key])
                # Sanitize lists of dicts
                elif isinstance(d[key], list):
                    for item in d[key]:
                        if isinstance(item, dict):
                            remove_sensitive(item)
            
            # Remove sensitive keys
            for key in keys_to_remove:
                d[key] = "[REDACTED]"
        
        remove_sensitive(sanitized)
        
        # Ensure all values are JSON serializable
        def make_serializable(obj: Any) -> Any:
            """Convert non-serializable objects."""
            if isinstance(obj, (Path, type)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return make_serializable(obj.__dict__)
            else:
                return obj
        
        sanitized = make_serializable(sanitized)
        
        return sanitized


def create_cpwopt_analyzer(experiment_name: str,
                          config: Optional[Dict[str, Any]] = None,
                          preset: Optional[str] = None) -> CPWOPTAnalyzer:
    """Factory function to create CP-WOPT analyzer.
    
    Args:
        experiment_name: Name of the experiment
        config: Optional configuration overrides
        preset: Optional preset name
        
    Returns:
        Configured CPWOPTAnalyzer
    """
    # Load base configuration
    from src.config.cpwopt import get_cpwopt_config
    base_config = get_cpwopt_config(preset=preset)
    
    # Apply overrides
    if config:
        base_config.update(config)
    
    return CPWOPTAnalyzer(base_config, experiment_name)
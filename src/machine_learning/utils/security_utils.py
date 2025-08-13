"""Security utilities for safe data handling and validation.

Provides path validation, file size checks, and input sanitization.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import yaml
import numpy as np
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError

logger = logging.getLogger(__name__)


class PathValidator:
    """Validate and sanitize file paths for security."""
    
    # Default limits
    MAX_FILE_SIZE_MB = 10000  # 10GB default max
    ALLOWED_EXTENSIONS = {'.parquet', '.csv', '.pkl', '.tsv', '.json', '.yaml', '.yml'}
    
    def __init__(self, 
                 max_file_size_mb: float = MAX_FILE_SIZE_MB,
                 allowed_extensions: Optional[set] = None,
                 allowed_directories: Optional[List[Path]] = None):
        """Initialize path validator.
        
        Args:
            max_file_size_mb: Maximum allowed file size in MB
            allowed_extensions: Set of allowed file extensions
            allowed_directories: List of allowed base directories (None = no restriction)
        """
        self.max_file_size_mb = max_file_size_mb
        self.allowed_extensions = allowed_extensions or self.ALLOWED_EXTENSIONS
        self.allowed_directories = allowed_directories
    
    def validate_path(self, file_path: Union[str, Path]) -> Path:
        """Validate a file path for security and constraints.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If path is invalid or violates constraints
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        # Check for path traversal attempts BEFORE resolving
        if ".." in str(file_path):
            raise ValueError("Path traversal detected in file path")
        
        # Convert to Path object and resolve
        path = Path(file_path).resolve()
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check if it's a file (not directory)
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(
                f"File extension '{path.suffix}' not allowed. "
                f"Allowed extensions: {self.allowed_extensions}"
            )
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum "
                f"allowed size ({self.max_file_size_mb}MB)"
            )
        
        # Check if path is in allowed directories
        if self.allowed_directories:
            is_allowed = False
            for allowed_dir in self.allowed_directories:
                try:
                    path.relative_to(allowed_dir.resolve())
                    is_allowed = True
                    break
                except ValueError:
                    continue
            
            if not is_allowed:
                raise ValueError(
                    f"Path {path} is not in allowed directories"
                )
        
        # Check read permissions
        if not os.access(path, os.R_OK):
            raise PermissionError(f"No read permission for file: {path}")
        
        logger.info(f"Path validated: {path} ({file_size_mb:.1f}MB)")
        return path
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a file after validation.
        
        Args:
            file_path: Path to check
            
        Returns:
            Dictionary with file information
        """
        path = self.validate_path(file_path)
        
        stats = path.stat()
        return {
            'path': str(path),
            'size_mb': stats.st_size / (1024 * 1024),
            'extension': path.suffix,
            'modified': stats.st_mtime,
            'readable': os.access(path, os.R_OK),
            'writable': os.access(path, os.W_OK)
        }


class ConfigValidator:
    """Validate configuration files against schemas."""
    
    # Base schema for ML configuration
    BASE_ML_CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "regression_analysis": {
                "type": "object",
                "properties": {
                    "preprocessing": {"type": "object"},
                    "models": {"type": "object"},
                    "spatial": {"type": "object"},
                    "dml": {"type": "object"},
                    "hypothesis_testing": {"type": "object"},
                    "output": {"type": "object"},
                    "experiments": {"type": "object"}
                },
                "required": ["preprocessing", "models"]
            }
        },
        "required": ["regression_analysis"]
    }
    
    # Detailed preprocessing schema
    PREPROCESSING_SCHEMA = {
        "type": "object",
        "properties": {
            "missing_data": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": ["complete_case", "imputation", "drop"]},
                    "log_message": {"type": "string"}
                },
                "required": ["strategy"]
            },
            "standardization": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["gelman_2sd", "z_score", "none"]},
                    "apply_to": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """Initialize config validator.
        
        Args:
            schema: JSON schema for validation (uses default if None)
        """
        self.schema = schema or self.BASE_ML_CONFIG_SCHEMA
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validate(instance=config, schema=self.schema)
            logger.info("Configuration validated successfully")
            return True
        except JsonSchemaValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            logger.error(f"Failed at path: {' -> '.join(str(p) for p in e.path)}")
            raise
    
    def validate_yaml_file(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
            ValueError: If YAML parsing fails
        """
        # First validate the path
        path_validator = PathValidator(allowed_extensions={'.yaml', '.yml'})
        path = path_validator.validate_path(yaml_path)
        
        # Load YAML safely
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}")
        
        # Validate against schema
        self.validate_config(config)
        
        return config
    
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration by removing dangerous values.
        
        Args:
            config: Configuration to sanitize
            
        Returns:
            Sanitized configuration
        """
        import copy
        sanitized = copy.deepcopy(config)
        
        # Remove any executable code patterns
        def sanitize_value(obj):
            if isinstance(obj, str):
                # Remove potential code injection patterns
                dangerous_patterns = ['eval(', 'exec(', '__import__', 'os.', 'sys.']
                for pattern in dangerous_patterns:
                    if pattern in obj:
                        logger.warning(f"Removed dangerous pattern '{pattern}' from config")
                        return obj.replace(pattern, '')
            elif isinstance(obj, dict):
                return {k: sanitize_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_value(item) for item in obj]
            return obj
        
        return sanitize_value(sanitized)


class DataLoader:
    """Secure H2O data loading with validation."""
    
    def __init__(self, 
                 max_file_size_mb: float = 10000,
                 validate_columns: Optional[List[str]] = None,
                 required_columns: Optional[List[str]] = None):
        """Initialize secure data loader.
        
        Args:
            max_file_size_mb: Maximum file size in MB
            validate_columns: Expected column names
            required_columns: Required column names
        """
        self.path_validator = PathValidator(max_file_size_mb=max_file_size_mb)
        self.validate_columns = validate_columns
        self.required_columns = required_columns or []
    
    def load_data(self, 
                  file_path: Union[str, Path],
                  validate_shape: bool = True,
                  spatial_grid_step: Optional[int] = None) -> 'h2o.H2OFrame':
        """Load data directly into H2O with security validation.
        
        Args:
            file_path: Path to data file
            validate_shape: Whether to validate data shape
            spatial_grid_step: Spatial sampling step (e.g., 100 = take every 100th row)
            
        Returns:
            Loaded H2O Frame
            
        Raises:
            ValueError: If data validation fails
        """
        import h2o
        
        # Validate path
        path = self.path_validator.validate_path(file_path)
        
        # Load directly into H2O
        logger.info(f"Loading data into H2O from {path}")
        
        # H2O.import_file supports parquet, csv, tsv, and more
        if path.suffix not in ['.parquet', '.csv', '.tsv', '.txt']:
            raise ValueError(f"Unsupported file type for H2O: {path.suffix}")
        
        # Load with H2O
        data = h2o.import_file(str(path))
        
        # Apply spatial grid sampling if specified (take every Nth row)
        if spatial_grid_step and spatial_grid_step > 1:
            logger.info(f"Applying spatial grid sampling (step={spatial_grid_step})")
            original_rows = data.nrows
            # Take every Nth row
            row_indices = list(range(0, data.nrows, spatial_grid_step))
            data = data[row_indices, :]
            logger.info(f"Sampled {data.nrows} rows from {original_rows} total rows")
        
        # Validate shape
        if validate_shape:
            if data.nrows == 0:
                raise ValueError("Loaded H2O Frame has no rows")
            
            logger.info(f"Loaded data shape: ({data.nrows}, {data.ncols})")
        
        # Validate columns
        if self.required_columns:
            missing_cols = set(self.required_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        if self.validate_columns:
            unexpected_cols = set(data.columns) - set(self.validate_columns)
            if unexpected_cols:
                logger.warning(f"Unexpected columns found: {unexpected_cols}")
        
        # Check for suspicious column names
        suspicious_patterns = ['__', 'eval', 'exec', 'system']
        for col in data.columns:
            for pattern in suspicious_patterns:
                if pattern in col.lower():
                    raise ValueError(f"Suspicious column name detected: {col}")
        
        logger.info(f"Data loaded successfully: {data.nrows} rows, {data.ncols} columns")
        return data


class InputSanitizer:
    """Sanitize user inputs for security."""
    
    @staticmethod
    def sanitize_string(input_str: str, 
                       max_length: int = 1000,
                       allowed_chars: Optional[str] = None) -> str:
        """Sanitize string input.
        
        Args:
            input_str: String to sanitize
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters
            
        Returns:
            Sanitized string
        """
        import re
        
        # Truncate if too long
        if len(input_str) > max_length:
            logger.warning(f"Truncating string from {len(input_str)} to {max_length} chars")
            input_str = input_str[:max_length]
        
        # Remove control characters
        input_str = ''.join(char for char in input_str if ord(char) >= 32 or char == '\n')
        
        # Apply allowed characters filter if specified
        if allowed_chars:
            pattern = re.compile(allowed_chars)
            input_str = ''.join(char for char in input_str if pattern.match(char))
        
        # Remove potential SQL injection patterns
        sql_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'EXEC', '--', '/*', '*/']
        for pattern in sql_patterns:
            if pattern in input_str.upper():
                logger.warning(f"Removed potential SQL injection pattern: {pattern}")
                input_str = input_str.replace(pattern, '')
                input_str = input_str.replace(pattern.lower(), '')
        
        return input_str
    
    @staticmethod
    def sanitize_number(value: Any, 
                       min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> float:
        """Sanitize numeric input.
        
        Args:
            value: Value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Sanitized float value
            
        Raises:
            ValueError: If value cannot be converted to float
        """
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert {value} to number")
        
        # Check for special values
        if not np.isfinite(num):
            raise ValueError(f"Non-finite number: {num}")
        
        # Apply bounds
        if min_val is not None and num < min_val:
            logger.warning(f"Clamping {num} to minimum {min_val}")
            num = min_val
        
        if max_val is not None and num > max_val:
            logger.warning(f"Clamping {num} to maximum {max_val}")
            num = max_val
        
        return num


# Exception classes for specific error handling
class DataLoadingError(Exception):
    """Raised when data loading fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class SecurityValidationError(Exception):
    """Raised when security validation fails."""
    pass
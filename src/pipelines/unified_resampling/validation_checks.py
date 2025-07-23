# src/pipelines/unified_resampling/validation_checks.py
"""Validation utilities for the resampling pipeline."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

from src.config.config import Config

logger = logging.getLogger(__name__)


class ValidationChecks:
    """Comprehensive validation checks for the resampling pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_files = config.get('data_files', {})
        self.data_dir = Path(config.get('paths.data_dir', 'data'))
        
    def validate_dataset_config(self, dataset_config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single dataset configuration.
        
        Args:
            dataset_config: Dataset configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Use the new dataset resolver for robust validation
        from src.config.dataset_utils import DatasetPathResolver
        
        try:
            resolver = DatasetPathResolver(self.config)
            normalized_config = resolver.validate_dataset_config(dataset_config)
            return True, None
        except ValueError as e:
            return False, str(e)
    
    def validate_resampling_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the resampling configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        resampling_config = self.config.get('resampling', {})
        
        # Check target resolution
        target_resolution = resampling_config.get('target_resolution')
        if not target_resolution or target_resolution <= 0:
            return False, "Invalid target_resolution. Must be a positive number."
        
        if target_resolution > 1.0:
            logger.warning(f"Large target resolution ({target_resolution}°) may result in very coarse data")
        elif target_resolution < 0.001:
            logger.warning(f"Very fine target resolution ({target_resolution}°) may require extensive processing time")
        
        # Check target CRS
        target_crs = resampling_config.get('target_crs', 'EPSG:4326')
        valid_crs = ['EPSG:4326', 'EPSG:3857']  # Add more as needed
        if target_crs not in valid_crs:
            return False, f"Unsupported target_crs '{target_crs}'. Supported: {valid_crs}"
        
        # Check strategies
        strategies = resampling_config.get('strategies', {})
        valid_methods = ['sum', 'bilinear', 'majority', 'nearest', 'mean', 'area_weighted']
        
        for data_type, method in strategies.items():
            if method not in valid_methods:
                return False, f"Invalid resampling method '{method}' for {data_type}. Valid methods: {valid_methods}"
        
        # Check engine
        engine = resampling_config.get('engine', 'numpy')
        if engine not in ['numpy', 'gdal']:
            return False, f"Invalid engine '{engine}'. Must be 'numpy' or 'gdal'"
        
        return True, None
    
    def validate_datasets_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate all datasets configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        datasets_config = self.config.get('datasets', {})
        target_datasets = datasets_config.get('target_datasets', [])
        
        if not target_datasets:
            return False, "No target datasets configured"
        
        # Check each dataset
        enabled_datasets = [ds for ds in target_datasets if ds.get('enabled', True)]
        if len(enabled_datasets) < 2:
            return False, "Need at least 2 enabled datasets for meaningful analysis"
        
        # Validate individual datasets
        for i, dataset_config in enumerate(enabled_datasets):
            is_valid, error_msg = self.validate_dataset_config(dataset_config)
            if not is_valid:
                return False, f"Dataset {i+1} validation failed: {error_msg}"
        
        # Check for duplicate names and band names
        names = [ds['name'] for ds in enabled_datasets]
        band_names = [ds['band_name'] for ds in enabled_datasets]
        
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            return False, f"Duplicate dataset names: {duplicates}"
        
        if len(band_names) != len(set(band_names)):
            duplicates = [name for name in band_names if band_names.count(name) > 1]
            return False, f"Duplicate band names: {duplicates}"
        
        return True, None
    
    def validate_system_requirements(self) -> Tuple[bool, List[str]]:
        """
        Validate system requirements for the pipeline.
        
        Returns:
            Tuple of (all_requirements_met, list_of_warnings)
        """
        warnings = []
        critical_errors = []
        
        # Check available memory
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Estimate memory requirements
            target_resolution = self.config.get('resampling.target_resolution', 0.05)
            datasets_count = len([ds for ds in self.config.get('datasets.target_datasets', []) if ds.get('enabled', True)])
            
            # Rough memory estimate
            estimated_memory_gb = (datasets_count * 4 * (180/target_resolution) * (360/target_resolution) * 4) / (1024**3)
            
            if available_memory_gb < estimated_memory_gb:
                critical_errors.append(f"Insufficient memory: {available_memory_gb:.1f}GB available, ~{estimated_memory_gb:.1f}GB estimated required")
            elif available_memory_gb < estimated_memory_gb * 1.5:
                warnings.append(f"Low memory: {available_memory_gb:.1f}GB available, ~{estimated_memory_gb:.1f}GB estimated required")
                
        except ImportError:
            warnings.append("psutil not available - cannot check memory requirements")
        except Exception as e:
            warnings.append(f"Memory check failed: {e}")
        
        # Check disk space
        try:
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            
            if free_space_gb < 10:
                critical_errors.append(f"Insufficient disk space: {free_space_gb:.1f}GB available")
            elif free_space_gb < 50:
                warnings.append(f"Low disk space: {free_space_gb:.1f}GB available")
                
        except Exception as e:
            warnings.append(f"Disk space check failed: {e}")
        
        # Check data files accessibility
        for path_key, filename in self.data_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                critical_errors.append(f"Data file not found: {file_path}")
            elif not file_path.is_file():
                critical_errors.append(f"Data path is not a file: {file_path}")
            else:
                try:
                    # Try to get basic file info
                    file_size_mb = file_path.stat().st_size / (1024**2)
                    if file_size_mb < 1:
                        warnings.append(f"Small data file ({file_size_mb:.1f}MB): {file_path}")
                except Exception as e:
                    warnings.append(f"Cannot access file stats for {file_path}: {e}")
        
        # Return results
        all_requirements_met = len(critical_errors) == 0
        all_warnings = warnings + [f"CRITICAL: {error}" for error in critical_errors]
        
        return all_requirements_met, all_warnings
    
    def validate_database_connection(self, db_connection) -> Tuple[bool, Optional[str]]:
        """
        Validate database connection and required tables.
        
        Args:
            db_connection: DatabaseManager instance
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Test basic connection
            if not db_connection.test_connection():
                return False, "Database connection test failed"
            
            # Check if required tables exist
            required_tables = ['resampled_datasets']  # Add other required tables
            
            with db_connection.get_connection() as conn:
                cur = conn.cursor()
                
                for table in required_tables:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table,))
                    
                    if not cur.fetchone()[0]:
                        return False, f"Required table '{table}' does not exist. Run database schema creation."
            
            return True, None
            
        except Exception as e:
            return False, f"Database validation failed: {e}"
    
    def create_validation_report(self) -> str:
        """Create a comprehensive validation report."""
        report_lines = [
            "Resampling Pipeline Validation Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Resampling config validation
        is_valid, error_msg = self.validate_resampling_config()
        report_lines.extend([
            "Resampling Configuration:",
            f"  Status: {'✅ Valid' if is_valid else '❌ Invalid'}",
            f"  Message: {error_msg or 'All checks passed'}",
            ""
        ])
        
        # Datasets config validation
        is_valid, error_msg = self.validate_datasets_config()
        report_lines.extend([
            "Datasets Configuration:",
            f"  Status: {'✅ Valid' if is_valid else '❌ Invalid'}",
            f"  Message: {error_msg or 'All checks passed'}",
            ""
        ])
        
        # System requirements
        requirements_met, warnings = self.validate_system_requirements()
        report_lines.extend([
            "System Requirements:",
            f"  Status: {'✅ Met' if requirements_met else '⚠️ Issues Found'}",
        ])
        
        if warnings:
            report_lines.append("  Warnings/Issues:")
            for warning in warnings:
                report_lines.append(f"    - {warning}")
        else:
            report_lines.append("  All requirements met")
        
        return "\n".join(report_lines)


# Import datetime at the top of the file
from datetime import datetime
"""
Biodiversity Data Inspector
===========================

Module for inspecting and reporting properties of biodiversity richness datasets.
Focuses on simple, interpretable metrics and clear documentation of data issues.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import json
from src.abstractions import save_json
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass, asdict

@dataclass
class DatasetProperties:
    """Store properties of a biodiversity dataset"""
    name: str
    path: str
    data_type: str  # 'observed_counts' or 'sdm_prediction'
    
    # Spatial properties
    shape: Tuple[int, int]
    resolution: float  # in degrees
    extent: Dict[str, float]  # min_lon, max_lon, min_lat, max_lat
    crs: str
    
    # Data properties  
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_value: float
    
    # Coverage metrics
    total_cells: int
    valid_cells: int  # non-null cells
    zero_cells: int   # cells with value = 0
    coverage_percent: float
    
    # Distribution metrics
    quantiles: Dict[str, float]  # 5%, 25%, 50%, 75%, 95%
    unique_values: int
    
    # Data quality indicators
    has_negative_values: bool
    has_extreme_outliers: bool
    suspicious_patterns: List[str]
    
    def to_dict(self):
        return asdict(self)


class BiodiversityDataInspector:
    """
    Inspector for biodiversity richness datasets.
    Provides simple, ecologically meaningful analysis of data properties.
    """
    
    KNOWN_DATASETS = {
        'plants': {'type': 'observed_counts', 'description': 'Plant species richness from observations'},
        'animals': {'type': 'observed_counts', 'description': 'Animal (vertebrate) richness from observations'},
        'am_fungi': {'type': 'sdm_prediction', 'description': 'AM fungi richness from SDM predictions'},
        'ecm_fungi': {'type': 'sdm_prediction', 'description': 'EcM fungi richness from SDM predictions'}
    }
    
    def __init__(self, data_dir: str = "/scratch/yl998"):
        """
        Initialize the data inspector.
        
        Parameters
        ----------
        data_dir : str
            Base directory containing biodiversity datasets
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.properties = {}
        
    def load_dataset(self, file_path: str, name: str = None, data_type: str = None) -> DatasetProperties:
        """
        Load and inspect a single biodiversity dataset.
        
        Parameters
        ----------
        file_path : str
            Path to the GeoTIFF file
        name : str
            Name identifier for the dataset
        data_type : str
            Either 'observed_counts' or 'sdm_prediction'
            
        Returns
        -------
        DatasetProperties
            Properties of the loaded dataset
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        # Auto-detect name and type if not provided
        if name is None:
            name = path.stem
            
        if data_type is None:
            if 'fungi' in name.lower() or 'am_' in name.lower() or 'ecm_' in name.lower():
                data_type = 'sdm_prediction'
            else:
                data_type = 'observed_counts'
                
        print(f"\nInspecting dataset: {name}")
        print(f"  Type: {data_type}")
        print(f"  Path: {file_path}")
        
        with rasterio.open(file_path) as src:
            # Read data
            data = src.read(1)
            
            # Get spatial properties
            bounds = src.bounds
            extent = {
                'min_lon': bounds.left,
                'max_lon': bounds.right,
                'min_lat': bounds.bottom,
                'max_lat': bounds.top
            }
            
            # Calculate resolution (assuming degrees)
            pixel_width = (bounds.right - bounds.left) / src.width
            pixel_height = (bounds.top - bounds.bottom) / src.height
            resolution = np.mean([pixel_width, pixel_height])
            
            # Handle nodata values
            nodata = src.nodata
            if nodata is not None:
                valid_mask = data != nodata
                valid_data = data[valid_mask]
            else:
                valid_mask = ~np.isnan(data)
                valid_data = data[valid_mask]
                
            # Calculate basic statistics
            if len(valid_data) > 0:
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
                mean_val = float(np.mean(valid_data))
                median_val = float(np.median(valid_data))
                std_val = float(np.std(valid_data))
                
                # Calculate quantiles
                quantiles = {
                    '5%': float(np.percentile(valid_data, 5)),
                    '25%': float(np.percentile(valid_data, 25)),
                    '50%': float(np.percentile(valid_data, 50)),
                    '75%': float(np.percentile(valid_data, 75)),
                    '95%': float(np.percentile(valid_data, 95))
                }
                
                unique_values = len(np.unique(valid_data))
                has_negative = np.any(valid_data < 0)
                
                # Count zero cells (important for richness data)
                zero_cells = np.sum(valid_data == 0)
                
            else:
                min_val = max_val = mean_val = median_val = std_val = 0
                quantiles = {'5%': 0, '25%': 0, '50%': 0, '75%': 0, '95%': 0}
                unique_values = 0
                has_negative = False
                zero_cells = 0
                
            # Coverage metrics
            total_cells = data.size
            valid_cells = np.sum(valid_mask)
            coverage_percent = (valid_cells / total_cells) * 100
            
            # Detect suspicious patterns
            suspicious_patterns = self._detect_suspicious_patterns(
                data, valid_data, data_type
            )
            
            # Check for extreme outliers
            if len(valid_data) > 0:
                q1, q3 = quantiles['25%'], quantiles['75%']
                iqr = q3 - q1
                has_outliers = np.any(valid_data > q3 + 3 * iqr) if iqr > 0 else False
            else:
                has_outliers = False
                
            # Create properties object
            props = DatasetProperties(
                name=name,
                path=str(file_path),
                data_type=data_type,
                shape=data.shape,
                resolution=resolution,
                extent=extent,
                crs=str(src.crs),
                min_value=min_val,
                max_value=max_val,
                mean_value=mean_val,
                median_value=median_val,
                std_value=std_val,
                total_cells=total_cells,
                valid_cells=valid_cells,
                zero_cells=zero_cells,
                coverage_percent=coverage_percent,
                quantiles=quantiles,
                unique_values=unique_values,
                has_negative_values=has_negative,
                has_extreme_outliers=has_outliers,
                suspicious_patterns=suspicious_patterns
            )
            
            # Store the data and properties
            self.datasets[name] = data
            self.properties[name] = props
            
            return props
            
    def _detect_suspicious_patterns(self, data: np.ndarray, valid_data: np.ndarray, 
                                   data_type: str) -> List[str]:
        """
        Detect suspicious patterns in the data that might indicate issues.
        
        Returns
        -------
        List[str]
            List of detected suspicious patterns with descriptions
        """
        patterns = []
        
        if len(valid_data) == 0:
            patterns.append("No valid data found")
            return patterns
            
        # Check for unrealistic values
        if data_type == 'observed_counts':
            if np.any(valid_data < 0):
                patterns.append(f"Negative values in count data (min: {np.min(valid_data):.2f})")
            if np.any(valid_data > 10000):
                patterns.append(f"Extremely high counts (max: {np.max(valid_data):.0f})")
            if not np.allclose(valid_data, np.round(valid_data)):
                patterns.append("Non-integer values in count data")
                
        elif data_type == 'sdm_prediction':
            if np.any(valid_data < 0):
                patterns.append(f"Negative predictions (min: {np.min(valid_data):.2f})")
            # SDM predictions often normalized 0-1 or represent probabilities
            if np.max(valid_data) > 1000:
                patterns.append(f"Unusually high predictions (max: {np.max(valid_data):.0f})")
                
        # Check for data artifacts
        unique_vals = np.unique(valid_data)
        if len(unique_vals) < 10:
            patterns.append(f"Very few unique values ({len(unique_vals)})")
            
        # Check for suspicious spatial patterns (striping, blocks)
        if data.ndim == 2:
            # Check for row/column artifacts
            row_means = np.nanmean(data, axis=1)
            col_means = np.nanmean(data, axis=0)
            
            if len(row_means) > 10:
                row_cv = np.nanstd(row_means) / np.nanmean(row_means) if np.nanmean(row_means) > 0 else 0
                if row_cv < 0.01:
                    patterns.append("Suspicious horizontal striping pattern")
                    
            if len(col_means) > 10:
                col_cv = np.nanstd(col_means) / np.nanmean(col_means) if np.nanmean(col_means) > 0 else 0
                if col_cv < 0.01:
                    patterns.append("Suspicious vertical striping pattern")
                    
        return patterns
        
    def load_all_datasets(self) -> Dict[str, DatasetProperties]:
        """
        Load all known biodiversity datasets from the data directory.
        
        Returns
        -------
        Dict[str, DatasetProperties]
            Dictionary of dataset properties by name
        """
        # Find all relevant GeoTIFF files
        patterns = [
            "downsampled_rasters/*richness*.tif",
            "resampled_rasters/*Richness*.tif",
            "*.tif"
        ]
        
        found_files = []
        for pattern in patterns:
            found_files.extend(self.data_dir.glob(pattern))
            
        # Load each file
        for file_path in found_files:
            name = file_path.stem.lower()
            
            # Determine type based on filename
            if 'fungi' in name or 'am_' in name or 'ecm_' in name:
                data_type = 'sdm_prediction'
            else:
                data_type = 'observed_counts'
                
            try:
                self.load_dataset(str(file_path), name=name, data_type=data_type)
            except Exception as e:
                print(f"  Warning: Could not load {file_path}: {e}")
                
        return self.properties
        
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a comprehensive report of all loaded datasets.
        
        Parameters
        ----------
        output_path : str
            Optional path to save the report
            
        Returns
        -------
        str
            Formatted report text
        """
        report = []
        report.append("=" * 80)
        report.append("BIODIVERSITY DATASET INSPECTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total datasets loaded: {len(self.properties)}")
        
        obs_count = sum(1 for p in self.properties.values() if p.data_type == 'observed_counts')
        sdm_count = sum(1 for p in self.properties.values() if p.data_type == 'sdm_prediction')
        
        report.append(f"  Observed counts: {obs_count}")
        report.append(f"  SDM predictions: {sdm_count}")
        report.append("")
        
        # Critical issues
        report.append("CRITICAL DATA ISSUES")
        report.append("-" * 40)
        
        issues_found = False
        for name, props in self.properties.items():
            if props.suspicious_patterns:
                issues_found = True
                report.append(f"\n{name}:")
                for pattern in props.suspicious_patterns:
                    report.append(f"  ⚠️  {pattern}")
                    
        if not issues_found:
            report.append("  ✓ No critical issues detected")
            
        report.append("")
        
        # Dataset comparison
        report.append("DATASET PROPERTIES COMPARISON")
        report.append("-" * 40)
        
        # Resolution comparison
        report.append("\nSpatial Resolution:")
        for name, props in self.properties.items():
            report.append(f"  {name:30s}: {props.resolution:.6f}° (~{props.resolution * 111:.2f} km)")
            
        # Coverage comparison
        report.append("\nData Coverage:")
        for name, props in self.properties.items():
            report.append(f"  {name:30s}: {props.coverage_percent:.1f}% valid cells")
            
        # Value range comparison
        report.append("\nValue Ranges:")
        for name, props in self.properties.items():
            report.append(f"  {name:30s}: {props.min_value:.1f} - {props.max_value:.1f} "
                         f"(mean: {props.mean_value:.1f})")
            
        # Method comparison
        report.append("\nData Collection Method:")
        obs_datasets = [name for name, props in self.properties.items() 
                       if props.data_type == 'observed_counts']
        sdm_datasets = [name for name, props in self.properties.items() 
                       if props.data_type == 'sdm_prediction']
        
        report.append(f"  Observed (actual counts): {', '.join(obs_datasets)}")
        report.append(f"  Predicted (SDM models): {', '.join(sdm_datasets)}")
        
        report.append("")
        report.append("ECOLOGICAL INTERPRETATION NOTES")
        report.append("-" * 40)
        report.append("⚠️  CRITICAL: Fungi data are SDM PREDICTIONS (potential habitat)")
        report.append("⚠️  CRITICAL: Plant/vertebrate data are OBSERVED COUNTS (actual occurrences)")
        report.append("⚠️  Direct numerical comparison between these is NOT VALID")
        report.append("✓  Spatial pattern comparison remains ecologically meaningful")
        
        # Join report
        report_text = "\n".join(report)
        
        # Save if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")
            
        return report_text
        
    def save_properties_json(self, output_path: str):
        """
        Save dataset properties as JSON for later use.
        
        Parameters
        ----------
        output_path : str
            Path to save JSON file
        """
        props_dict = {name: props.to_dict() for name, props in self.properties.items()}
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(props_dict, f, indent=2)
            
        print(f"Properties saved to: {output_path}")
# src/raster/validators/value_validator.py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..loaders.base_loader import BaseRasterLoader, RasterMetadata

logger = logging.getLogger(__name__)

class ValueValidator:
    """Validate raster values for species richness data."""
    
    def __init__(self, loader: BaseRasterLoader):
        self.loader = loader
        
        # Expected value ranges for species richness
        self.expected_ranges = {
            'plants': (0, 20000),      # Max ~15000 in tropics
            'vertebrates': (0, 2000),   # Max ~1500 in tropics
            'terrestrial': (0, 5000),   # Combined terrestrial
            'marine': (0, 3000)         # Marine species
        }
    
    def validate_values(self, raster_path: Path, 
                       dataset_type: Optional[str] = None,
                       sample_size: int = 10000) -> Dict[str, Any]:
        """Validate raster values are within expected ranges."""
        
        # Detect dataset type from filename if not provided
        if dataset_type is None:
            dataset_type = self._detect_dataset_type(raster_path)
        
        metadata = self.loader.extract_metadata(raster_path)
        
        # Sample values
        sample_stats = self._sample_values(raster_path, sample_size)
        
        # Validate against expected ranges
        validation_results = self._validate_against_expected(
            sample_stats, dataset_type, metadata
        )
        
        # Check for anomalies
        anomalies = self._detect_anomalies(sample_stats, metadata)
        
        return {
            'dataset_type': dataset_type,
            'sample_stats': sample_stats,
            'validation': validation_results,
            'anomalies': anomalies,
            'metadata': {
                'data_type': metadata.data_type,
                'nodata_value': metadata.nodata_value
            }
        }
    
    def _detect_dataset_type(self, raster_path: Path) -> str:
        """Detect dataset type from filename."""
        name_lower = raster_path.name.lower()
        
        if 'plant' in name_lower:
            return 'plants'
        elif 'vertebrate' in name_lower or 'iucn' in name_lower:
            return 'vertebrates'
        elif 'terrestrial' in name_lower:
            return 'terrestrial'
        elif 'marine' in name_lower:
            return 'marine'
        else:
            return 'unknown'
    
    def _sample_values(self, raster_path: Path, sample_size: int) -> Dict[str, Any]:
        """Sample values from raster for statistics."""
        values = []
        nodata_count = 0
        
        with self.loader.open_lazy(raster_path) as reader:
            metadata = reader.metadata
            
            # Calculate sampling interval
            total_pixels = metadata.width * metadata.height
            interval = max(1, int(np.sqrt(total_pixels / sample_size)))
            
            # Sample systematically
            for row in range(0, metadata.height, interval):
                for col in range(0, metadata.width, interval):
                    # Convert to geographic coordinates
                    x = metadata.bounds[0] + col * metadata.pixel_size[0]
                    y = metadata.bounds[3] + row * metadata.pixel_size[1]
                    
                    value = reader.read_point(x, y)
                    
                    if value is None or value == metadata.nodata_value:
                        nodata_count += 1
                    else:
                        values.append(value)
        
        if not values:
            return {
                'count': 0,
                'nodata_count': nodata_count,
                'has_valid_data': False
            }
        
        values = np.array(values)
        
        return {
            'count': len(values),
            'nodata_count': nodata_count,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'percentiles': {
                '1%': float(np.percentile(values, 1)),
                '5%': float(np.percentile(values, 5)),
                '25%': float(np.percentile(values, 25)),
                '75%': float(np.percentile(values, 75)),
                '95%': float(np.percentile(values, 95)),
                '99%': float(np.percentile(values, 99))
            },
            'has_valid_data': True
        }
    
    def _validate_against_expected(self, sample_stats: Dict, 
                                  dataset_type: str,
                                  metadata: RasterMetadata) -> Dict[str, Any]:
        """Validate values against expected ranges."""
        if not sample_stats.get('has_valid_data', False):
            return {'valid': False, 'reason': 'No valid data found'}
        
        if dataset_type not in self.expected_ranges:
            logger.warning(f"Unknown dataset type: {dataset_type}")
            return {'valid': True, 'warnings': ['Unknown dataset type']}
        
        expected_min, expected_max = self.expected_ranges[dataset_type]
        
        issues = []
        warnings = []
        
        # Check minimum values
        if sample_stats['min'] < expected_min:
            if sample_stats['min'] < 0:
                issues.append(f"Negative values found: min={sample_stats['min']}")
            else:
                warnings.append(f"Values below expected minimum: {sample_stats['min']} < {expected_min}")
        
        # Check maximum values
        if sample_stats['max'] > expected_max:
            ratio = sample_stats['max'] / expected_max
            if ratio > 2.0:
                issues.append(f"Maximum value {ratio:.1f}x higher than expected")
            else:
                warnings.append(f"Maximum value slightly high: {sample_stats['max']} > {expected_max}")
        
        # Check data type compatibility
        if metadata.data_type == 'Float32' or metadata.data_type == 'Float64':
            # Check for fractional species counts
            sample_fractional = sample_stats['mean'] % 1.0
            if sample_fractional > 0.01:
                warnings.append("Floating point data for species counts")
        
        # Check distribution
        if sample_stats['std'] > sample_stats['mean'] * 2:
            warnings.append("Very high variance in values")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'expected_range': (expected_min, expected_max),
            'actual_range': (sample_stats['min'], sample_stats['max'])
        }
    
    def _detect_anomalies(self, sample_stats: Dict, metadata: RasterMetadata) -> List[Dict]:
        """Detect statistical anomalies in the data."""
        anomalies = []
        
        if not sample_stats.get('has_valid_data', False):
            return anomalies
        
        # Check for suspiciously uniform data
        if sample_stats['std'] < 0.1:
            anomalies.append({
                'type': 'uniform_data',
                'description': 'Data appears to be nearly uniform',
                'severity': 'warning'
            })
        
        # Check for discrete vs continuous
        if metadata.data_type in ['Float32', 'Float64']:
            # Sample some values to check if they're actually integers
            is_discrete = abs(sample_stats['mean'] - round(sample_stats['mean'])) < 0.01
            if is_discrete:
                anomalies.append({
                    'type': 'discrete_in_float',
                    'description': 'Integer values stored as float',
                    'severity': 'info'
                })
        
        # Check for outliers using IQR
        q1 = sample_stats['percentiles']['25%']
        q3 = sample_stats['percentiles']['75%']
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if sample_stats['min'] < lower_bound:
                anomalies.append({
                    'type': 'lower_outliers',
                    'description': f'Values below {lower_bound:.1f} detected',
                    'severity': 'info'
                })
            
            if sample_stats['max'] > upper_bound:
                outlier_ratio = (sample_stats['max'] - upper_bound) / upper_bound
                severity = 'warning' if outlier_ratio > 1.0 else 'info'
                anomalies.append({
                    'type': 'upper_outliers',
                    'description': f'Values above {upper_bound:.1f} detected (max: {sample_stats["max"]:.1f})',
                    'severity': severity
                })
        
        # Check NoData distribution
        total_samples = sample_stats['count'] + sample_stats['nodata_count']
        nodata_ratio = sample_stats['nodata_count'] / total_samples if total_samples > 0 else 0
        
        if nodata_ratio > 0.7:
            anomalies.append({
                'type': 'excessive_nodata',
                'description': f'{nodata_ratio:.1%} of sampled pixels are NoData',
                'severity': 'warning'
            })
        
        return anomalies

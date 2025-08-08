"""
Statistical Analysis Configuration

Configuration for analysis parameters, resource management, and execution settings.
"""

import os
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ResourceConfig:
    """Resource management configuration"""
    max_memory_gb: float = 32.0  # Maximum memory usage
    max_workers: int = None  # Auto-detect based on system
    chunk_size: int = 50000  # Default chunk size for processing
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.max_workers is None:
            # Use 75% of available cores, with min 1 and max 32
            self.max_workers = max(1, min(32, int(multiprocessing.cpu_count() * 0.75)))
        
        if self.cache_dir is None:
            self.cache_dir = os.getenv('BIODIV_CACHE_DIR', 
                str(Path.home() / '.cache' / 'biodiversity_analysis'))


@dataclass  
class AnalysisParameters:
    """Analysis-specific parameters"""
    # Coverage analysis
    coverage_sample_fraction: float = 0.1
    coverage_min_points: int = 1000
    
    # Trust maps
    trust_maps_resolution: str = 'medium'  # low, medium, high
    trust_maps_agreement_threshold: float = 0.7
    
    # Latitudinal analysis
    latitudinal_sample_factor: int = 5
    latitudinal_smoothing_window: int = 3
    
    # Global maps
    global_maps_dpi: int = 300
    global_maps_scale_method: str = '95th_percentile'
    
    # Correlation analysis
    correlation_methods: list = None
    correlation_significance_level: float = 0.05
    
    def __post_init__(self):
        if self.correlation_methods is None:
            self.correlation_methods = ['pearson', 'spearman', 'kendall']


class StatisticalAnalysisConfig:
    """Main configuration class for statistical analyses"""
    
    def __init__(self):
        self.resource_config = ResourceConfig()
        self.analysis_params = AnalysisParameters()
        
        # Load from environment variables
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Resource configuration
        if os.getenv('BIODIV_MAX_MEMORY_GB'):
            self.resource_config.max_memory_gb = float(os.getenv('BIODIV_MAX_MEMORY_GB'))
        
        if os.getenv('BIODIV_MAX_WORKERS'):
            self.resource_config.max_workers = int(os.getenv('BIODIV_MAX_WORKERS'))
        
        if os.getenv('BIODIV_CHUNK_SIZE'):
            self.resource_config.chunk_size = int(os.getenv('BIODIV_CHUNK_SIZE'))
        
        if os.getenv('BIODIV_ENABLE_CACHING'):
            self.resource_config.enable_caching = os.getenv('BIODIV_ENABLE_CACHING').lower() == 'true'
        
        # Analysis parameters
        if os.getenv('BIODIV_COVERAGE_SAMPLE_FRACTION'):
            self.analysis_params.coverage_sample_fraction = float(os.getenv('BIODIV_COVERAGE_SAMPLE_FRACTION'))
        
        if os.getenv('BIODIV_TRUST_MAPS_RESOLUTION'):
            self.analysis_params.trust_maps_resolution = os.getenv('BIODIV_TRUST_MAPS_RESOLUTION')
        
        if os.getenv('BIODIV_LATITUDINAL_SAMPLE_FACTOR'):
            self.analysis_params.latitudinal_sample_factor = int(os.getenv('BIODIV_LATITUDINAL_SAMPLE_FACTOR'))
        
        if os.getenv('BIODIV_GLOBAL_MAPS_DPI'):
            self.analysis_params.global_maps_dpi = int(os.getenv('BIODIV_GLOBAL_MAPS_DPI'))
    
    def get_analysis_config(self, analysis_type: str) -> Dict[str, Any]:
        """Get configuration specific to an analysis type"""
        
        base_config = {
            'max_workers': self.resource_config.max_workers,
            'chunk_size': self.resource_config.chunk_size,
            'max_memory_gb': self.resource_config.max_memory_gb,
            'enable_caching': self.resource_config.enable_caching,
            'cache_dir': self.resource_config.cache_dir
        }
        
        if analysis_type == 'coverage':
            base_config.update({
                'sample_fraction': self.analysis_params.coverage_sample_fraction,
                'min_points': self.analysis_params.coverage_min_points
            })
        
        elif analysis_type == 'trust_maps':
            base_config.update({
                'resolution': self.analysis_params.trust_maps_resolution,
                'agreement_threshold': self.analysis_params.trust_maps_agreement_threshold
            })
        
        elif analysis_type == 'latitudinal':
            base_config.update({
                'sample_factor': self.analysis_params.latitudinal_sample_factor,
                'smoothing_window': self.analysis_params.latitudinal_smoothing_window
            })
        
        elif analysis_type == 'global_maps':
            base_config.update({
                'dpi': self.analysis_params.global_maps_dpi,
                'scale_method': self.analysis_params.global_maps_scale_method
            })
        
        elif analysis_type == 'correlation':
            base_config.update({
                'methods': self.analysis_params.correlation_methods,
                'significance_level': self.analysis_params.correlation_significance_level
            })
        
        return base_config
    
    def get_summary(self) -> str:
        """Get human-readable configuration summary"""
        lines = ["Statistical Analysis Configuration:", "=" * 40]
        
        # Resource configuration
        lines.append("Resource Management:")
        lines.append(f"  Max Memory: {self.resource_config.max_memory_gb} GB")
        lines.append(f"  Max Workers: {self.resource_config.max_workers}")
        lines.append(f"  Chunk Size: {self.resource_config.chunk_size:,}")
        lines.append(f"  Caching: {'Enabled' if self.resource_config.enable_caching else 'Disabled'}")
        lines.append(f"  Cache Dir: {self.resource_config.cache_dir}")
        
        # Analysis parameters
        lines.append("\nAnalysis Parameters:")
        lines.append(f"  Coverage Sample Fraction: {self.analysis_params.coverage_sample_fraction}")
        lines.append(f"  Trust Maps Resolution: {self.analysis_params.trust_maps_resolution}")
        lines.append(f"  Latitudinal Sample Factor: {self.analysis_params.latitudinal_sample_factor}")
        lines.append(f"  Global Maps DPI: {self.analysis_params.global_maps_dpi}")
        lines.append(f"  Correlation Methods: {', '.join(self.analysis_params.correlation_methods)}")
        
        return '\n'.join(lines)


# Global instance for easy import
statistical_analysis_config = StatisticalAnalysisConfig()
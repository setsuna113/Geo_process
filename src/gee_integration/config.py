"""GEE Integration Configuration

Handles configuration for the GEE workflow including dual resolution support.
Centralized configuration for all GEE parameters including authentication.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from dataclasses import dataclass, field

from src.infrastructure.logging import get_logger
from config.gee.constants import DEFAULT_LAT_MIN, DEFAULT_LAT_MAX

logger = get_logger(__name__)




@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    band_name: str
    path: str  # Path to the dataset file
    description: Optional[str] = None
    no_data_value: Optional[float] = None
    asset_name: Optional[str] = None
    data_type: str = "continuous"  # continuous, count, categorical
    resampling: str = "bilinear"  # bilinear, bicubic, max, min, mean, mode


# ResolutionConfig class removed - now using resolution_settings dict format


@dataclass
class AdditionalLayerConfig:
    """Configuration for additional data layers from GEE"""
    enabled: bool
    source: str
    band: Optional[str] = None
    bands: Optional[List[str]] = None
    variables: Optional[List[str]] = None
    description: Optional[str] = None


@dataclass
class MaskingConfig:
    """Configuration for masking options"""
    enabled: bool = True
    latitude_bounds: Dict[str, float] = field(default_factory=lambda: {"min": DEFAULT_LAT_MIN, "max": DEFAULT_LAT_MAX})


@dataclass
class GEEConfig:
    """Complete GEE workflow configuration"""
    # Experiment
    experiment_name: str = "biodiversity_analysis"
    
    # Authentication
    auth_method: str = "service_account"
    service_account_key: Optional[str] = None
    project_id: Optional[str] = None
    
    # Recursive resolution settings (base + outputs)
    resolution_settings: Dict = field(default_factory=lambda: {
        'base_resolution': 0.016666666666667,
        'outputs': {'1x': True}
    })
    
    # Input datasets
    biodiversity_datasets: List[DatasetConfig] = field(default_factory=list)
    
    # Additional layers from GEE (climate, elevation, etc.)
    additional_layers: Dict[str, AdditionalLayerConfig] = field(default_factory=dict)
    
    # Masking configuration
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    
    # Upload settings
    gcs_bucket: Optional[str] = None
    asset_prefix: str = "biodiversity"
    chunk_size_mb: int = 100
    parallel_uploads: int = 4
    
    # Processing settings
    merge_strategy: str = "addBands"
    tile_scale: int = 16
    max_pixels: float = 1e13
    
    # Export settings
    export_destination: str = "cloud_storage"
    export_folder: str = "merged_outputs"
    export_format: str = "CSV"  # For parquet conversion
    export_timeout: int = 7200  # 2 hours default, configurable
    export_check_interval: int = 30  # Check every 30 seconds
    file_format_options: Optional[Dict] = None
    export_max_pixels: float = 1e13
    hemisphere_split: Optional[Dict] = None  # Hemisphere splitting config
    export: Optional[Dict] = None  # Full export config including bounds
    
    # Output settings
    output_dir: Path = Path("data/gee_merged")
    parquet_dir: Path = Path("data/merged")
    temp_dir: Path = Path("data/temp/gee")
    cleanup_temp: bool = True
    
    # Performance settings
    parquet_chunk_size: int = 100000  # Rows per chunk for parquet conversion
    
    # Validation settings
    check_alignment: bool = True
    check_bounds: bool = True
    tolerance: float = 0.0001
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'GEEConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Expand environment variables
        data = cls._expand_env_vars(data)
        
        # Parse resolution settings (new format only)
        resolution_settings = data.get('resolution_settings', {
            'base_resolution': 0.016666666666667,
            'outputs': {'1x': True}
        })
        
        # Parse biodiversity datasets
        biodiv_datasets = []
        for dataset_data in data.get('biodiversity_datasets', []):
            biodiv_datasets.append(DatasetConfig(**dataset_data))
        
        # Parse additional layers (climate, elevation, etc.)
        additional_layers = {}
        layers_config = data.get('additional_layers', {})
        for layer_name, layer_data in layers_config.items():
            if isinstance(layer_data, dict):
                additional_layers[layer_name] = AdditionalLayerConfig(
                    enabled=layer_data.get('enabled', False),
                    source=layer_data.get('source', ''),
                    band=layer_data.get('band'),
                    bands=layer_data.get('bands'),
                    variables=layer_data.get('variables'),
                    description=layer_data.get('description')
                )
        
        # Parse masking configuration
        mask_config = data.get('masking', {})
        masking = MaskingConfig(
            enabled=mask_config.get('enabled', True),
            latitude_bounds={
                'min': mask_config.get('latitude_bounds', {}).get('min', DEFAULT_LAT_MIN),
                'max': mask_config.get('latitude_bounds', {}).get('max', DEFAULT_LAT_MAX)
            }
        )
        
        # Create config instance
        return cls(
            experiment_name=data.get('experiment_name', 'biodiversity_analysis'),
            auth_method=data.get('gee', {}).get('auth', {}).get('method', 'service_account'),
            service_account_key=data.get('gee', {}).get('auth', {}).get('service_account_key'),
            project_id=data.get('gee', {}).get('auth', {}).get('project_id'),
            resolution_settings=resolution_settings,
            biodiversity_datasets=biodiv_datasets,
            additional_layers=additional_layers,
            masking=masking,
            gcs_bucket=data.get('gee', {}).get('upload', {}).get('bucket'),
            asset_prefix=data.get('gee', {}).get('upload', {}).get('asset_prefix', 'biodiversity'),
            chunk_size_mb=data.get('gee', {}).get('upload', {}).get('chunk_size_mb', 100),
            parallel_uploads=data.get('gee', {}).get('upload', {}).get('parallel_uploads', 4),
            merge_strategy=data.get('gee', {}).get('processing', {}).get('merge_strategy', 'addBands'),
            tile_scale=data.get('gee', {}).get('processing', {}).get('tileScale', 16),
            max_pixels=data.get('gee', {}).get('processing', {}).get('maxPixels', 1e13),
            export_destination=data.get('gee', {}).get('export', {}).get('destination', 'cloud_storage'),
            export_folder=data.get('gee', {}).get('export', {}).get('folder', 'merged_outputs'),
            export_format=data.get('gee', {}).get('export', {}).get('format', 'CSV'),
            file_format_options=data.get('gee', {}).get('export', {}).get('file_format_options'),
            export_timeout=data.get('gee', {}).get('export', {}).get('timeout', 7200),
            export_check_interval=data.get('gee', {}).get('export', {}).get('check_interval', 30),
            export_max_pixels=data.get('gee', {}).get('export', {}).get('maxPixels', 1e13),
            hemisphere_split=data.get('export', {}).get('hemisphere_split'),
            export=data.get('export'),  # Pass the full export config
            output_dir=Path(data.get('gee', {}).get('output', {}).get('local_dir', 'data/gee_merged')),
            parquet_dir=Path(data.get('gee', {}).get('output', {}).get('parquet_dir', 'data/merged')),
            temp_dir=Path(data.get('gee', {}).get('output', {}).get('temp_dir', 'data/temp/gee')),
            cleanup_temp=data.get('gee', {}).get('output', {}).get('cleanup_temp', True),
            parquet_chunk_size=data.get('gee', {}).get('performance', {}).get('parquet_chunk_size', 100000),
            check_alignment=data.get('gee', {}).get('validation', {}).get('check_alignment', True),
            check_bounds=data.get('gee', {}).get('validation', {}).get('check_bounds', True),
            tolerance=data.get('gee', {}).get('validation', {}).get('tolerance', 0.0001),
        )
    
    @staticmethod
    def _expand_env_vars(data: Any) -> Any:
        """Recursively expand environment variables in config"""
        if isinstance(data, dict):
            return {k: GEEConfig._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [GEEConfig._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            env_var = data[2:-1]
            return os.environ.get(env_var, data)
        elif isinstance(data, str) and '~' in data:
            # Expand home directory
            return os.path.expanduser(data)
        return data
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check authentication
        if self.auth_method == 'service_account' and not self.service_account_key:
            issues.append("Service account key not provided")
        
        # Check GCS bucket
        if self.export_destination == 'cloud_storage' and not self.gcs_bucket:
            issues.append("GCS bucket not provided for cloud storage export")
        
        # Check datasets exist
        for dataset in self.biodiversity_datasets:
            if not Path(dataset.path).exists():
                issues.append(f"Dataset file not found: {dataset.path}")
        
        # Check resolution settings
        if not self.resolution_settings:
            issues.append("No resolution_settings defined")
        else:
            outputs = self.resolution_settings.get('outputs', {})
            if not outputs:
                issues.append("No outputs defined in resolution_settings")
            elif not any(outputs.values()):
                issues.append("No resolutions enabled for export in resolution_settings")
            
            base_res = self.resolution_settings.get('base_resolution')
            if not base_res or base_res <= 0:
                issues.append("Invalid or missing base_resolution in resolution_settings")
        
        return issues
    
    def get_resolution_outputs(self) -> Dict[str, bool]:
        """Get resolution outputs configuration"""
        return self.resolution_settings.get('outputs', {})
    
    def get_base_resolution(self) -> float:
        """Get base resolution in degrees"""
        return self.resolution_settings.get('base_resolution', 0.016666666666667)
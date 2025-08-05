"""GEE Integration Configuration

Handles configuration for the GEE workflow including dual resolution support.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from dataclasses import dataclass, field

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    band_name: str
    gcs_path: Optional[str] = None  # For new config
    path: Optional[str] = None  # For old config compatibility
    description: Optional[str] = None
    no_data_value: Optional[float] = None
    no_data: Optional[float] = None  # Old field name
    asset_name: Optional[str] = None
    data_type: str = "continuous"  # continuous, count, categorical
    resampling: str = "bilinear"  # bilinear, bicubic, max, min, mean, mode
    
    def __post_init__(self):
        # Handle old vs new field names
        if self.gcs_path and not self.path:
            self.path = self.gcs_path
        if self.no_data_value is not None and self.no_data is None:
            self.no_data = self.no_data_value


@dataclass
class ResolutionConfig:
    """Configuration for output resolution"""
    name: str
    resolution: float  # in degrees
    description: str


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
    latitude_bounds: Dict[str, float] = field(default_factory=lambda: {"min": -60, "max": 75})


@dataclass
class GEEConfig:
    """Complete GEE workflow configuration"""
    # Experiment
    experiment_name: str = "biodiversity_analysis"
    
    # Authentication
    auth_method: str = "service_account"
    service_account_key: Optional[str] = None
    project_id: Optional[str] = None
    
    # Target resolutions
    resolutions: Dict[str, ResolutionConfig] = field(default_factory=dict)
    
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
        
        # Parse resolutions (check both old and new locations)
        resolutions = {}
        res_data_source = data.get('resolutions', {}) or data.get('gee', {}).get('target_resolutions', {})
        for res_name, res_data in res_data_source.items():
            # Handle both old and new resolution format
            if 'scale' in res_data:
                resolutions[res_name] = ResolutionConfig(
                    name=res_name,
                    resolution=res_data['scale'] / 111320.0,  # Convert meters to degrees (approximate)
                    description=res_data.get('description', f'{res_name} resolution')
                )
            else:
                resolutions[res_name] = ResolutionConfig(**res_data)
        
        # Parse biodiversity datasets (check both old and new locations)
        biodiv_datasets = []
        biodiv_source = data.get('biodiversity_datasets', []) or data.get('gee', {}).get('inputs', {}).get('biodiversity', [])
        for dataset_data in biodiv_source:
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
                'min': mask_config.get('latitude_bounds', {}).get('min', -60),
                'max': mask_config.get('latitude_bounds', {}).get('max', 75)
            }
        )
        
        # Create config instance
        return cls(
            experiment_name=data.get('experiment_name', 'biodiversity_analysis'),
            auth_method=data.get('gee', {}).get('auth', {}).get('method', 'service_account'),
            service_account_key=data.get('gee', {}).get('auth', {}).get('service_account_key'),
            project_id=data.get('gee', {}).get('auth', {}).get('project_id'),
            resolutions=resolutions,
            biodiversity_datasets=biodiv_datasets,
            additional_layers=additional_layers,
            masking=masking,
            gcs_bucket=data.get('gcs', {}).get('bucket') or data.get('gee', {}).get('upload', {}).get('bucket'),
            asset_prefix=data.get('earth_engine', {}).get('asset_prefix') or data.get('gee', {}).get('upload', {}).get('asset_prefix', 'biodiversity'),
            chunk_size_mb=data.get('gcs', {}).get('chunk_size_mb', 100) or data.get('gee', {}).get('upload', {}).get('chunk_size_mb', 100),
            parallel_uploads=data.get('gcs', {}).get('parallel_uploads', 4) or data.get('gee', {}).get('upload', {}).get('parallel_uploads', 4),
            merge_strategy=data.get('processing', {}).get('merge_strategy', 'addBands') or data.get('gee', {}).get('processing', {}).get('merge_strategy', 'addBands'),
            tile_scale=data.get('processing', {}).get('tileScale', 16) or data.get('gee', {}).get('processing', {}).get('tileScale', 16),
            max_pixels=data.get('processing', {}).get('maxPixels', 1e13) or data.get('gee', {}).get('processing', {}).get('maxPixels', 1e13),
            export_destination=data.get('export', {}).get('destination', 'cloud_storage') or data.get('gee', {}).get('export', {}).get('destination', 'cloud_storage'),
            export_folder=data.get('gcs', {}).get('export_folder') or data.get('gee', {}).get('export', {}).get('folder', 'merged_outputs'),
            export_format=data.get('export', {}).get('format', 'GeoTIFF') or data.get('gee', {}).get('export', {}).get('format', 'CSV'),
            file_format_options=data.get('gee', {}).get('export', {}).get('file_format_options'),
            export_timeout=data.get('gee', {}).get('export', {}).get('timeout', 7200),
            export_check_interval=data.get('gee', {}).get('export', {}).get('check_interval', 30),
            export_max_pixels=data.get('gee', {}).get('export', {}).get('maxPixels', 1e13),
            hemisphere_split=data.get('export', {}).get('hemisphere_split'),
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
        
        # Check resolutions
        if not self.resolutions:
            issues.append("No target resolutions defined")
        
        return issues
    
    def get_resolution(self, name: str) -> ResolutionConfig:
        """Get resolution config by name"""
        if name not in self.resolutions:
            raise ValueError(f"Resolution '{name}' not found. Available: {list(self.resolutions.keys())}")
        return self.resolutions[name]
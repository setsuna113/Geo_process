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
    path: str
    name: str
    band_name: str
    data_type: str = "continuous"  # continuous, count, categorical
    resampling: str = "bilinear"  # bilinear, bicubic, max, min, mean, mode
    no_data: Optional[float] = None


@dataclass
class ResolutionConfig:
    """Configuration for output resolution"""
    name: str
    resolution: float  # in degrees
    description: str


@dataclass
class GEEConfig:
    """Complete GEE workflow configuration"""
    # Authentication
    auth_method: str = "service_account"
    service_account_key: Optional[str] = None
    project_id: Optional[str] = None
    
    # Target resolutions
    resolutions: Dict[str, ResolutionConfig] = field(default_factory=dict)
    
    # Input datasets
    biodiversity_datasets: List[DatasetConfig] = field(default_factory=list)
    climate_variables: List[str] = field(default_factory=list)
    
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
    
    # Output settings
    output_dir: Path = Path("data/gee_merged")
    parquet_dir: Path = Path("data/merged")
    temp_dir: Path = Path("data/temp/gee")
    cleanup_temp: bool = True
    
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
        
        # Parse resolutions
        resolutions = {}
        for res_name, res_data in data.get('gee', {}).get('target_resolutions', {}).items():
            resolutions[res_name] = ResolutionConfig(**res_data)
        
        # Parse biodiversity datasets
        biodiv_datasets = []
        for dataset_data in data.get('gee', {}).get('inputs', {}).get('biodiversity', []):
            biodiv_datasets.append(DatasetConfig(**dataset_data))
        
        # Parse climate variables
        climate_config = data.get('gee', {}).get('inputs', {}).get('climate', {})
        if climate_config.get('variables') == 'all':
            # All 19 bioclimatic variables
            climate_vars = [f'bio{i:02d}' for i in range(1, 20)]
        else:
            variables = climate_config.get('variables', [])
            if variables and isinstance(variables[0], str):
                # Direct list of variable names
                climate_vars = variables
            else:
                # List of dicts with 'name' field
                climate_vars = [v['name'] for v in variables]
        
        # Create config instance
        return cls(
            auth_method=data.get('gee', {}).get('auth', {}).get('method', 'service_account'),
            service_account_key=data.get('gee', {}).get('auth', {}).get('service_account_key'),
            project_id=data.get('gee', {}).get('auth', {}).get('project_id'),
            resolutions=resolutions,
            biodiversity_datasets=biodiv_datasets,
            climate_variables=climate_vars,
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
            output_dir=Path(data.get('gee', {}).get('output', {}).get('local_dir', 'data/gee_merged')),
            parquet_dir=Path(data.get('gee', {}).get('output', {}).get('parquet_dir', 'data/merged')),
            temp_dir=Path(data.get('gee', {}).get('output', {}).get('temp_dir', 'data/temp/gee')),
            cleanup_temp=data.get('gee', {}).get('output', {}).get('cleanup_temp', True),
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
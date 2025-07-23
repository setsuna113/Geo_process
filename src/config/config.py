import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from . import defaults

class Config:
    """Configuration manager with YAML override support."""

    def __init__(self, config_file: Optional[Path] = None):
        self.settings = self.load_defaults()

        # Detect test mode and override database settings if needed
        if self._is_test_mode():
            self._apply_test_database_config()

        # Only load config.yml if NOT in test mode
        if not self._is_test_mode():
            # Auto-discover config.yml if not specified
            if config_file is None:
                # Look for config.yml in project root
                project_root = Path(__file__).parent.parent.parent
                potential_config = project_root / 'config.yml'
                if potential_config.exists():
                    config_file = potential_config

            if config_file and config_file.exists():
                self._load_yaml_config(config_file, preserve_test_db=False)
        else:
            print("🧪 Test mode detected - using test database configuration (port 5432)")
            print("🧪 Ignoring config.yml completely in test mode")

        self._ensure_directories()
    
    def _is_test_mode(self) -> bool:
        """Detect if we're running in test mode."""
        import sys
        return (
            'pytest' in sys.modules or
            'unittest' in sys.modules or
            any('test_' in module for module in sys.modules)
        )
    
    def _apply_test_database_config(self):
        """Apply test-safe database configuration."""
        self.settings['database'] = {
            'host': 'localhost',
            'port': 5432,  # Standard PostgreSQL port for testing
            'database': 'geoprocess_db',  # Use existing database for tests
            'user': 'jason',
            'password': '123456',
            'max_connections': 5,
            'connection_timeout': 10,
            'retry_attempts': 3
        }
        print("🧪 Test mode detected - using test database configuration (port 5432)")
    
    def load_defaults(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            'database': defaults.DATABASE.copy(),
            'grids': defaults.GRIDS.copy(),
            'logging': defaults.LOGGING.copy(),
            'processing': defaults.PROCESSING.copy(),
            'raster_processing': defaults.RASTER_PROCESSING.copy(),
            'features': defaults.FEATURES.copy(),
            'output_formats': defaults.OUTPUT_FORMATS.copy(),
            'processing_bounds': defaults.PROCESSING_BOUNDS.copy(),
            'species_filters': defaults.SPECIES_FILTERS.copy(),
            'data_preparation': defaults.DATA_PREPARATION.copy(),
            'data_cleaning': defaults.DATA_CLEANING.copy(),
            'testing': defaults.TESTING.copy(),
            'data_files': defaults.DATA_FILES.copy(),
            'datasets': defaults.DATASETS.copy(),
            'resampling': defaults.RESAMPLING.copy(),
            'paths': {
                'project_root': defaults.PROJECT_ROOT,
                'data_dir': defaults.DATA_DIR,
                'rawdata_dir': defaults.RAWDATA_DIR,
                'logs_dir': defaults.LOGS_DIR,
            }
        }
    
    def _load_yaml_config(self, config_file: Path, preserve_test_db: bool = False):
        """Load and merge configuration from a YAML file."""
        with open(config_file, 'r') as file:
            yaml_config = yaml.safe_load(file)
            if yaml_config:
                # In test mode, preserve certain configs from test defaults
                if preserve_test_db:
                    yaml_config = dict(yaml_config)  # Make a copy
                    
                    # Don't override test database config
                    if 'database' in yaml_config:
                        del yaml_config['database']
                        print("🧪 Ignoring database config from YAML in test mode")
                    
                    # Don't override raster processing config (tests expect defaults)
                    if 'raster_processing' in yaml_config:
                        del yaml_config['raster_processing']
                        print("🧪 Ignoring raster_processing config from YAML in test mode")
                
                self._deep_merge(self.settings, yaml_config)
    
    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base dictionary."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist and are accessible."""
        for path_key in ['data_dir', 'logs_dir']:
            path = self.settings['paths'][path_key]
            # Convert to Path object if it's a string (from YAML)
            if isinstance(path, str):
                path = Path(path)
                self.settings['paths'][path_key] = path
            
            # Only try to create directories if we have permission
            try:
                path.mkdir(parents=True, exist_ok=True)
            except (PermissionError, FileNotFoundError):
                # Skip directory creation if we don't have permission or parent doesn't exist
                # This allows the config to load even if cluster paths don't exist locally
                pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self.settings

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    @property
    def database(self) -> Dict[str, Any]:
        return self.settings['database']
    
    @property
    def grids(self) -> Dict[str, Any]:
        return self.settings['grids']
    
    @property
    def raster_processing(self) -> Dict[str, Any]:
        return self.settings['raster_processing']
    
    @property
    def output_formats(self) -> Dict[str, Any]:
        return self.settings['output_formats']
    
    @property
    def processing_bounds(self) -> Dict[str, Any]:
        return self.settings['processing_bounds']
    
    @property
    def paths(self) -> Dict[str, Any]:
        return self.settings['paths']
    
    @property
    def processing(self) -> Dict[str, Any]:
        return self.settings['processing']
    
    @property
    def species_filters(self) -> Dict[str, Any]:
        return self.settings['species_filters']
    
    @property
    def data_preparation(self) -> Dict[str, Any]:
        return self.settings['data_preparation']
    
    @property
    def data_cleaning(self) -> Dict[str, Any]:
        return self.settings['data_cleaning']
    
    @property 
    def config(self) -> Dict[str, Any]:
        """Access to settings dict for backward compatibility."""
        return self.settings
    
    @property
    def testing(self) -> Dict[str, Any]:
        return self.settings.get('testing', {})
    
    @config.setter
    def config(self, value: Dict[str, Any]):
        """Set settings dict for test compatibility."""
        self.settings = value

# Global configuration instance
config = Config()

# Update src/config/config.py to include new configuration sections

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

        # Load config.yml unless explicitly in test mode
        if not self._is_test_mode():
            # Auto-discover config.yml with comprehensive fallback
            if config_file is None:
                config_file = self._find_config_file()

            if config_file and config_file.exists():
                try:
                    self._load_yaml_config(config_file, preserve_test_db=False)
                    print(f"✅ Loaded configuration from {config_file}")
                except Exception as e:
                    print(f"⚠️  Config file loading failed: {e} - using defaults")
            else:
                print("ℹ️  No config.yml found - using defaults only")
        else:
            print("🧪 Test mode detected - using test database configuration (port 5432)")
            print("🧪 Ignoring config.yml completely in test mode")

        self._ensure_directories()
    
    def _find_config_file(self) -> Optional[Path]:
        """Find config.yml with multiple fallback locations."""
        project_root = Path(__file__).parent.parent.parent
        
        potential_locations = [
            project_root / 'config.yml',
            project_root / 'config' / 'config.yml', 
            Path.cwd() / 'config.yml',
            Path.home() / '.geo' / 'config.yml',
        ]
        
        for location in potential_locations:
            if location.exists() and location.is_file():
                return location
                
        return None
    
    def _is_test_mode(self) -> bool:
        """Detect if we're running in test mode - ONLY for actual testing."""
        import os
        
        # STRICT test mode detection - only if explicitly set
        # Never auto-detect to avoid production failures
        return (
            os.environ.get('FORCE_TEST_MODE', 'false').lower() == 'true' or
            os.environ.get('PYTEST_CURRENT_TEST') is not None
        )
    
    def _apply_test_database_config(self):
        """Apply test-safe database configuration."""
        import os
        test_user = os.getenv('USER', 'testuser')
        
        # Respect explicit DB_NAME if set, otherwise use test database
        db_name = os.getenv('DB_NAME')
        if not db_name:
            db_name = f'{test_user}_geo_test_db'  # User-specific test database
        
        self.settings['database'] = {
            'host': os.getenv('DB_HOST', '/var/run/postgresql'),  # Use Unix socket for local connection
            'port': 5432,  # Standard PostgreSQL port for testing
            'database': db_name,
            'user': os.getenv('DB_USER', test_user),
            'password': os.getenv('DB_PASSWORD', ''),
            'max_connections': 5,
            'connection_timeout': 10,
            'retry_attempts': 3,
            'auto_create_database': True,
            'fallback_databases': ['postgres', 'template1']
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
            'storage': defaults.STORAGE.copy(),
            # New configuration sections
            'progress_monitoring': defaults.PROGRESS_MONITORING.copy(),
            'checkpointing': defaults.CHECKPOINTING.copy(),
            'timeouts': defaults.TIMEOUTS.copy(),
            'process_management': defaults.PROCESS_MANAGEMENT.copy(),
            'memory_management': defaults.MEMORY_MANAGEMENT.copy(),
            'paths': defaults.PATHS.copy(),
            'pipeline': defaults.PIPELINE.copy(),
            'processing': defaults.PROCESSING.copy()
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
        # Ensure all configured directories exist
        directories = [
            self.settings['paths']['data_dir'],
            self.settings['paths']['logs_dir'],
            self.settings.get('checkpointing', {}).get('checkpoint_dir', Path('checkpoint_outputs'))
        ]
        
        for path in directories:
            # Convert to Path object if it's a string (from YAML)
            if isinstance(path, str):
                path = Path(path)
            
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
    
    # New property accessors for enhanced configuration
    @property
    def progress_monitoring(self) -> Dict[str, Any]:
        return self.settings.get('progress_monitoring', {})
    
    @property
    def checkpointing(self) -> Dict[str, Any]:
        return self.settings.get('checkpointing', {})
    
    @property
    def timeouts(self) -> Dict[str, Any]:
        return self.settings.get('timeouts', {})
    
    @property
    def process_management(self) -> Dict[str, Any]:
        return self.settings.get('process_management', {})
    
    @property
    def memory_management(self) -> Dict[str, Any]:
        return self.settings.get('memory_management', {})
    
    # Existing property accessors
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
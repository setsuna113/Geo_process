import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from . import defaults

class Config:
    """Congiguration manager with YAML override support."""

    def __init__(self, config_file: Optional[Path] = None):
        self.settings = self.load_defaults()

        if config_file and config_file.exists():
            self._load_yaml_config(config_file)

        self._ensure_directories()
    
    def load_defaults(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            'database': defaults.DATABASE.copy(),
            'grids': defaults.GRIDS.copy(),
            'logging': defaults.LOGGING.copy(),
            'processing': defaults.PROCESSING.copy(),
            'features': defaults.FEATURES.copy(),
            'paths': {
                'project_root': defaults.PROJECT_ROOT,
                'data_dir': defaults.RAWDATA_DIR,
                'logs_dir': defaults.LOGS_DIR,
            }
        }
    def _load_yaml_config(self, config_file: Path):
        """Load and merge configuration from a YAML file."""
        with open(config_file, 'r') as file:
            yaml_config = yaml.safe_load(file)
            if yaml_config:
                self._deep_merge(self.settings, yaml_config)
    
    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base dictionary."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _ensure_directories(self):
        """Create necessary directories."""
        for path_key in ['data_dir', 'logs_dir']:
            path = self.settings['paths'][path_key]
            path.mkdir(parents=True, exist_ok=True)
    
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

# Global configuration instance
config = Config()
from .config import config, Config
from .biodiversity_config import get_biodiversity_config, get_method_config
from .method_registry import MethodConfigRegistry

# Import method config loaders
from .cpwopt.cpwopt_config_loader import CPWOPTConfigLoader
from .som.som_config_loader import SOMConfigLoader

# Register all available methods
MethodConfigRegistry.register('cpwopt', CPWOPTConfigLoader)
MethodConfigRegistry.register('som', SOMConfigLoader)

__all__ = ['config', 'Config', 'get_biodiversity_config', 'get_method_config', 'MethodConfigRegistry']
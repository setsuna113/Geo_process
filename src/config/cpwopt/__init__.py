"""CP-WOPT configuration module."""

from .cpwopt_config_loader import (
    get_cpwopt_config,
    register_cpwopt_config,
    CPWOPTConfigLoader
)

__all__ = [
    'get_cpwopt_config',
    'register_cpwopt_config',
    'CPWOPTConfigLoader'
]
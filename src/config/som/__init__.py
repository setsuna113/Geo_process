"""SOM Configuration Module

This module provides configuration management for Self-Organizing Map (SOM) analysis.
Use get_som_config() to access the singleton configuration instance.
"""

from .som_config import SOMConfig, get_som_config

__all__ = ['SOMConfig', 'get_som_config']
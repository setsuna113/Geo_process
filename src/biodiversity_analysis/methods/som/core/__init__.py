"""Core SOM implementation module."""

from .geo_som_core import GeoSOM, GeoSOMConfig

# Keep alias for backwards compatibility
GeoSOMVLRSOM = GeoSOM

__all__ = ['GeoSOM', 'GeoSOMVLRSOM', 'GeoSOMConfig']
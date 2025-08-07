"""Centralized band configuration management.

This module ensures consistent band ordering across the entire pipeline.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class BandConfig:
    """Configuration for band ordering and metadata."""
    
    biodiversity_bands: List[str]
    climate_bands: List[str] 
    elevation_band: str = "elevation"
    include_elevation: bool = True
    
    def get_ordered_bands(self) -> List[str]:
        """Get bands in the order they will appear in exports.
        
        This is the SINGLE SOURCE OF TRUTH for band ordering.
        """
        bands = []
        
        # Order is determined here and used everywhere
        bands.extend(self.biodiversity_bands)
        bands.extend(self.climate_bands)
        
        if self.include_elevation:
            bands.append(self.elevation_band)
            
        return bands
    
    def get_band_index(self, band_name: str) -> int:
        """Get the index (1-based) for a specific band."""
        bands = self.get_ordered_bands()
        try:
            return bands.index(band_name) + 1  # 1-based for GeoTIFF
        except ValueError:
            raise ValueError(f"Band '{band_name}' not found in configuration")
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for GEE and Parquet."""
        return {
            'band_names': self.get_ordered_bands(),
            'biodiversity_count': len(self.biodiversity_bands),
            'climate_count': len(self.climate_bands),
            'elevation_included': self.include_elevation,
            'total_bands': len(self.get_ordered_bands())
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BandConfig':
        """Create from GEE config."""
        biodiv_bands = []
        for dataset in config.get('biodiversity_datasets', []):
            biodiv_bands.append(dataset.get('band_name', dataset['name']))
        
        climate_bands = config.get('climate_variables', [
            'bio01', 'bio04', 'bio12', 'bio15'
        ])
        
        include_elevation = False
        if 'elevation' in config.get('additional_layers', {}):
            include_elevation = config['additional_layers']['elevation'].get('enabled', False)
        
        return cls(
            biodiversity_bands=biodiv_bands,
            climate_bands=climate_bands,
            include_elevation=include_elevation
        )


# Default production configuration
DEFAULT_BAND_CONFIG = BandConfig(
    biodiversity_bands=[
        'animal_richness',
        'plant_richness',
        'am_fungi_richness',
        'am_fungi_coeff_var',
        'ecm_fungi_richness',
        'ecm_fungi_coeff_var'
    ],
    climate_bands=[
        'bio01',  # Annual Mean Temperature
        'bio04',  # Temperature Seasonality
        'bio12',  # Annual Precipitation
        'bio15'   # Precipitation Seasonality
    ],
    include_elevation=True
)
"""
Statistical Analysis Data Paths Configuration

Centralized configuration for all biodiversity data file paths with fallback mechanisms.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetPaths:
    """Configuration for a single dataset with validation"""
    primary_path: str
    fallback_paths: List[str]
    description: str
    
    def get_valid_path(self) -> Optional[Path]:
        """Get the first valid path from primary and fallback options"""
        all_paths = [self.primary_path] + self.fallback_paths
        
        for path_str in all_paths:
            path = Path(path_str)
            if path.exists():
                logger.info(f"Using {self.description} from: {path}")
                return path
            else:
                logger.debug(f"Path not found: {path}")
        
        logger.error(f"No valid path found for {self.description}")
        logger.error(f"Tried paths: {all_paths}")
        return None


class StatisticalDataPaths:
    """Centralized paths configuration for statistical analyses"""
    
    def __init__(self):
        """Initialize with environment variable support"""
        
        # Base data directories - can be overridden via environment
        self.base_scratch_dir = os.getenv('BIODIV_SCRATCH_DIR', '/scratch/yl998')
        self.base_project_dir = os.getenv('BIODIV_PROJECT_DIR', '/home/yl998/dev/geo')
        
        # Climate data configuration
        self.climate_data = DatasetPaths(
            primary_path=os.getenv('BIODIV_CLIMATE_FILE', 
                f'{self.base_scratch_dir}/biodiversity_climate_production_1x_corrected.parquet'),
            fallback_paths=[
                f'{self.base_scratch_dir}/climate_data.parquet',
                f'{self.base_project_dir}/data/climate_data.parquet',
                f'{self.base_project_dir}/outputs/climate_data.parquet'
            ],
            description="Climate data (ocean-masked)"
        )
        
        # Richness datasets configuration
        self.datasets = {
            'plants': DatasetPaths(
                primary_path=os.getenv('BIODIV_PLANTS_FILE',
                    f'{self.base_scratch_dir}/downsampled_rasters/plants-richness_downsampled_0.16667.tif'),
                fallback_paths=[
                    f'{self.base_scratch_dir}/plants-richness.tif',
                    f'{self.base_project_dir}/data/plants-richness.tif',
                    '/maps/mwd24/richness/daru-plants-richness.tif'
                ],
                description="Plant species richness"
            ),
            'animals': DatasetPaths(
                primary_path=os.getenv('BIODIV_ANIMALS_FILE',
                    f'{self.base_scratch_dir}/downsampled_rasters/terrestrial-richness_downsampled_0.16667.tif'),
                fallback_paths=[
                    f'{self.base_scratch_dir}/terrestrial-richness.tif',
                    f'{self.base_project_dir}/data/terrestrial-richness.tif',
                    '/maps/mwd24/richness/iucn-terrestrial-richness.tif'
                ],
                description="Animal (terrestrial vertebrate) species richness"
            ),
            'am_fungi': DatasetPaths(
                primary_path=os.getenv('BIODIV_AM_FUNGI_FILE',
                    f'{self.base_scratch_dir}/downsampled_rasters/am-fungi-richness_downsampled_0.16667.tif'),
                fallback_paths=[
                    f'{self.base_scratch_dir}/am-fungi-richness.tif',
                    f'{self.base_project_dir}/data/am-fungi-richness.tif',
                    '/maps/mwd24/richness/am-fungi-richness.tif'
                ],
                description="AM fungi species richness"
            ),
            'ecm_fungi': DatasetPaths(
                primary_path=os.getenv('BIODIV_ECM_FUNGI_FILE',
                    f'{self.base_scratch_dir}/downsampled_rasters/ecm-fungi-richness_downsampled_0.16667.tif'),
                fallback_paths=[
                    f'{self.base_scratch_dir}/ecm-fungi-richness.tif',
                    f'{self.base_project_dir}/data/ecm-fungi-richness.tif',
                    '/maps/mwd24/richness/ecm-fungi-richness.tif'
                ],
                description="EcM fungi species richness"
            )
        }
        
        # Output directory configuration  
        self.output_base_dir = Path(os.getenv('BIODIV_OUTPUT_DIR', 
            f'{self.base_project_dir}/outputs/biodiversity_analysis'))
    
    def get_climate_data_path(self) -> Optional[Path]:
        """Get validated climate data path"""
        return self.climate_data.get_valid_path()
    
    def get_dataset_paths(self) -> Dict[str, Path]:
        """Get all validated dataset paths"""
        validated_paths = {}
        
        for name, dataset in self.datasets.items():
            path = dataset.get_valid_path()
            if path:
                validated_paths[name] = path
            else:
                logger.warning(f"Skipping {name} - no valid path found")
        
        return validated_paths
    
    def get_dataset_path(self, name: str) -> Optional[Path]:
        """Get specific dataset path"""
        if name not in self.datasets:
            logger.error(f"Unknown dataset: {name}")
            return None
        
        return self.datasets[name].get_valid_path()
    
    def validate_all_paths(self) -> Dict[str, bool]:
        """Validate all configured paths and return status"""
        validation_results = {}
        
        # Check climate data
        climate_path = self.get_climate_data_path()
        validation_results['climate_data'] = climate_path is not None
        
        # Check all datasets
        for name in self.datasets:
            dataset_path = self.get_dataset_path(name)
            validation_results[name] = dataset_path is not None
        
        # Check output directory (create if needed)
        try:
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            validation_results['output_dir'] = True
        except Exception as e:
            logger.error(f"Cannot create output directory {self.output_base_dir}: {e}")
            validation_results['output_dir'] = False
        
        return validation_results
    
    def get_summary(self) -> str:
        """Get human-readable summary of path configuration"""
        lines = ["Statistical Analysis Data Paths Configuration:", "=" * 50]
        
        # Climate data
        climate_path = self.get_climate_data_path()
        lines.append(f"Climate Data: {'✅' if climate_path else '❌'}")
        if climate_path:
            lines.append(f"  Path: {climate_path}")
        
        # Datasets
        lines.append("\nDatasets:")
        for name, dataset in self.datasets.items():
            path = dataset.get_valid_path()
            status = '✅' if path else '❌'
            lines.append(f"  {name:12} {status}")
            if path:
                lines.append(f"{'':15} {path}")
        
        # Output directory
        lines.append(f"\nOutput Directory: {'✅' if self.output_base_dir.exists() else '📁'}")
        lines.append(f"  Path: {self.output_base_dir}")
        
        # Environment variables
        lines.append("\nEnvironment Variables:")
        env_vars = [
            'BIODIV_SCRATCH_DIR', 'BIODIV_PROJECT_DIR', 'BIODIV_OUTPUT_DIR',
            'BIODIV_CLIMATE_FILE', 'BIODIV_PLANTS_FILE', 'BIODIV_ANIMALS_FILE',
            'BIODIV_AM_FUNGI_FILE', 'BIODIV_ECM_FUNGI_FILE'
        ]
        for var in env_vars:
            value = os.getenv(var, 'Not set')
            lines.append(f"  {var:<25}: {value}")
        
        return '\n'.join(lines)


# Global instance for easy import
statistical_data_paths = StatisticalDataPaths()
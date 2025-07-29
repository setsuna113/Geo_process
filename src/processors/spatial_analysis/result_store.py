# src/processors/spatial_analysis/result_store.py
"""File-based storage for spatial analysis results."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import xarray as xr
import geopandas as gpd
from datetime import datetime

from src.abstractions.interfaces.analyzer import AnalysisResult, AnalysisMetadata

logger = logging.getLogger(__name__)

class ResultStore:
    """Handle file-based storage and retrieval of analysis results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize result store.
        
        Args:
            output_dir: Base directory for storing results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, result: AnalysisResult, experiment_name: str) -> Path:
        """
        Save analysis results to files.
        
        Args:
            result: Analysis result to save
            experiment_name: Name for this experiment
            
        Returns:
            Path to saved results directory
        """
        # Create output subdirectory
        output_subdir = self.output_dir / experiment_name
        output_subdir.mkdir(exist_ok=True)
        
        # Save labels as numpy array
        np.save(output_subdir / 'labels.npy', result.labels)
        logger.info(f"Saved labels to {output_subdir / 'labels.npy'}")
        
        # Save metadata as JSON
        metadata_dict = {
            'analysis_type': result.metadata.analysis_type,
            'input_shape': result.metadata.input_shape,
            'input_bands': result.metadata.input_bands,
            'parameters': result.metadata.parameters,
            'processing_time': result.metadata.processing_time,
            'timestamp': result.metadata.timestamp,
            'data_source': result.metadata.data_source,
            'normalization_applied': result.metadata.normalization_applied,
            'coordinate_system': result.metadata.coordinate_system
        }
        
        with open(output_subdir / 'metadata.json', 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        logger.info(f"Saved metadata to {output_subdir / 'metadata.json'}")
        
        # Save statistics as JSON
        if result.statistics:
            with open(output_subdir / 'statistics.json', 'w') as f:
                json.dump(result.statistics, f, indent=2, default=str)
            logger.info(f"Saved statistics to {output_subdir / 'statistics.json'}")
        
        # Save spatial output if available
        if result.spatial_output is not None:
            if isinstance(result.spatial_output, (xr.Dataset, xr.DataArray)):
                result.spatial_output.to_netcdf(output_subdir / 'spatial_output.nc')
                logger.info(f"Saved spatial output to {output_subdir / 'spatial_output.nc'}")
            elif isinstance(result.spatial_output, gpd.GeoDataFrame):
                result.spatial_output.to_file(output_subdir / 'spatial_output.gpkg', driver='GPKG')
                logger.info(f"Saved spatial output to {output_subdir / 'spatial_output.gpkg'}")
        
        # Save additional outputs
        if result.additional_outputs:
            additional_dir = output_subdir / 'additional_outputs'
            additional_dir.mkdir(exist_ok=True)
            
            for key, value in result.additional_outputs.items():
                try:
                    if isinstance(value, np.ndarray):
                        np.save(additional_dir / f'{key}.npy', value)
                        logger.info(f"Saved {key} to {additional_dir / f'{key}.npy'}")
                    elif isinstance(value, (xr.Dataset, xr.DataArray)):
                        value.to_netcdf(additional_dir / f'{key}.nc')
                        logger.info(f"Saved {key} to {additional_dir / f'{key}.nc'}")
                    elif isinstance(value, gpd.GeoDataFrame):
                        value.to_file(additional_dir / f'{key}.gpkg', driver='GPKG')
                        logger.info(f"Saved {key} to {additional_dir / f'{key}.gpkg'}")
                    elif isinstance(value, (dict, list, str, int, float)):
                        with open(additional_dir / f'{key}.json', 'w') as f:
                            json.dump(value, f, indent=2, default=str)
                        logger.info(f"Saved {key} to {additional_dir / f'{key}.json'}")
                    else:
                        logger.warning(f"Could not save additional output '{key}' of type {type(value)}")
                except Exception as e:
                    logger.error(f"Error saving additional output '{key}': {e}")
        
        logger.info(f"Results saved to {output_subdir}")
        return output_subdir
    
    def load_results(self, experiment_name: str) -> Optional[AnalysisResult]:
        """
        Load previously saved results.
        
        Args:
            experiment_name: Name of experiment to load
            
        Returns:
            Loaded AnalysisResult or None if not found
        """
        result_dir = self.output_dir / experiment_name
        
        if not result_dir.exists():
            logger.warning(f"Results directory not found: {result_dir}")
            return None
        
        try:
            # Load labels
            labels_path = result_dir / 'labels.npy'
            if not labels_path.exists():
                logger.error(f"Labels file not found: {labels_path}")
                return None
            labels = np.load(labels_path)
            
            # Load metadata
            metadata_path = result_dir / 'metadata.json'
            if not metadata_path.exists():
                logger.error(f"Metadata file not found: {metadata_path}")
                return None
                
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = AnalysisMetadata(
                analysis_type=metadata_dict['analysis_type'],
                input_shape=tuple(metadata_dict['input_shape']),
                input_bands=metadata_dict['input_bands'],
                parameters=metadata_dict['parameters'],
                processing_time=metadata_dict['processing_time'],
                timestamp=metadata_dict['timestamp'],
                data_source=metadata_dict.get('data_source'),
                normalization_applied=metadata_dict.get('normalization_applied', False),
                coordinate_system=metadata_dict.get('coordinate_system', 'EPSG:4326')
            )
            
            # Load statistics
            statistics = {}
            statistics_path = result_dir / 'statistics.json'
            if statistics_path.exists():
                with open(statistics_path, 'r') as f:
                    statistics = json.load(f)
            
            # Load spatial output if exists
            spatial_output = None
            spatial_nc_path = result_dir / 'spatial_output.nc'
            spatial_gpkg_path = result_dir / 'spatial_output.gpkg'
            
            if spatial_nc_path.exists():
                spatial_output = xr.open_dataset(spatial_nc_path)
                # Convert to DataArray if single variable
                if len(spatial_output.data_vars) == 1:
                    spatial_output = spatial_output[list(spatial_output.data_vars)[0]]
            elif spatial_gpkg_path.exists():
                spatial_output = gpd.read_file(spatial_gpkg_path)
            
            # Load additional outputs
            additional_outputs = {}
            additional_dir = result_dir / 'additional_outputs'
            if additional_dir.exists():
                for file_path in additional_dir.iterdir():
                    key = file_path.stem
                    try:
                        if file_path.suffix == '.npy':
                            additional_outputs[key] = np.load(file_path)
                        elif file_path.suffix == '.nc':
                            additional_outputs[key] = xr.open_dataset(file_path)
                        elif file_path.suffix == '.gpkg':
                            additional_outputs[key] = gpd.read_file(file_path)
                        elif file_path.suffix == '.json':
                            with open(file_path, 'r') as f:
                                additional_outputs[key] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load additional output '{key}': {e}")
            
            result = AnalysisResult(
                labels=labels,
                metadata=metadata,
                statistics=statistics,
                spatial_output=spatial_output,
                additional_outputs=additional_outputs if additional_outputs else None
            )
            
            logger.info(f"Loaded analysis results from {result_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading results from {result_dir}: {e}")
            return None
    
    def list_experiments(self) -> list[str]:
        """List available experiment names."""
        if not self.output_dir.exists():
            return []
        
        experiments = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and (item / 'labels.npy').exists():
                experiments.append(item.name)
        
        return sorted(experiments)
    
    def delete_experiment(self, experiment_name: str) -> bool:
        """
        Delete an experiment's files.
        
        Args:
            experiment_name: Name of experiment to delete
            
        Returns:
            True if deleted successfully
        """
        result_dir = self.output_dir / experiment_name
        
        if not result_dir.exists():
            logger.warning(f"Experiment directory not found: {result_dir}")
            return False
        
        try:
            import shutil
            shutil.rmtree(result_dir)
            logger.info(f"Deleted experiment: {experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting experiment {experiment_name}: {e}")
            return False
# src/processors/data_preparation/data_normalizer.py
"""Normalize spatial data while preserving metadata."""

import logging
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Union as TypingUnion
import json

from src.config.config import Config
from src.base.processor import BaseProcessor
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class DataNormalizer(BaseProcessor):
    """Normalize spatial data while preserving structure and metadata."""
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        super().__init__(batch_size=1000, config=config)
        self.db = db_connection
        
        # Normalization parameters
        self.chunk_size = config.get('data_preparation', {}).get('chunk_size', 1000)
        self.scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
    
    def normalize(self,
                 data: Union[xr.Dataset, xr.DataArray],
                 method: str = 'standard',
                 feature_range: Tuple[float, float] = (0, 1),
                 by_band: bool = True,
                 save_params: bool = True) -> Dict[str, Any]:
        """
        Normalize spatial data using specified method.
        
        Args:
            data: Input xarray Dataset or DataArray
            method: Normalization method ('standard', 'minmax', 'robust')
            feature_range: Range for minmax scaling
            by_band: Whether to normalize each band separately
            save_params: Whether to save normalization parameters to database
            
        Returns:
            Dict with normalized data and scaler parameters
        """
        logger.info(f"Normalizing data using {method} method")
        
        if method not in self.scalers:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Initialize scaler
        scaler_class = self.scalers[method]
        if method == 'minmax':
            scaler = scaler_class(feature_range=feature_range)
        else:
            scaler = scaler_class()
        
        # Normalize based on data type
        normalized: Union[xr.DataArray, xr.Dataset]
        if isinstance(data, xr.DataArray):
            normalized, params = self._normalize_dataarray(data, scaler, by_band)
        else:
            normalized, params = self._normalize_dataset(data, scaler, by_band)
        # normalized is now Union[xr.DataArray, xr.Dataset]
        
        # Save parameters if requested
        if save_params:
            param_id = self._save_normalization_params(params, method)
        else:
            param_id = None
        
        return {
            'data': normalized,
            'parameters': params,
            'method': method,
            'parameter_id': param_id
        }
    
    def denormalize(self,
                   data: Union[xr.Dataset, xr.DataArray],
                   parameters: Optional[Dict[str, Any]] = None,
                   parameter_id: Optional[int] = None) -> Union[xr.DataArray, xr.Dataset]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data
            parameters: Normalization parameters (if not provided, uses parameter_id)
            parameter_id: ID of saved parameters in database
            
        Returns:
            Denormalized data
        """
        if parameters is None and parameter_id is None:
            raise ValueError("Either parameters or parameter_id must be provided")
        
        if parameters is None:
            if parameter_id is not None:
                parameters = self._load_normalization_params(parameter_id)
            else:
                raise ValueError("parameter_id cannot be None")
        
        method = parameters['method']
        scaler_class = self.scalers[method]
        
        if isinstance(data, xr.DataArray):
            return self._denormalize_dataarray(data, parameters, scaler_class)
        else:
            return self._denormalize_dataset(data, parameters, scaler_class)
    
    def _normalize_dataarray(self, 
                           da: xr.DataArray,
                           scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
                           by_band: bool) -> Tuple[xr.DataArray, Dict]:
        """Normalize a DataArray."""
        # Get dimensions
        spatial_dims = [d for d in da.dims if d in ['lat', 'lon', 'x', 'y']]
        
        if len(spatial_dims) != 2:
            raise ValueError("DataArray must have 2 spatial dimensions")
        
        # Flatten spatial dimensions
        original_shape = da.shape
        flat_data = np.asarray(da.stack(pixel=spatial_dims).values)
        
        # Remove NaN values for fitting
        valid_mask = ~np.isnan(flat_data)
        valid_data = flat_data[valid_mask]
        
        # Fit and transform
        scaler.fit(valid_data.reshape(-1, 1))
        normalized_flat = np.full_like(flat_data, np.nan)
        transformed = scaler.transform(valid_data.reshape(-1, 1))
        normalized_flat[valid_mask] = np.asarray(transformed).flatten()
        
        # Reshape back
        normalized = normalized_flat.reshape(original_shape)
        
        # Create output DataArray
        normalized_da = xr.DataArray(
            normalized,
            coords=da.coords,
            dims=da.dims,
            attrs=da.attrs.copy()
        )
        
        normalized_da.attrs.update({
            'normalized': True,
            'normalization_method': scaler.__class__.__name__
        })
        
        # Extract parameters
        params = {
            "method": scaler.__class__.__name__.lower().replace("scaler", ""),
            "shape": original_shape,
            "dims": list(da.dims),
        }
        
        # Extract scaler-specific parameters based on scaler type
        def safe_extract_scalar(value: Any) -> float:
            """Safely extract a scalar value from scaler attributes."""
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if hasattr(value, '__len__') and len(value) > 0:
                return float(value[0])
            if hasattr(value, 'item'):  # numpy scalar
                return float(value.item())
            return float(value)
        
        if isinstance(scaler, StandardScaler):
            if hasattr(scaler, "mean_") and scaler.mean_ is not None:
                params["mean"] = safe_extract_scalar(scaler.mean_)
            if hasattr(scaler, "scale_") and scaler.scale_ is not None:
                params["scale"] = safe_extract_scalar(scaler.scale_)
        elif isinstance(scaler, MinMaxScaler):
            if hasattr(scaler, "min_") and scaler.min_ is not None:
                params["min"] = safe_extract_scalar(scaler.min_)
            if hasattr(scaler, "scale_") and scaler.scale_ is not None:
                params["scale"] = safe_extract_scalar(scaler.scale_)
            if hasattr(scaler, "data_max_") and scaler.data_max_ is not None:
                params["data_max"] = safe_extract_scalar(scaler.data_max_)
        elif isinstance(scaler, RobustScaler):
            if hasattr(scaler, "center_") and scaler.center_ is not None:
                params["center"] = safe_extract_scalar(scaler.center_)
            if hasattr(scaler, "scale_") and scaler.scale_ is not None:
                params["scale"] = safe_extract_scalar(scaler.scale_)
        
        return normalized_da, params
    
    def _normalize_dataset(self,
                         ds: xr.Dataset,
                         scaler: Union[StandardScaler, MinMaxScaler, RobustScaler],
                         by_band: bool) -> Tuple[xr.Dataset, Dict]:
        """Normalize a Dataset."""
        normalized_vars = {}
        all_params = {}
        
        for var_name in ds.data_vars:
            logger.info(f"Normalizing variable: {var_name}")
            
            da = ds[var_name]
            normalized_da, params = self._normalize_dataarray(da, scaler, by_band)
            
            normalized_vars[var_name] = normalized_da
            all_params[var_name] = params
        
        # Create normalized dataset
        normalized_ds = xr.Dataset(normalized_vars, attrs=ds.attrs.copy())
        normalized_ds.attrs['normalized'] = True
        
        return normalized_ds, all_params
    
    def _denormalize_dataarray(self,
                              da: xr.DataArray,
                              params: Dict,
                              scaler_class) -> xr.DataArray:
        """Denormalize a DataArray."""
        # Recreate scaler with saved parameters
        scaler = scaler_class()
        
        # Flatten, inverse transform, reshape
        spatial_dims = [d for d in da.dims if d in ["lat", "lon", "x", "y"]]
        flat_data = np.asarray(da.stack(pixel=spatial_dims).values)
        
        valid_mask = ~np.isnan(flat_data)
        denorm_flat = np.full_like(flat_data, np.nan)
        
        if np.any(valid_mask):
            # Manual inverse transformation based on method
            if "mean" in params and params["mean"] is not None:
                # StandardScaler: x = (z * scale) + mean
                denorm_flat[valid_mask] = (flat_data[valid_mask] * params["scale"]) + params["mean"]
            elif "min" in params and params["min"] is not None:
                # MinMaxScaler: x = z / scale + min
                denorm_flat[valid_mask] = (flat_data[valid_mask] / params["scale"]) + params["min"]
            else:
                # Fallback to scaler inverse_transform (may fail if scaler not properly fitted)
                denorm_flat[valid_mask] = scaler.inverse_transform(
                    flat_data[valid_mask].reshape(-1, 1)
                ).flatten()
        
        # Reshape and create output
        denormalized = denorm_flat.reshape(da.shape)
        
        denorm_da = xr.DataArray(
            denormalized,
            coords=da.coords,
            dims=da.dims,
            attrs={k: v for k, v in da.attrs.items() if k != "normalized"}
        )
        
        return denorm_da
    
    def _denormalize_dataset(self,
                           ds: xr.Dataset,
                           params: Dict,
                           scaler_class) -> xr.Dataset:
        """Denormalize a Dataset."""
        denormalized_vars = {}
        
        for var_name in ds.data_vars:
            da = ds[var_name]
            var_params = params.get(var_name, params)  # Use variable-specific params if available
            denorm_da = self._denormalize_dataarray(da, var_params, scaler_class)
            denormalized_vars[var_name] = denorm_da
        
        # Create denormalized dataset
        denorm_ds = xr.Dataset(denormalized_vars, attrs=ds.attrs.copy())
        if 'normalized' in denorm_ds.attrs:
            del denorm_ds.attrs['normalized']
        
        return denorm_ds
    
    def _save_normalization_params(self, params: Dict, method: str) -> int:
        """Save normalization parameters to database."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # Create normalization parameters record
            cur.execute("""
                INSERT INTO normalization_parameters (method, parameters)
                VALUES (%s, %s)
                RETURNING id
            """, (
                method,
                json.dumps(params)
            ))
            
            param_id = cur.fetchone()[0]
            conn.commit()
            
        logger.info(f"Saved normalization parameters with ID: {param_id}")
        return param_id
    
    def _load_normalization_params(self, param_id: int) -> Dict:
        """Load normalization parameters from database."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT method, parameters 
                FROM normalization_parameters 
                WHERE id = %s
            """, (param_id,))
            
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"No parameters found with ID: {param_id}")
            
            method, params = row
            params['method'] = method
            
        return params
    
    def process_single(self, item: Any) -> Any:
        """Process a single item - implementation depends on specific use case."""
        return item
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        return True, None

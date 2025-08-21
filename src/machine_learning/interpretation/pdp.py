"""Partial Dependence Plot implementation for model interpretation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Optional, List, Union, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PartialDependencePlot:
    """
    Calculate and visualize partial dependence plots for understanding
    feature effects and interactions.
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize PDP calculator.
        
        Args:
            model: Fitted model with predict method
            feature_names: Optional list of feature names
        """
        self.model = model
        self.feature_names = feature_names


class PartialDependence:
    """Static methods for partial dependence computation."""
    
    @staticmethod
    def compute_pdp_interaction(model, features: List[str], data, grid_resolution: int = 20, sample_size: Optional[int] = 10000):
        """Compute partial dependence for feature interaction - OPTIMIZED.
        
        For H2: Shows how the combined effect of bio4 and plants affects fungi.
        
        Key optimizations:
        1. Sample data if too large
        2. Use means for other covariates instead of copying full dataset
        3. Batch all predictions in one call
        
        Args:
            model: Fitted model
            features: List of two features [var1, var2]
            data: Data frame (H2O or pandas)
            grid_resolution: Number of grid points per feature
            sample_size: Max rows to use for computing means (None = use all)
            
        Returns:
            Dictionary with PDP results
        """
        import h2o
        
        logger.info(f"Computing optimized PDP for {features} interaction")
        
        # Access the actual model
        if hasattr(model, 'model'):
            h2o_model = model.model
        else:
            h2o_model = model
        
        # OPTIMIZATION 1: Sample if dataset is too large
        if isinstance(data, h2o.H2OFrame):
            n_rows = data.nrows
            if sample_size and n_rows > sample_size:
                logger.info(f"Sampling {sample_size:,} from {n_rows:,} rows for PDP")
                sample_frac = sample_size / n_rows
                data_sample = data.split_frame(ratios=[sample_frac], seed=42)[0]
            else:
                data_sample = data
            
            # Get means for other variables (in H2O, no conversion!)
            logger.info("Computing covariate means in H2O...")
            means_list = data_sample.mean()
            col_names = data_sample.columns
            means_dict = {col: float(means_list[i]) for i, col in enumerate(col_names)}
            
            # Get ranges for grid variables
            var1, var2 = features
            var1_stats = data_sample[var1].quantile([0.05, 0.95])
            var2_stats = data_sample[var2].quantile([0.05, 0.95])
            var1_min, var1_max = float(var1_stats[0,0]), float(var1_stats[1,0])
            var2_min, var2_max = float(var2_stats[0,0]), float(var2_stats[1,0])
        else:
            # Pandas path
            if sample_size and len(data) > sample_size:
                data_sample = data.sample(n=sample_size, random_state=42)
            else:
                data_sample = data
                
            means_dict = data_sample.mean().to_dict()
            var1, var2 = features
            var1_min, var1_max = data_sample[var1].quantile([0.05, 0.95])
            var2_min, var2_max = data_sample[var2].quantile([0.05, 0.95])
        
        # Create grid
        var1_grid = np.linspace(var1_min, var1_max, grid_resolution)
        var2_grid = np.linspace(var2_min, var2_max, grid_resolution)
        
        # OPTIMIZATION 2: Create all prediction points at once (just means + grid values)
        logger.info(f"Creating {grid_resolution**2} prediction points...")
        pred_data = []
        for v1 in var1_grid:
            for v2 in var2_grid:
                row = means_dict.copy()  # Just copy the small means dict!
                row[var1] = v1
                row[var2] = v2
                pred_data.append(row)
        
        # OPTIMIZATION 3: Batch predict all at once
        logger.info("Batch predicting all grid points...")
        pred_df = pd.DataFrame(pred_data)
        
        if isinstance(data, h2o.H2OFrame):
            pred_h2o = h2o.H2OFrame(pred_df)
            preds = h2o_model.predict(pred_h2o)
            # Get predictions as dataframe and extract the 'predict' column
            preds_df = preds.as_data_frame(use_multi_thread=True)
            # H2O GLM returns 'predict' column for regression
            if 'predict' in preds_df.columns:
                pred_values = preds_df['predict'].values
            else:
                # Fallback to first column
                pred_values = preds_df.iloc[:, 0].values
        else:
            pred_values = h2o_model.predict(pred_df)
        
        # Ensure we have the right number of predictions
        expected_size = grid_resolution * grid_resolution
        if len(pred_values) != expected_size:
            logger.warning(f"Got {len(pred_values)} predictions, expected {expected_size}")
        
        # Reshape to matrix
        pd_matrix = pred_values.reshape(grid_resolution, grid_resolution)
        
        logger.info(f"PDP complete. Range: [{pd_matrix.min():.3f}, {pd_matrix.max():.3f}]")
        
        return {
            'features': features,
            'var1_grid': var1_grid,
            'var2_grid': var2_grid,
            'pd_matrix': pd_matrix,
            'sample_size_used': data_sample.nrows if isinstance(data_sample, h2o.H2OFrame) else len(data_sample),
            'interpretation': PartialDependence._interpret_pdp(pd_matrix, var1, var2)
        }
    
    @staticmethod
    def _interpret_pdp(pd_matrix, var1, var2):
        """Interpret PDP results for H2 hypothesis."""
        # Check if effect changes sign (supports interaction)
        min_val = pd_matrix.min()
        max_val = pd_matrix.max()
        range_val = max_val - min_val
        
        # Check corners for H2 pattern
        # Low seasonality, high plants (should be high if H2 true)
        corner_low_season_high_plant = pd_matrix[0, -1]
        # High seasonality, high plants (should be low if H2 true)
        corner_high_season_high_plant = pd_matrix[-1, -1]
        
        h2_pattern = corner_high_season_high_plant < corner_low_season_high_plant
        
        return {
            'range': range_val,
            'shows_interaction': range_val > 0.1 * abs(max_val),  # Substantial variation
            'h2_pattern': h2_pattern,
            'interpretation': f"PDP shows {'strong' if range_val > 0.2 * abs(max_val) else 'weak'} interaction between {var1} and {var2}"
        }
        
    def calculate_1d_pdp(self, 
                        X: pd.DataFrame,
                        feature: Union[str, int],
                        grid_resolution: int = 50,
                        percentiles: Tuple[float, float] = (0.05, 0.95)) -> dict:
        """
        Calculate 1D partial dependence for a single feature.
        
        Args:
            X: Feature matrix
            feature: Feature name or index
            grid_resolution: Number of points in the grid
            percentiles: Range of feature values to consider
            
        Returns:
            Dict with grid values and partial dependence
        """
        # Get feature column
        if isinstance(feature, str):
            if feature not in X.columns:
                raise ValueError(f"Feature '{feature}' not found in data")
            feature_idx = X.columns.get_loc(feature)
        else:
            feature_idx = feature
            feature = X.columns[feature_idx] if hasattr(X, 'columns') else f'feature_{feature_idx}'
        
        # Create grid
        feature_values = X.iloc[:, feature_idx]
        min_val = np.percentile(feature_values, percentiles[0] * 100)
        max_val = np.percentile(feature_values, percentiles[1] * 100)
        grid = np.linspace(min_val, max_val, grid_resolution)
        
        # Calculate partial dependence
        pd_values = []
        
        for val in tqdm(grid, desc=f"Calculating PDP for {feature}"):
            X_temp = X.copy()
            X_temp.iloc[:, feature_idx] = val
            predictions = self.model.predict(X_temp)
            pd_values.append(np.mean(predictions))
        
        return {
            'feature': feature,
            'grid': grid,
            'pd_values': np.array(pd_values),
            'feature_idx': feature_idx
        }
    
    def calculate_2d_pdp(self,
                        X: pd.DataFrame,
                        features: Tuple[Union[str, int], Union[str, int]],
                        grid_resolution: Union[int, Tuple[int, int]] = 20,
                        percentiles: Tuple[float, float] = (0.05, 0.95)) -> dict:
        """
        Calculate 2D partial dependence for feature interaction.
        
        Args:
            X: Feature matrix
            features: Tuple of two feature names or indices
            grid_resolution: Number of points in each dimension
            percentiles: Range of feature values to consider
            
        Returns:
            Dict with grid values and 2D partial dependence
        """
        # Handle grid resolution
        if isinstance(grid_resolution, int):
            grid_res = (grid_resolution, grid_resolution)
        else:
            grid_res = grid_resolution
        
        # Get feature columns
        feature_info = []
        for feat in features:
            if isinstance(feat, str):
                if feat not in X.columns:
                    raise ValueError(f"Feature '{feat}' not found in data")
                idx = X.columns.get_loc(feat)
                name = feat
            else:
                idx = feat
                name = X.columns[idx] if hasattr(X, 'columns') else f'feature_{idx}'
            feature_info.append((name, idx))
        
        # Create grids
        grids = []
        for i, (name, idx) in enumerate(feature_info):
            feature_values = X.iloc[:, idx]
            min_val = np.percentile(feature_values, percentiles[0] * 100)
            max_val = np.percentile(feature_values, percentiles[1] * 100)
            grid = np.linspace(min_val, max_val, grid_res[i])
            grids.append(grid)
        
        # Create meshgrid
        grid1, grid2 = np.meshgrid(grids[0], grids[1])
        
        # Calculate partial dependence
        pd_values = np.zeros_like(grid1)
        
        total_points = grid_res[0] * grid_res[1]
        with tqdm(total=total_points, desc=f"Calculating 2D PDP for {feature_info[0][0]} x {feature_info[1][0]}") as pbar:
            for i in range(grid_res[0]):
                for j in range(grid_res[1]):
                    X_temp = X.copy()
                    X_temp.iloc[:, feature_info[0][1]] = grid1[j, i]
                    X_temp.iloc[:, feature_info[1][1]] = grid2[j, i]
                    predictions = self.model.predict(X_temp)
                    pd_values[j, i] = np.mean(predictions)
                    pbar.update(1)
        
        return {
            'features': [info[0] for info in feature_info],
            'feature_indices': [info[1] for info in feature_info],
            'grid1': grid1,
            'grid2': grid2,
            'pd_values': pd_values,
            'grid1_1d': grids[0],
            'grid2_1d': grids[1]
        }
    
    def plot_1d_pdp(self, pdp_result: dict, figsize: tuple = (8, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """Plot 1D partial dependence."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(pdp_result['grid'], pdp_result['pd_values'], linewidth=2)
        ax.set_xlabel(pdp_result['feature'])
        ax.set_ylabel('Partial Dependence')
        ax.set_title(f'Partial Dependence Plot for {pdp_result["feature"]}')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_2d_pdp(self, 
                    pdp_result: dict,
                    figsize: tuple = (10, 8),
                    cmap: str = 'RdBu_r',
                    show_contours: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 2D partial dependence with interaction visualization.
        
        This is the key plot for testing the temperate mismatch hypothesis!
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the main heatmap
        im = ax.contourf(pdp_result['grid1'], 
                         pdp_result['grid2'], 
                         pdp_result['pd_values'],
                         levels=20,
                         cmap=cmap)
        
        # Add contour lines if requested
        if show_contours:
            contours = ax.contour(pdp_result['grid1'], 
                                 pdp_result['grid2'], 
                                 pdp_result['pd_values'],
                                 levels=10,
                                 colors='black',
                                 alpha=0.4,
                                 linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Labels and title
        ax.set_xlabel(pdp_result['features'][0])
        ax.set_ylabel(pdp_result['features'][1])
        ax.set_title(f'2D Partial Dependence: {pdp_result["features"][0]} × {pdp_result["features"][1]}')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Partial Dependence', rotation=270, labelpad=20)
        
        return fig, ax
    
    def plot_interaction_slices(self,
                               pdp_result: dict,
                               n_slices: int = 5,
                               figsize: tuple = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot interaction as slices at different values of the second feature.
        
        This helps visualize how the effect of feature 1 changes at different
        levels of feature 2 - perfect for the temperate mismatch hypothesis!
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get slice indices
        n_points = pdp_result['grid2_1d'].shape[0]
        slice_indices = np.linspace(0, n_points-1, n_slices, dtype=int)
        
        # Create color map for slices
        colors = cm.viridis(np.linspace(0, 1, n_slices))
        
        # Plot slices along feature 2
        for i, idx in enumerate(slice_indices):
            slice_val = pdp_result['grid2_1d'][idx]
            pd_slice = pdp_result['pd_values'][idx, :]
            
            label = f"{pdp_result['features'][1]} = {slice_val:.2f}"
            ax1.plot(pdp_result['grid1_1d'], pd_slice, 
                    color=colors[i], linewidth=2, label=label)
        
        ax1.set_xlabel(pdp_result['features'][0])
        ax1.set_ylabel('Partial Dependence')
        ax1.set_title(f'Effect of {pdp_result["features"][0]} at different {pdp_result["features"][1]} levels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot heatmap on second axis
        im = ax2.contourf(pdp_result['grid1'], 
                         pdp_result['grid2'], 
                         pdp_result['pd_values'],
                         levels=20,
                         cmap='RdBu_r')
        
        # Add slice lines
        for idx in slice_indices:
            ax2.axhline(y=pdp_result['grid2_1d'][idx], 
                       color='black', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel(pdp_result['features'][0])
        ax2.set_ylabel(pdp_result['features'][1])
        ax2.set_title('2D Partial Dependence with Slice Locations')
        
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        
        return fig, (ax1, ax2)


class InteractionStrengthAnalyzer:
    """Analyze the strength and nature of feature interactions."""
    
    @staticmethod
    def calculate_h_statistic(pdp_2d: dict, pdp_1d_feat1: dict, pdp_1d_feat2: dict) -> float:
        """
        Calculate Friedman's H-statistic for interaction strength.
        
        H = 0 means no interaction, H = 1 means complete interaction.
        """
        # Get the 2D PDP values
        pd_joint = pdp_2d['pd_values']
        
        # Create the additive (no interaction) prediction
        pd_1 = pdp_1d_feat1['pd_values']
        pd_2 = pdp_1d_feat2['pd_values']
        
        # Broadcast to create additive grid
        pd_additive = pd_1[np.newaxis, :] + pd_2[:, np.newaxis]
        
        # Center both
        pd_joint_centered = pd_joint - np.mean(pd_joint)
        pd_additive_centered = pd_additive - np.mean(pd_additive)
        
        # Calculate H-statistic
        numerator = np.sum((pd_joint_centered - pd_additive_centered) ** 2)
        denominator = np.sum(pd_joint_centered ** 2)
        
        h_stat = numerator / denominator if denominator > 0 else 0
        
        return h_stat
"""
Optimized interaction plots for large datasets.

Uses efficient computation strategies:
1. Computes means in H2O (no conversion)
2. Only converts small prediction grid
3. Optional: Uses datashader for millions of points
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import h2o

logger = logging.getLogger(__name__)


class OptimizedInteractionPlots:
    """Fast interaction plots for large datasets."""
    
    @staticmethod
    def plot_continuous_interaction_fast(
        model: Any,
        var1: str,
        var2: str,
        var1_range: Tuple[float, float],
        var2_values: List[float],
        data: h2o.H2OFrame,
        ylabel: str = 'Predicted Outcome',
        xlabel: str = None,
        title: str = None,
        n_points: int = 100
    ) -> plt.Figure:
        """
        Create interaction plot WITHOUT converting full dataset.
        
        Key optimization: Compute means in H2O, only convert prediction grid.
        
        Args:
            model: Fitted H2O GLM model
            var1: Continuous variable for x-axis
            var2: Moderator variable
            var1_range: Range for var1 (min, max)
            var2_values: Specific values of var2 to plot
            data: H2O frame (stays in H2O!)
            ylabel: Y-axis label
            xlabel: X-axis label
            title: Plot title
            n_points: Number of points in prediction grid
            
        Returns:
            matplotlib figure
        """
        logger.info(f"Creating optimized interaction plot for {var1} × {var2}")
        
        # Access the actual H2O model
        if hasattr(model, 'model'):
            h2o_model = model.model
        else:
            h2o_model = model
        
        # OPTIMIZATION 1: Compute means in H2O (no conversion!)
        logger.info("Computing covariate means in H2O...")
        means_list = data.mean()  # Returns a list of means
        
        # Create dictionary mapping column names to their means
        means_dict = {col: float(means_list[i]) for i, col in enumerate(data.columns)}
        logger.info(f"Means computed for {len(means_dict)} variables")
        
        # Create prediction grid
        var1_grid = np.linspace(var1_range[0], var1_range[1], n_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(var2_values)))
        
        # OPTIMIZATION 2: Batch predictions
        for i, v2_val in enumerate(var2_values):
            # Create all prediction rows at once
            pred_data = []
            for v1_val in var1_grid:
                row = means_dict.copy()
                row[var1] = v1_val
                row[var2] = v2_val
                pred_data.append(row)
            
            # Convert to H2O frame once for all points
            pred_df = pd.DataFrame(pred_data)
            pred_h2o = h2o.H2OFrame(pred_df)
            
            # Batch predict
            predictions = h2o_model.predict(pred_h2o)
            
            # Convert predictions (small data!)
            preds_df = predictions.as_data_frame(use_multi_thread=True)
            # H2O GLM returns 'predict' column for regression
            if 'predict' in preds_df.columns:
                pred_values = preds_df['predict'].values
            else:
                # Fallback to first column
                pred_values = preds_df.iloc[:, 0].values
            
            # Plot line
            label = f"{var2} = {v2_val:.1f}"
            if v2_val == -1:
                label += " (Low)"
            elif v2_val == 0:
                label += " (Mean)"
            elif v2_val == 1:
                label += " (High)"
            
            ax.plot(var1_grid, pred_values, color=colors[i], 
                   linewidth=2.5, label=label, alpha=0.8)
        
        # Formatting
        ax.set_xlabel(xlabel or var1, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title or f'Interaction: {var1} × {var2}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Reference line at y=0
        if min(ax.get_ylim()) < 0 < max(ax.get_ylim()):
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        logger.info("Interaction plot completed")
        return fig
    
    @staticmethod
    def plot_with_datashader(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        plot_width: int = 800,
        plot_height: int = 600
    ) -> Any:
        """
        Use datashader for millions of points.
        
        Datashader can handle 100M+ points efficiently by rasterizing
        to a fixed-size image instead of rendering individual points.
        
        Args:
            data: DataFrame with data (can be huge!)
            x_col: X column name
            y_col: Y column name
            color_col: Optional column for coloring
            plot_width: Width in pixels
            plot_height: Height in pixels
            
        Returns:
            Datashader image or hvplot object
        """
        try:
            import datashader as ds
            import datashader.transfer_functions as tf
            from datashader.utils import export_image
            
            logger.info(f"Using datashader for {len(data):,} points")
            
            # Create canvas
            canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
            
            # Aggregate
            if color_col:
                agg = canvas.points(data, x_col, y_col, ds.count_cat(color_col))
                img = tf.shade(agg, color_key=['blue', 'green', 'red'])
            else:
                agg = canvas.points(data, x_col, y_col)
                img = tf.shade(agg, cmap=['lightblue', 'darkblue'])
            
            return img
            
        except ImportError:
            logger.warning("Datashader not installed, falling back to sampling")
            # Fallback: sample if datashader not available
            sample_size = min(10000, len(data))
            sample = data.sample(n=sample_size)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if color_col:
                scatter = ax.scatter(sample[x_col], sample[y_col], 
                                   c=sample[color_col], alpha=0.5, s=1)
                plt.colorbar(scatter)
            else:
                ax.scatter(sample[x_col], sample[y_col], alpha=0.5, s=1)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Sample of {sample_size:,} from {len(data):,} points")
            
            return fig
    
    @staticmethod
    def plot_partial_dependence_fast(
        model: Any,
        var_name: str,
        data: h2o.H2OFrame,
        n_points: int = 50,
        sample_size: Optional[int] = 10000
    ) -> plt.Figure:
        """
        Fast partial dependence plot using sampling.
        
        Instead of using all 43M rows, samples intelligently.
        
        Args:
            model: H2O model
            var_name: Variable to plot
            data: H2O frame
            n_points: Number of points in grid
            sample_size: How many rows to sample (None = use all)
            
        Returns:
            matplotlib figure
        """
        logger.info(f"Creating fast PDP for {var_name}")
        
        # Sample if needed
        if sample_size and data.nrows > sample_size:
            logger.info(f"Sampling {sample_size:,} from {data.nrows:,} rows")
            sample_frac = sample_size / data.nrows
            data_sample = data.split_frame(ratios=[sample_frac], seed=42)[0]
        else:
            data_sample = data
        
        # Use H2O's built-in PDP (it's optimized!)
        pdp = model.partial_plot(
            data=data_sample,
            cols=[var_name],
            plot=False  # Get data, not plot
        )
        
        # Create our own optimized plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract and plot (pdp is already computed efficiently)
        if pdp and len(pdp) > 0:
            pdp_data = pdp[0]
            ax.plot(pdp_data[0], pdp_data[1], linewidth=2, color='blue')
            ax.set_xlabel(var_name, fontsize=12)
            ax.set_ylabel('Partial Dependence', fontsize=12)
            ax.set_title(f'Partial Dependence Plot: {var_name}', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
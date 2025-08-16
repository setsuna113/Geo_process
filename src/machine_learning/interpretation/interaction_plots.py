"""Interaction plots for visualizing how effects change by context.

Particularly useful for testing H2 hypothesis (temperate × plant_high interaction).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class InteractionPlots:
    """Create and analyze interaction plots for GLM models.
    
    Visualizes how the effect of one variable changes across levels of another.
    Essential for understanding the temperate mismatch hypothesis (H2).
    """
    
    def __init__(self, model: Any):
        """Initialize with fitted GLM model.
        
        Args:
            model: Fitted GLM model (H2O or similar)
        """
        self.model = model
    
    @staticmethod
    def plot_continuous_interaction(model: Any,
                                   var1: str,
                                   var2: str,
                                   var1_range: Tuple[float, float],
                                   var2_values: List[float],
                                   data: Any,
                                   ylabel: str = 'Predicted Outcome',
                                   xlabel: str = None,
                                   title: str = None) -> plt.Figure:
        """Create interaction plot for continuous variables.
        
        For H2: Shows how plant-fungi relationship changes at different bio4 levels.
        
        Args:
            model: Fitted model with interaction
            var1: Continuous variable for x-axis (e.g., 'plants_richness_log')
            var2: Moderator variable (e.g., 'bio04_std')
            var1_range: Range for var1 (min, max)
            var2_values: Specific values of var2 to plot
            data: Data frame for baseline values
            ylabel: Y-axis label
            xlabel: X-axis label (default: var1 name)
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import h2o
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Access the actual model
        if hasattr(model, 'model'):
            h2o_model = model.model
        else:
            h2o_model = model
        
        # Create prediction grid
        var1_grid = np.linspace(var1_range[0], var1_range[1], 100)
        
        # Get baseline data
        if hasattr(data, 'as_data_frame'):
            base_df = data.as_data_frame(use_multi_thread=True)
        else:
            base_df = data.copy()
        
        # Plot line for each var2 value
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(var2_values)))
        
        for i, v2_val in enumerate(var2_values):
            predictions = []
            
            # Create prediction data for this var2 value
            for v1_val in var1_grid:
                # Use mean values for other variables
                pred_df = pd.DataFrame([base_df.mean()])
                pred_df[var1] = v1_val
                pred_df[var2] = v2_val
                
                # Convert to H2O frame if needed
                if isinstance(data, h2o.H2OFrame):
                    pred_h2o = h2o.H2OFrame(pred_df)
                    pred = h2o_model.predict(pred_h2o)
                    pred_val = float(pred.as_data_frame(use_multi_thread=True).values[0][0])
                else:
                    pred = h2o_model.predict(pred_df)
                    pred_val = float(pred)
                
                predictions.append(pred_val)
            
            # Plot line
            label = f"{var2} = {v2_val:.1f}"
            if v2_val == -1:
                label += " (Low seasonality)"
            elif v2_val == 0:
                label += " (Mean seasonality)"
            elif v2_val == 1:
                label += " (High seasonality)"
                
            ax.plot(var1_grid, predictions, color=colors[i], linewidth=2.5, label=label)
        
        # Formatting
        ax.set_xlabel(xlabel if xlabel else var1, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title if title else f'Interaction: {var1} × {var2}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add reference line at y=0 if range includes it
        if min(ax.get_ylim()) < 0 < max(ax.get_ylim()):
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
        
    def plot_interaction(self,
                        data: Union[pd.DataFrame, Any],
                        var1: str,
                        var2: str,
                        var1_levels: Optional[List[float]] = None,
                        var2_levels: Optional[List[float]] = None,
                        plot_type: str = 'both',
                        figsize: Tuple[int, int] = (12, 5)) -> Dict[str, Any]:
        """Create interaction plot showing how var1 effect changes across var2 levels.
        
        Args:
            data: Data for predictions
            var1: First variable (x-axis in plot)
            var2: Second variable (different lines/colors)
            var1_levels: Values to use for var1 (default: quantiles for continuous, unique for categorical)
            var2_levels: Values to use for var2 (default: quantiles for continuous, unique for categorical)
            plot_type: 'lines', 'heatmap', or 'both'
            figsize: Figure size
            
        Returns:
            Dictionary with plot data and interpretation
        """
        # Convert data if needed
        if hasattr(data, 'as_data_frame'):
            df = data.as_data_frame(use_multi_thread=True)
        else:
            df = data.copy()
        
        # Determine levels for each variable
        if var1_levels is None:
            var1_levels = self._get_variable_levels(df[var1])
        if var2_levels is None:
            var2_levels = self._get_variable_levels(df[var2])
        
        # Calculate predictions for each combination
        predictions_matrix = np.zeros((len(var2_levels), len(var1_levels)))
        
        for i, v2_val in enumerate(var2_levels):
            for j, v1_val in enumerate(var1_levels):
                # Create prediction data
                pred_data = df.copy()
                pred_data[var1] = v1_val
                pred_data[var2] = v2_val
                
                # Get prediction
                if hasattr(self.model, 'predict'):
                    preds = self.model.predict(pred_data)
                    if hasattr(preds, 'as_data_frame'):
                        preds = preds.as_data_frame(use_multi_thread=True).values.flatten()
                    predictions_matrix[i, j] = np.mean(preds)
        
        # Create plots based on type
        if plot_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            self._create_line_plot(axes[0], var1, var2, var1_levels, var2_levels, predictions_matrix)
            self._create_heatmap(axes[1], var1, var2, var1_levels, var2_levels, predictions_matrix)
        elif plot_type == 'lines':
            fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
            self._create_line_plot(ax, var1, var2, var1_levels, var2_levels, predictions_matrix)
        else:  # heatmap
            fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
            self._create_heatmap(ax, var1, var2, var1_levels, var2_levels, predictions_matrix)
        
        plt.tight_layout()
        
        # Calculate interaction strength
        interaction_strength = self._calculate_interaction_strength(predictions_matrix)
        
        return {
            'figure': fig,
            'predictions_matrix': predictions_matrix,
            'var1_levels': var1_levels,
            'var2_levels': var2_levels,
            'interaction_strength': interaction_strength,
            'interpretation': self._interpret_interaction(var1, var2, predictions_matrix, interaction_strength)
        }
    
    def plot_categorical_interaction(self,
                                    data: Union[pd.DataFrame, Any],
                                    continuous_var: str,
                                    categorical_var: str,
                                    n_points: int = 50,
                                    figsize: Tuple[int, int] = (10, 6)) -> Dict[str, Any]:
        """Special method for categorical × continuous interactions (like temperate × plant_high).
        
        Args:
            data: Data for predictions
            continuous_var: Continuous variable (x-axis)
            categorical_var: Categorical variable (different lines)
            n_points: Number of points for continuous variable
            figsize: Figure size
            
        Returns:
            Dictionary with plot and interpretation
        """
        # Convert data if needed
        if hasattr(data, 'as_data_frame'):
            df = data.as_data_frame(use_multi_thread=True)
        else:
            df = data.copy()
        
        # Get categories
        categories = df[categorical_var].unique()
        
        # Create continuous range
        cont_min = df[continuous_var].min()
        cont_max = df[continuous_var].max()
        cont_range = np.linspace(cont_min, cont_max, n_points)
        
        # Calculate predictions for each category
        fig, ax = plt.subplots(figsize=figsize)
        
        predictions_by_category = {}
        
        for category in categories:
            predictions = []
            for cont_val in cont_range:
                pred_data = df.copy()
                pred_data[continuous_var] = cont_val
                pred_data[categorical_var] = category
                
                if hasattr(self.model, 'predict'):
                    preds = self.model.predict(pred_data)
                    if hasattr(preds, 'as_data_frame'):
                        preds = preds.as_data_frame(use_multi_thread=True).values.flatten()
                    predictions.append(np.mean(preds))
            
            predictions_by_category[category] = predictions
            
            # Plot line
            label = f"{categorical_var}={category}"
            ax.plot(cont_range, predictions, linewidth=2, label=label, marker='o', markersize=3)
        
        ax.set_xlabel(continuous_var)
        ax.set_ylabel('Predicted Fungi Richness')
        ax.set_title(f'Interaction: {continuous_var} × {categorical_var}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate slopes for each category
        slopes = {}
        for category, preds in predictions_by_category.items():
            # Simple linear regression for slope
            slope = np.polyfit(cont_range, preds, 1)[0]
            slopes[category] = slope
        
        # Test for interaction (different slopes)
        slope_diff = max(slopes.values()) - min(slopes.values())
        
        return {
            'figure': fig,
            'predictions_by_category': predictions_by_category,
            'slopes': slopes,
            'slope_difference': slope_diff,
            'interpretation': self._interpret_categorical_interaction(
                continuous_var, categorical_var, slopes, slope_diff
            )
        }
    
    def plot_h2_hypothesis(self,
                          data: Union[pd.DataFrame, Any],
                          figsize: Tuple[int, int] = (15, 5)) -> Dict[str, Any]:
        """Specialized plot for H2 hypothesis (temperate × plant_high interaction).
        
        Creates three panels:
        1. Interaction plot (lines)
        2. Difference plot (effect of temperate at different plant levels)
        3. Statistical summary
        
        Args:
            data: Data with 'temperate' and 'plant_high' variables
            figsize: Figure size
            
        Returns:
            Dictionary with comprehensive H2 analysis
        """
        # Convert data if needed
        if hasattr(data, 'as_data_frame'):
            df = data.as_data_frame(use_multi_thread=True)
        else:
            df = data.copy()
        
        # Check for required variables
        if 'temperate' not in df.columns or 'plant_high' not in df.columns:
            raise ValueError("Data must contain 'temperate' and 'plant_high' columns")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Panel 1: Basic interaction plot
        predictions = np.zeros((2, 2))  # [temperate, plant_high]
        
        for i, temp_val in enumerate([0, 1]):
            for j, plant_val in enumerate([0, 1]):
                pred_data = df.copy()
                pred_data['temperate'] = temp_val
                pred_data['plant_high'] = plant_val
                
                if hasattr(self.model, 'predict'):
                    preds = self.model.predict(pred_data)
                    if hasattr(preds, 'as_data_frame'):
                        preds = preds.as_data_frame(use_multi_thread=True).values.flatten()
                    predictions[i, j] = np.mean(preds)
        
        # Plot lines
        ax1 = axes[0]
        x = [0, 1]
        x_labels = ['Low Plant', 'High Plant']
        
        ax1.plot(x, predictions[0, :], 'b-o', linewidth=2, markersize=8, label='Non-temperate')
        ax1.plot(x, predictions[1, :], 'r--s', linewidth=2, markersize=8, label='Temperate')
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels)
        ax1.set_ylabel('Predicted Fungi Richness')
        ax1.set_title('Temperate × Plant Interaction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Effect of temperate at each plant level
        ax2 = axes[1]
        temperate_effect_low = predictions[1, 0] - predictions[0, 0]
        temperate_effect_high = predictions[1, 1] - predictions[0, 1]
        
        effects = [temperate_effect_low, temperate_effect_high]
        colors = ['blue' if e >= 0 else 'red' for e in effects]
        
        bars = ax2.bar(x_labels, effects, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Effect of Temperate Climate')
        ax2.set_title('Temperate Effect by Plant Level')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, effects):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top')
        
        # Panel 3: Statistical summary
        ax3 = axes[2]
        ax3.axis('off')
        
        # Calculate interaction metrics
        interaction_effect = (predictions[1, 1] - predictions[1, 0]) - (predictions[0, 1] - predictions[0, 0])
        
        # Create summary text
        summary_text = f"""H2 Hypothesis Test Summary
        
Predicted Values:
  Low Plant, Non-temp: {predictions[0, 0]:.2f}
  Low Plant, Temperate: {predictions[1, 0]:.2f}
  High Plant, Non-temp: {predictions[0, 1]:.2f}
  High Plant, Temperate: {predictions[1, 1]:.2f}

Effects:
  Temperate effect (low plant): {temperate_effect_low:.2f}
  Temperate effect (high plant): {temperate_effect_high:.2f}
  Interaction effect: {interaction_effect:.2f}

H2 Pattern:
  Expected: Negative temperate effect
            stronger at high plant
  Observed: {"SUPPORTED" if temperate_effect_high < 0 and abs(temperate_effect_high) > abs(temperate_effect_low) else "NOT SUPPORTED"}
"""
        
        ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        plt.tight_layout()
        
        # Determine H2 support
        h2_supported = (temperate_effect_high < 0 and 
                       abs(temperate_effect_high) > abs(temperate_effect_low))
        
        return {
            'figure': fig,
            'predictions': predictions,
            'temperate_effect_low_plant': temperate_effect_low,
            'temperate_effect_high_plant': temperate_effect_high,
            'interaction_effect': interaction_effect,
            'h2_supported': h2_supported,
            'interpretation': self._interpret_h2_results(
                predictions, temperate_effect_low, temperate_effect_high, interaction_effect
            )
        }
    
    def _get_variable_levels(self, series: pd.Series, n_levels: int = 5) -> List[float]:
        """Get representative levels for a variable."""
        if series.nunique() <= 10:
            # Categorical or few unique values
            return sorted(series.unique())
        else:
            # Continuous - use quantiles
            quantiles = np.linspace(0.1, 0.9, n_levels)
            return [series.quantile(q) for q in quantiles]
    
    def _create_line_plot(self, ax, var1: str, var2: str, 
                         var1_levels: List[float], var2_levels: List[float],
                         predictions_matrix: np.ndarray):
        """Create line plot for interaction."""
        for i, v2_val in enumerate(var2_levels):
            label = f"{var2}={v2_val:.2f}" if isinstance(v2_val, float) else f"{var2}={v2_val}"
            ax.plot(var1_levels, predictions_matrix[i, :], 
                   'o-', linewidth=2, markersize=6, label=label)
        
        ax.set_xlabel(var1)
        ax.set_ylabel('Predicted Fungi Richness')
        ax.set_title(f'Interaction: {var1} × {var2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_heatmap(self, ax, var1: str, var2: str,
                       var1_levels: List[float], var2_levels: List[float],
                       predictions_matrix: np.ndarray):
        """Create heatmap for interaction."""
        im = ax.imshow(predictions_matrix, aspect='auto', cmap='RdBu_r')
        
        # Set ticks
        ax.set_xticks(range(len(var1_levels)))
        ax.set_yticks(range(len(var2_levels)))
        
        # Set labels
        x_labels = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in var1_levels]
        y_labels = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in var2_levels]
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(f'Interaction Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    def _calculate_interaction_strength(self, predictions_matrix: np.ndarray) -> float:
        """Calculate interaction strength (0 = no interaction, higher = stronger)."""
        # Calculate row means and column means
        row_means = predictions_matrix.mean(axis=1)
        col_means = predictions_matrix.mean(axis=0)
        grand_mean = predictions_matrix.mean()
        
        # Calculate expected values under no interaction
        expected = np.outer(row_means - grand_mean, col_means - grand_mean) / grand_mean + grand_mean
        
        # Calculate interaction strength as deviation from additivity
        interaction_strength = np.sqrt(np.mean((predictions_matrix - expected) ** 2))
        
        return interaction_strength
    
    def _interpret_interaction(self, var1: str, var2: str, 
                              predictions_matrix: np.ndarray,
                              interaction_strength: float) -> str:
        """Generate interpretation of interaction."""
        if interaction_strength < 0.1:
            strength_desc = "negligible"
        elif interaction_strength < 0.5:
            strength_desc = "weak"
        elif interaction_strength < 1.0:
            strength_desc = "moderate"
        else:
            strength_desc = "strong"
        
        # Check if lines cross (qualitative interaction)
        slopes = []
        for i in range(predictions_matrix.shape[0]):
            slope = predictions_matrix[i, -1] - predictions_matrix[i, 0]
            slopes.append(slope)
        
        if len(slopes) > 1 and np.sign(slopes[0]) != np.sign(slopes[-1]):
            interaction_type = "qualitative (lines cross)"
        else:
            interaction_type = "quantitative (lines don't cross)"
        
        return (f"The interaction between {var1} and {var2} is {strength_desc} "
                f"(strength={interaction_strength:.3f}) and {interaction_type}. "
                f"The effect of {var1} varies across levels of {var2}.")
    
    def _interpret_categorical_interaction(self, continuous_var: str, categorical_var: str,
                                          slopes: Dict, slope_diff: float) -> str:
        """Interpret categorical × continuous interaction."""
        if abs(slope_diff) < 0.01:
            return f"No meaningful interaction: {continuous_var} has similar effects across {categorical_var} levels."
        
        # Find which category has stronger effect
        max_slope_cat = max(slopes, key=slopes.get)
        min_slope_cat = min(slopes, key=slopes.get)
        
        return (f"Significant interaction detected. "
                f"The effect of {continuous_var} is strongest when {categorical_var}={max_slope_cat} "
                f"(slope={slopes[max_slope_cat]:.3f}) and weakest when {categorical_var}={min_slope_cat} "
                f"(slope={slopes[min_slope_cat]:.3f}). "
                f"Slope difference: {slope_diff:.3f}")
    
    def _interpret_h2_results(self, predictions: np.ndarray,
                             temp_effect_low: float, temp_effect_high: float,
                             interaction_effect: float) -> str:
        """Interpret H2 hypothesis results."""
        if temp_effect_high < 0 and abs(temp_effect_high) > abs(temp_effect_low):
            return (f"H2 SUPPORTED: Temperate climate shows negative effect on fungi richness "
                   f"that is stronger at high plant richness ({temp_effect_high:.2f}) "
                   f"than at low plant richness ({temp_effect_low:.2f}). "
                   f"This indicates decoupling between plant and fungi in temperate regions. "
                   f"Interaction effect: {interaction_effect:.2f}")
        else:
            return (f"H2 NOT SUPPORTED: The expected pattern of stronger negative temperate effect "
                   f"at high plant richness was not observed. "
                   f"Temperate effect at low plant: {temp_effect_low:.2f}, "
                   f"at high plant: {temp_effect_high:.2f}. "
                   f"Interaction effect: {interaction_effect:.2f}")
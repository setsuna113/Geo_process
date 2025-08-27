"""
Professional map visualizer - Create publication-quality global maps.

Linus: "Talk is cheap. Show me the code."
This creates maps that actually look professional.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class ProfessionalMapVisualizer:
    """Static methods for map visualization. No cartopy dependency!"""
    
    @staticmethod
    def create_global_biodiversity_map(
        data: np.ndarray,
        name: str,
        output_path: Union[str, Path],
        title: Optional[str] = None,
        cmap: str = 'viridis',
        figsize: Tuple[int, int] = (15, 8),
        **kwargs
    ) -> Path:
        """Create a professional global biodiversity map.
        
        No projections, no cartopy. Just a clean, professional-looking map.
        
        Args:
            data: 2D array of values
            name: Dataset name
            output_path: Where to save
            title: Map title (auto-generated if None)
            cmap: Colormap
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Mask invalid values
        masked = np.ma.masked_invalid(data)
        
        # Calculate percentiles for better visualization
        valid = masked.compressed()
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [2, 98])
        else:
            vmin, vmax = 0, 1
        
        # Downsample if too large
        display_data = masked
        if masked.size > 2_000_000:
            factor = int(np.sqrt(masked.size / 2_000_000)) + 1
            display_data = masked[::factor, ::factor]
            logger.info(f"Downsampled {factor}x for display")
        
        # Create the map
        im = ax.imshow(
            display_data,
            extent=[-180, 180, -90, 90],
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Labels
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Title
        if title is None:
            title = f'{name.replace("_", " ").title()} Distribution'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Richness', rotation=270, labelpad=20)
        
        # Add statistics
        stats_text = f'Mean: {np.nanmean(data):.1f}\n'
        stats_text += f'Max: {np.nanmax(data):.1f}\n'
        stats_text += f'Coverage: {(~np.isnan(data)).mean()*100:.1f}%'
        
        ax.text(0.02, 0.02, stats_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Save
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved map to {output_path}")
        return output_path
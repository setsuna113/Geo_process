"""Biome-like visualization reporter for CP-WOPT results.

This reporter creates geographic maps showing biodiversity patterns as biome-like regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings

# Try to import cartopy for proper map projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn("Cartopy not available, using simple cylindrical projection")

logger = logging.getLogger(__name__)


class BiomeReporter:
    """Reporter for creating biome-like visualizations from CP-WOPT components."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize biome reporter.
        
        Args:
            output_dir: Directory to save visualizations
            config: Reporter configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Define color schemes for biomes
        self.biome_colors = [
            '#228B22',  # Forest Green - Component 0
            '#FFD700',  # Gold - Component 1
            '#4169E1',  # Royal Blue - Component 2
            '#DC143C',  # Crimson - Component 3
            '#FF8C00',  # Dark Orange - Component 4
            '#9370DB',  # Medium Purple - Component 5
            '#20B2AA',  # Light Sea Green - Component 6
            '#FF1493',  # Deep Pink - Component 7
            '#8B4513',  # Saddle Brown - Component 8
            '#00CED1',  # Dark Turquoise - Component 9
        ]
        
    def report(self, factors: List[np.ndarray], tensor_shape: Tuple[int, ...],
               feature_names: List[str], coordinates: Optional[Dict] = None,
               component_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate biome-like visualizations.
        
        Args:
            factors: List of factor matrices [latitude, longitude, features]
            tensor_shape: Original tensor shape
            feature_names: Names of features
            coordinates: Optional coordinate information
            component_weights: Optional component importance weights
            
        Returns:
            Dictionary of generated plot paths
        """
        logger.info("Generating biome visualizations...")
        
        plots = {}
        
        # 1. Component dominance map (main biome map)
        plots['biome_map'] = self._create_biome_map(
            factors, tensor_shape, component_weights
        )
        
        # 2. Individual component intensity maps
        plots['component_maps'] = self._create_component_maps(
            factors, tensor_shape
        )
        
        # 3. Feature-specific biome maps
        plots['feature_biomes'] = self._create_feature_biomes(
            factors, tensor_shape, feature_names
        )
        
        # 4. Mixed biodiversity visualization (RGB composite)
        plots['biodiversity_composite'] = self._create_biodiversity_composite(
            factors, tensor_shape, feature_names
        )
        
        # 5. Component overlap analysis
        plots['component_overlap'] = self._create_overlap_analysis(
            factors, tensor_shape
        )
        
        logger.info(f"Generated {len(plots)} biome visualizations")
        return plots
    
    def _create_biome_map(self, factors: List[np.ndarray], 
                          tensor_shape: Tuple[int, ...],
                          component_weights: Optional[np.ndarray] = None) -> Path:
        """Create main biome map showing dominant components.
        
        Each geographic location is colored by its dominant component,
        creating biome-like regions.
        """
        lat_factor = factors[0]  # Shape: (n_lat, n_components)
        lon_factor = factors[1]  # Shape: (n_lon, n_components)
        feat_factor = factors[2]  # Shape: (n_features, n_components)
        
        n_lat, n_lon, n_feat = tensor_shape
        n_components = lat_factor.shape[1]
        
        # Reconstruct spatial patterns for all components
        # For each location, sum across all features
        spatial_patterns = np.zeros((n_lat, n_lon, n_components))
        
        for comp in range(n_components):
            # Outer product of lat and lon factors
            spatial_pattern = np.outer(lat_factor[:, comp], lon_factor[:, comp])
            
            # Weight by sum of feature loadings for this component
            feature_weight = np.sum(np.abs(feat_factor[:, comp]))
            spatial_patterns[:, :, comp] = spatial_pattern * feature_weight
            
            # Apply component weight if provided
            if component_weights is not None:
                spatial_patterns[:, :, comp] *= component_weights[comp]
        
        # Find dominant component at each location
        dominant_component = np.argmax(np.abs(spatial_patterns), axis=2)
        
        # Create figure
        if HAS_CARTOPY:
            fig = plt.figure(figsize=(15, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, alpha=0.3)
            ax.add_feature(cfeature.OCEAN, alpha=0.1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        else:
            fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create discrete colormap
        cmap = ListedColormap(self.biome_colors[:n_components])
        
        # Create coordinate grids
        lat_bins = np.linspace(-90, 90, n_lat + 1)
        lon_bins = np.linspace(-180, 180, n_lon + 1)
        lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)
        
        # Plot biomes
        im = ax.pcolormesh(lon_grid, lat_grid, dominant_component, 
                          cmap=cmap, vmin=0, vmax=n_components-1,
                          alpha=0.8, shading='auto')
        
        # Add legend
        legend_elements = []
        for i in range(n_components):
            legend_elements.append(
                mpatches.Patch(color=self.biome_colors[i], 
                             label=f'Biome {i+1}')
            )
        ax.legend(handles=legend_elements, loc='center left', 
                 bbox_to_anchor=(1, 0.5), frameon=True)
        
        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Biodiversity Biomes from CP-WOPT Components', 
                    fontsize=16, fontweight='bold')
        
        if HAS_CARTOPY:
            ax.set_global()
            ax.gridlines(draw_labels=True, alpha=0.3)
        else:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(True, alpha=0.3)
        
        # Save
        output_path = self.output_dir / 'biome_map.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_component_maps(self, factors: List[np.ndarray],
                              tensor_shape: Tuple[int, ...]) -> Path:
        """Create individual maps for each component showing intensity."""
        lat_factor = factors[0]
        lon_factor = factors[1]
        n_components = lat_factor.shape[1]
        
        # Create grid of subplots
        n_cols = 5
        n_rows = (n_components + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for comp in range(n_components):
            ax = axes[comp]
            
            # Create spatial pattern for this component
            spatial_pattern = np.outer(lat_factor[:, comp], lon_factor[:, comp])
            
            # Plot
            im = ax.imshow(spatial_pattern, cmap='RdBu_r', aspect='auto',
                          extent=[-180, 180, -90, 90], origin='lower')
            ax.set_title(f'Component {comp + 1}', fontsize=12)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Hide unused subplots
        for i in range(n_components, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Individual Component Spatial Patterns', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'component_intensity_maps.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_feature_biomes(self, factors: List[np.ndarray],
                              tensor_shape: Tuple[int, ...],
                              feature_names: List[str]) -> Path:
        """Create biome maps for each biodiversity feature."""
        lat_factor = factors[0]
        lon_factor = factors[1]
        feat_factor = factors[2]
        
        n_features = len(feature_names)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for feat_idx in range(min(n_features, 4)):
            ax = axes[feat_idx]
            
            # Find components that load strongly on this feature
            feature_loadings = np.abs(feat_factor[feat_idx, :])
            
            # Create weighted spatial pattern for this feature
            spatial_pattern = np.zeros(tensor_shape[:2])
            for comp in range(lat_factor.shape[1]):
                component_pattern = np.outer(lat_factor[:, comp], lon_factor[:, comp])
                spatial_pattern += component_pattern * feature_loadings[comp]
            
            # Normalize
            if np.max(np.abs(spatial_pattern)) > 0:
                spatial_pattern /= np.max(np.abs(spatial_pattern))
            
            # Plot
            im = ax.imshow(spatial_pattern, cmap='YlOrRd', aspect='auto',
                          extent=[-180, 180, -90, 90], origin='lower',
                          vmin=0, vmax=1)
            ax.set_title(f'{feature_names[feat_idx]} Distribution', fontsize=14)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.046, label='Intensity')
        
        plt.suptitle('Biodiversity Feature Distributions', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'feature_biome_maps.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_biodiversity_composite(self, factors: List[np.ndarray],
                                     tensor_shape: Tuple[int, ...],
                                     feature_names: List[str]) -> Path:
        """Create RGB composite showing multiple biodiversity types."""
        lat_factor = factors[0]
        lon_factor = factors[1]
        feat_factor = factors[2]
        
        # Select three features for RGB channels
        # Preferably: plants (R), terrestrial (G), fungi (B)
        feature_indices = {
            'plants': None,
            'terrestrial': None,
            'fungi': None
        }
        
        for i, name in enumerate(feature_names):
            if 'plant' in name.lower():
                feature_indices['plants'] = i
            elif 'terrestrial' in name.lower():
                feature_indices['terrestrial'] = i
            elif 'fungi' in name.lower() or 'fung' in name.lower():
                if feature_indices['fungi'] is None:
                    feature_indices['fungi'] = i
        
        # Create RGB channels
        rgb_image = np.zeros((*tensor_shape[:2], 3))
        
        channels = ['plants', 'terrestrial', 'fungi']
        colors = ['red', 'green', 'blue']
        
        for chan_idx, (channel, color) in enumerate(zip(channels, colors)):
            feat_idx = feature_indices[channel]
            if feat_idx is not None:
                # Create spatial pattern for this feature
                spatial_pattern = np.zeros(tensor_shape[:2])
                feature_loadings = np.abs(feat_factor[feat_idx, :])
                
                for comp in range(lat_factor.shape[1]):
                    component_pattern = np.outer(lat_factor[:, comp], lon_factor[:, comp])
                    spatial_pattern += component_pattern * feature_loadings[comp]
                
                # Normalize to [0, 1]
                if np.max(spatial_pattern) > 0:
                    spatial_pattern = spatial_pattern / np.max(spatial_pattern)
                
                rgb_image[:, :, chan_idx] = spatial_pattern
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot RGB composite
        im = ax.imshow(rgb_image, aspect='auto',
                      extent=[-180, 180, -90, 90], origin='lower')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='red', label='Plants richness'),
            mpatches.Patch(color='green', label='Terrestrial richness'),
            mpatches.Patch(color='blue', label='Fungi richness'),
            mpatches.Patch(color='yellow', label='Plants + Terrestrial'),
            mpatches.Patch(color='cyan', label='Terrestrial + Fungi'),
            mpatches.Patch(color='magenta', label='Plants + Fungi'),
            mpatches.Patch(color='white', label='All types')
        ]
        ax.legend(handles=legend_elements, loc='center left',
                 bbox_to_anchor=(1, 0.5), frameon=True)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Biodiversity Composite Map (RGB)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'biodiversity_composite.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_overlap_analysis(self, factors: List[np.ndarray],
                               tensor_shape: Tuple[int, ...]) -> Path:
        """Analyze and visualize component overlap."""
        lat_factor = factors[0]
        lon_factor = factors[1]
        n_components = lat_factor.shape[1]
        
        # Calculate spatial patterns for all components
        spatial_patterns = []
        for comp in range(n_components):
            pattern = np.outer(lat_factor[:, comp], lon_factor[:, comp])
            # Normalize
            if np.max(np.abs(pattern)) > 0:
                pattern = pattern / np.max(np.abs(pattern))
            spatial_patterns.append(pattern)
        
        # Count overlaps (how many components are strong at each location)
        threshold = 0.3  # Consider component "present" if above threshold
        overlap_count = np.zeros(tensor_shape[:2])
        
        for pattern in spatial_patterns:
            overlap_count += (np.abs(pattern) > threshold).astype(int)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Overlap count
        im1 = ax1.imshow(overlap_count, cmap='hot', aspect='auto',
                        extent=[-180, 180, -90, 90], origin='lower')
        ax1.set_title('Component Overlap Count', fontsize=14)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(im1, ax=ax1, label='Number of Components')
        
        # Plot 2: Diversity index (entropy of component strengths)
        patterns_array = np.array(spatial_patterns)
        patterns_array = np.abs(patterns_array)
        
        # Calculate entropy at each location
        diversity = np.zeros(tensor_shape[:2])
        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                strengths = patterns_array[:, i, j]
                if np.sum(strengths) > 0:
                    probs = strengths / np.sum(strengths)
                    # Shannon entropy
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    diversity[i, j] = entropy
        
        im2 = ax2.imshow(diversity, cmap='viridis', aspect='auto',
                        extent=[-180, 180, -90, 90], origin='lower')
        ax2.set_title('Component Diversity (Entropy)', fontsize=14)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(im2, ax=ax2, label='Diversity Index')
        
        plt.suptitle('Component Overlap and Diversity Analysis', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'component_overlap_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
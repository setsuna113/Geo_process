"""
Global Biodiversity Map Visualizer
Creates individual global maps for each biodiversity dataset with optional log transformation.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Dict, Optional, Tuple
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import warnings
warnings.filterwarnings('ignore')

class GlobalBiodiversityMapper:
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis/global_maps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for each dataset
        self.color_schemes = {
            'plants': 'Greens',
            'terrestrial': 'Blues', 
            'am_fungi': 'Oranges',
            'ecm_fungi': 'Purples'
        }
        
        # Dataset display names
        self.display_names = {
            'plants': 'Plant Species Richness',
            'terrestrial': 'Terrestrial Vertebrate Richness',
            'am_fungi': 'AM Fungi Species Richness',
            'ecm_fungi': 'EcM Fungi Species Richness'
        }
    
    def load_raster_data(self, raster_path: str) -> Tuple[np.ndarray, Dict]:
        """Load raster data and metadata."""
        print(f"Loading: {raster_path}")
        
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            
            # Get geographic bounds
            bounds = src.bounds
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            
            # Calculate coordinate arrays more efficiently
            # Get corner coordinates and create 1D arrays
            west, south, east, north = bounds
            
            # Create 1D coordinate arrays
            x_coords = np.linspace(west, east, width)
            y_coords = np.linspace(north, south, height)  # North to south
            
            # Create 2D coordinate meshgrids
            lon, lat = np.meshgrid(x_coords, y_coords)
            
            metadata = {
                'bounds': bounds,
                'transform': transform,
                'crs': crs,
                'width': width,
                'height': height,
                'lon': lon,
                'lat': lat
            }
            
            return data, metadata
    
    def create_individual_map(self, dataset_name: str, raster_path: str) -> None:
        """Create individual global map for one dataset (raw values only)."""
        
        data, metadata = self.load_raster_data(raster_path)
        
        # Mask invalid values
        data = np.ma.masked_where((data <= 0) | (data > 1e6), data)
        
        # Calculate statistics
        valid_data = data.compressed()
        if len(valid_data) == 0:
            print(f"Warning: No valid data for {dataset_name}")
            return
            
        stats = {
            'min': np.min(valid_data),
            'max': np.max(valid_data), 
            'mean': np.mean(valid_data),
            'median': np.median(valid_data),
            'std': np.std(valid_data),
            'count': len(valid_data),
            'p95': np.percentile(valid_data, 95),
            'p99': np.percentile(valid_data, 99)
        }
        
        print(f"{dataset_name} stats: min={stats['min']:.1f}, max={stats['max']:.1f}, "
              f"mean={stats['mean']:.1f}, median={stats['median']:.1f}, 95th={stats['p95']:.1f}")
        
        # Create figure with single plot
        fig = plt.figure(figsize=(16, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Plot raw data only
        self._plot_map(ax, data, metadata, dataset_name, stats=stats)
        ax.set_title(f"{self.display_names[dataset_name]} - Global Distribution", 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f"{dataset_name}_global_map_raw.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def _plot_map(self, ax, data, metadata, dataset_name: str, stats: Dict = None):
        """Plot map on given axis (raw data only)."""
        
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE, alpha=0.8, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, alpha=0.5, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.1)
        
        # Set global extent
        ax.set_global()
        
        # Use raw data with percentile-based normalization to handle extreme outliers
        # Use 95th percentile for better visualization of the majority of data
        vmax = stats['p95'] if stats else np.percentile(data.compressed(), 95)
        norm = Normalize(vmin=0, vmax=vmax)
        
        # Create the map
        lon = metadata['lon']
        lat = metadata['lat']
        
        im = ax.pcolormesh(lon, lat, data, 
                          transform=ccrs.PlateCarree(),
                          cmap=self.color_schemes[dataset_name],
                          norm=norm,
                          shading='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8, aspect=30)
        cbar.set_label(f"Species Richness (values >95th percentile shown as max)", fontsize=11)
        
        # Add comprehensive statistics text
        if stats:
            text = (f"Valid pixels: {stats['count']:,}\n"
                   f"Range: {stats['min']:.1f} - {stats['max']:.1f}\n"
                   f"Mean: {stats['mean']:.1f} | Median: {stats['median']:.1f}\n"
                   f"95th percentile: {stats['p95']:.1f}")
            
            ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=9)
        
        # Add gridlines
        ax.gridlines(draw_labels=True, alpha=0.3)
    
    def create_comparison_grid(self, datasets: Dict[str, str]) -> None:
        """Create a 2x2 grid comparing all datasets (raw scale only)."""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), 
                                subplot_kw={'projection': ccrs.PlateCarree()})
        fig.suptitle('Global Biodiversity Richness Patterns - Raw Data', fontsize=18, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, (dataset_name, raster_path) in enumerate(datasets.items()):
            if i >= 4:  # Safety check
                break
                
            ax = axes_flat[i]
            
            try:
                data, metadata = self.load_raster_data(raster_path)
                data = np.ma.masked_where((data <= 0) | (data > 1e6), data)
                
                valid_data = data.compressed()
                if len(valid_data) == 0:
                    ax.text(0.5, 0.5, f"No valid data\nfor {dataset_name}", 
                           transform=ax.transAxes, ha='center', va='center')
                    continue
                
                stats = {
                    'min': np.min(valid_data),
                    'max': np.max(valid_data),
                    'median': np.median(valid_data),
                    'mean': np.mean(valid_data),
                    'count': len(valid_data),
                    'p95': np.percentile(valid_data, 95)
                }
                
                # Use 95th percentile for better visualization
                vmax = stats['p95']
                
                # Add geographic features
                ax.add_feature(cfeature.COASTLINE, alpha=0.8, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, alpha=0.5, linewidth=0.3)
                ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
                ax.set_global()
                
                # Plot raw data
                im = ax.pcolormesh(metadata['lon'], metadata['lat'], data,
                                  transform=ccrs.PlateCarree(),
                                  cmap=self.color_schemes[dataset_name],
                                  norm=Normalize(vmin=0, vmax=vmax),
                                  shading='nearest')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                                   pad=0.08, shrink=0.8, aspect=20)
                cbar.set_label("Species Richness", fontsize=10)
                
                ax.set_title(f"{self.display_names[dataset_name]}", 
                           fontsize=12, fontweight='bold')
                
                # Add comprehensive statistics
                stats_text = (f"Max: {stats['max']:.0f}\n"
                             f"Mean: {stats['mean']:.1f}\n"
                             f"Median: {stats['median']:.1f}")
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=8)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                ax.text(0.5, 0.5, f"Error loading\n{dataset_name}", 
                       transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        # Save comparison grid
        output_path = self.output_dir / f"biodiversity_comparison_grid_raw.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison grid: {output_path}")
        plt.close()
    
    def generate_all_maps(self, datasets: Dict[str, str]) -> None:
        """Generate all individual maps and comparison grid (raw scale only)."""
        
        print("Creating individual global maps (raw scale)...")
        for dataset_name, raster_path in datasets.items():
            try:
                self.create_individual_map(dataset_name, raster_path)
            except Exception as e:
                print(f"Error creating map for {dataset_name}: {e}")
        
        print("\nCreating comparison grid (raw scale)...")
        try:
            self.create_comparison_grid(datasets)
        except Exception as e:
            print(f"Error creating comparison grid: {e}")
        
        print(f"\nAll raw-scale maps saved to: {self.output_dir}")

def main():
    """Main function to create global biodiversity maps."""
    
    # Define datasets
    datasets = {
        'plants': '/scratch/yl998/downsampled_rasters/plants-richness_downsampled_0.16667.tif',
        'terrestrial': '/scratch/yl998/downsampled_rasters/terrestrial-richness_downsampled_0.16667.tif', 
        'am_fungi': '/scratch/yl998/downsampled_rasters/am-fungi-richness_downsampled_0.16667.tif',
        'ecm_fungi': '/scratch/yl998/downsampled_rasters/ecm-fungi-richness_downsampled_0.16667.tif'
    }
    
    mapper = GlobalBiodiversityMapper()
    mapper.generate_all_maps(datasets)

if __name__ == "__main__":
    main()
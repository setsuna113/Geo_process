#!/usr/bin/env python
"""
Latitudinal Analyzer: Shows how biodiversity datasets change across latitude.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class LatitudinalAnalyzer:
    """Analyze biodiversity patterns across latitude"""
    
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis/latitudinal"):
        """Initialize analyzer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_latitudinal_patterns(self, datasets: Dict[str, str], sample_factor: int = 5) -> None:
        """Analyze and visualize latitudinal patterns for all datasets"""
        print("Analyzing latitudinal patterns...")
        
        # Load and process data
        lat_profiles = {}
        
        for name, path in datasets.items():
            print(f"  Processing {name}...")
            lat_data, lat_bins = self._extract_latitudinal_profile(path, sample_factor)
            lat_profiles[name] = {
                'mean': lat_data['mean'],
                'std': lat_data['std'],
                'count': lat_data['count'],
                'median': lat_data['median'],
                'p25': lat_data['p25'],
                'p75': lat_data['p75']
            }
        
        # Create comprehensive visualization
        self._create_latitudinal_plots(lat_profiles, lat_bins)
        
        # Generate detailed analysis
        self._analyze_patterns(lat_profiles, lat_bins)
        
        print(f"Latitudinal analysis saved to: {self.output_dir}")
    
    def _extract_latitudinal_profile(self, raster_path: str, sample_factor: int = 5) -> Tuple[Dict, np.ndarray]:
        """Extract latitudinal profile from raster data"""
        
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            
            # Clean up corrupt data (handle extreme negative values)
            # These appear to be NoData values incorrectly encoded
            data[data < -1e10] = np.nan  # Replace extreme negatives with NaN
            data[data < 0] = np.nan      # Also remove any remaining negatives for species richness
            
            # Get geographic coordinates
            height, width = data.shape
            
            # Calculate latitude for each row (using transform)
            # Transform gives us: (top-left x, pixel width, 0, top-left y, 0, -pixel height)
            top_lat = transform[3]  # Top latitude
            pixel_height = abs(transform[5])  # Pixel height in degrees
            
            # Create latitude array for each row
            latitudes = np.array([top_lat - (i * pixel_height) for i in range(height)])
            
            # Apply spatial sampling to reduce computation
            if sample_factor > 1:
                step = sample_factor
                data = data[::step, ::step]
                latitudes = latitudes[::step]
            
            # Create latitude bins (1-degree resolution)
            lat_min = max(-90, np.min(latitudes) - 1)
            lat_max = min(90, np.max(latitudes) + 1)
            lat_bins = np.arange(lat_min, lat_max + 1, 1.0)
            lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
            
            # Initialize arrays for statistics
            lat_stats = {
                'mean': np.full(len(lat_centers), np.nan),
                'std': np.full(len(lat_centers), np.nan),
                'count': np.zeros(len(lat_centers), dtype=int),
                'median': np.full(len(lat_centers), np.nan),
                'p25': np.full(len(lat_centers), np.nan),
                'p75': np.full(len(lat_centers), np.nan)
            }
            
            # Calculate statistics for each latitude band
            for i, (lat_min_bin, lat_max_bin) in enumerate(zip(lat_bins[:-1], lat_bins[1:])):
                # Find rows within this latitude band
                lat_mask = (latitudes >= lat_min_bin) & (latitudes < lat_max_bin)
                
                if np.any(lat_mask):
                    # Extract data for this latitude band
                    lat_data = data[lat_mask, :]
                    
                    # Remove NaN and zero values
                    valid_data = lat_data[(lat_data > 0) & ~np.isnan(lat_data)]
                    
                    if len(valid_data) > 0:
                        lat_stats['mean'][i] = np.mean(valid_data)
                        lat_stats['std'][i] = np.std(valid_data)
                        lat_stats['count'][i] = len(valid_data)
                        lat_stats['median'][i] = np.median(valid_data)
                        lat_stats['p25'][i] = np.percentile(valid_data, 25)
                        lat_stats['p75'][i] = np.percentile(valid_data, 75)
        
        return lat_stats, lat_centers
    
    def _create_latitudinal_plots(self, profiles: Dict, lat_centers: np.ndarray) -> None:
        """Create comprehensive latitudinal analysis plots"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Biodiversity Patterns Across Latitude', fontsize=16, fontweight='bold')
        
        colors = {
            'plants': 'green',
            'terrestrial': 'blue', 
            'am_fungi': 'orange',
            'ecm_fungi': 'red'
        }
        
        # 1. Mean richness across latitude
        ax1 = axes[0, 0]
        for name, profile in profiles.items():
            valid_mask = ~np.isnan(profile['mean'])
            if np.any(valid_mask):
                ax1.plot(lat_centers[valid_mask], profile['mean'][valid_mask], 
                        color=colors.get(name, 'black'), label=name.replace('_', ' ').title(), 
                        linewidth=2, marker='o', markersize=3)
        
        ax1.set_xlabel('Latitude (°)')
        ax1.set_ylabel('Mean Species Richness')
        ax1.set_title('A) Mean Species Richness vs Latitude', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-60, 80)
        
        # 2. Median richness with quartiles (focus on EcM fungi pattern)
        ax2 = axes[0, 1]
        for name, profile in profiles.items():
            valid_mask = ~np.isnan(profile['median'])
            if np.any(valid_mask):
                lats = lat_centers[valid_mask]
                median = profile['median'][valid_mask]
                p25 = profile['p25'][valid_mask]
                p75 = profile['p75'][valid_mask]
                
                ax2.plot(lats, median, color=colors.get(name, 'black'), 
                        label=name.replace('_', ' ').title(), linewidth=2)
                ax2.fill_between(lats, p25, p75, color=colors.get(name, 'black'), 
                               alpha=0.2)
        
        ax2.set_xlabel('Latitude (°)')
        ax2.set_ylabel('Median Species Richness')
        ax2.set_title('B) Median Richness with Quartiles', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-60, 80)
        
        # 3. Data availability across latitude
        ax3 = axes[1, 0]
        for name, profile in profiles.items():
            valid_mask = profile['count'] > 0
            if np.any(valid_mask):
                ax3.plot(lat_centers[valid_mask], profile['count'][valid_mask], 
                        color=colors.get(name, 'black'), label=name.replace('_', ' ').title(), 
                        linewidth=2)
        
        ax3.set_xlabel('Latitude (°)')
        ax3.set_ylabel('Number of Valid Pixels')
        ax3.set_title('C) Data Availability Across Latitude', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-60, 80)
        
        # 4. Coefficient of variation (variability)
        ax4 = axes[1, 1]
        for name, profile in profiles.items():
            valid_mask = (~np.isnan(profile['mean'])) & (~np.isnan(profile['std'])) & (profile['mean'] > 0)
            if np.any(valid_mask):
                cv = profile['std'][valid_mask] / profile['mean'][valid_mask]
                ax4.plot(lat_centers[valid_mask], cv, 
                        color=colors.get(name, 'black'), label=name.replace('_', ' ').title(), 
                        linewidth=2, marker='s', markersize=3)
        
        ax4.set_xlabel('Latitude (°)')
        ax4.set_ylabel('Coefficient of Variation')
        ax4.set_title('D) Variability Across Latitude', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-60, 80)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / "latitudinal_patterns_comprehensive.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved comprehensive plot: {output_path}")
        plt.close()
        
        # Create focused EcM fungi analysis
        self._create_ecm_focus_plot(profiles, lat_centers, colors)
    
    def _create_ecm_focus_plot(self, profiles: Dict, lat_centers: np.ndarray, colors: Dict) -> None:
        """Create focused plot highlighting EcM fungi patterns"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('EcM Fungi Latitudinal Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: EcM vs other datasets
        ax1.set_title('A) EcM Fungi vs Other Datasets', fontweight='bold')
        
        for name, profile in profiles.items():
            valid_mask = ~np.isnan(profile['mean'])
            if np.any(valid_mask):
                lats = lat_centers[valid_mask]
                means = profile['mean'][valid_mask]
                
                if name == 'ecm_fungi':
                    # Highlight EcM fungi with thicker line and markers
                    ax1.plot(lats, means, color=colors[name], label='EcM Fungi', 
                            linewidth=4, marker='o', markersize=6, zorder=10)
                else:
                    ax1.plot(lats, means, color=colors[name], 
                            label=name.replace('_', ' ').title(), linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Latitude (°)')
        ax1.set_ylabel('Mean Species Richness')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-60, 80)
        
        # Add annotations for key features
        if 'ecm_fungi' in profiles:
            ecm_profile = profiles['ecm_fungi']
            valid_mask = ~np.isnan(ecm_profile['mean'])
            if np.any(valid_mask):
                lats = lat_centers[valid_mask]
                means = ecm_profile['mean'][valid_mask]
                
                # Find the characteristic drop
                # Look for steep decline in northern latitudes
                gradients = np.gradient(means, lats)
                steep_decline_idx = np.argmin(gradients)
                
                if steep_decline_idx < len(lats) - 1:
                    decline_lat = lats[steep_decline_idx]
                    decline_richness = means[steep_decline_idx]
                    
                    ax1.annotate(f'Steep decline\nat {decline_lat:.1f}°N', 
                               xy=(decline_lat, decline_richness), xytext=(decline_lat + 15, decline_richness + 5),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2),
                               fontsize=12, color='red', fontweight='bold')
        
        # Plot 2: EcM fungi detailed analysis
        ax2.set_title('B) EcM Fungi Detailed Pattern', fontweight='bold')
        
        if 'ecm_fungi' in profiles:
            profile = profiles['ecm_fungi']
            valid_mask = ~np.isnan(profile['median'])
            
            if np.any(valid_mask):
                lats = lat_centers[valid_mask]
                median = profile['median'][valid_mask]
                p25 = profile['p25'][valid_mask]
                p75 = profile['p75'][valid_mask]
                
                # Plot median with quartiles
                ax2.plot(lats, median, color='red', linewidth=3, label='Median')
                ax2.fill_between(lats, p25, p75, color='red', alpha=0.3, label='25th-75th percentile')
                
                # Plot mean for comparison
                mean_valid = ~np.isnan(profile['mean'])
                if np.any(mean_valid):
                    ax2.plot(lat_centers[mean_valid], profile['mean'][mean_valid], 
                            color='darkred', linewidth=2, linestyle='--', label='Mean')
        
        ax2.set_xlabel('Latitude (°)')
        ax2.set_ylabel('EcM Fungi Richness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-60, 80)
        
        plt.tight_layout()
        
        # Save the focused plot
        output_path = self.output_dir / "ecm_fungi_latitudinal_focus.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved EcM focus plot: {output_path}")
        plt.close()
    
    def _analyze_patterns(self, profiles: Dict, lat_centers: np.ndarray) -> None:
        """Analyze and report key patterns"""
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("LATITUDINAL BIODIVERSITY PATTERNS ANALYSIS")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Analyze each dataset
        for name, profile in profiles.items():
            report_lines.append(f"{name.replace('_', ' ').upper()}:")
            report_lines.append("-" * (len(name) + 1))
            
            valid_mask = ~np.isnan(profile['mean'])
            if np.any(valid_mask):
                lats = lat_centers[valid_mask]
                means = profile['mean'][valid_mask]
                
                # Find peaks and patterns
                max_idx = np.argmax(means)
                min_idx = np.argmin(means)
                
                max_lat = lats[max_idx]
                max_richness = means[max_idx]
                min_lat = lats[min_idx]
                min_richness = means[min_idx]
                
                report_lines.append(f"  Peak richness: {max_richness:.1f} species at {max_lat:.1f}°N")
                report_lines.append(f"  Minimum richness: {min_richness:.1f} species at {min_lat:.1f}°N")
                
                # Calculate gradients to find steepest changes (only if enough data points)
                if len(means) > 2:
                    gradients = np.gradient(means, lats)
                    steepest_decline_idx = np.argmin(gradients)
                    steepest_increase_idx = np.argmax(gradients)
                    
                    if steepest_decline_idx < len(lats):
                        decline_lat = lats[steepest_decline_idx]
                        decline_rate = gradients[steepest_decline_idx]
                        report_lines.append(f"  Steepest decline: {decline_rate:.2f} species/degree at {decline_lat:.1f}°N")
                    
                    if steepest_increase_idx < len(lats):
                        increase_lat = lats[steepest_increase_idx]
                        increase_rate = gradients[steepest_increase_idx]
                        report_lines.append(f"  Steepest increase: {increase_rate:.2f} species/degree at {increase_lat:.1f}°N")
                
                # Latitudinal ranges
                tropical_mask = (lats >= -23.5) & (lats <= 23.5)
                temperate_mask = ((lats > 23.5) & (lats <= 66.5)) | ((lats >= -66.5) & (lats < -23.5))
                
                if np.any(tropical_mask):
                    tropical_mean = np.mean(means[tropical_mask])
                    report_lines.append(f"  Tropical mean (±23.5°): {tropical_mean:.1f} species")
                
                if np.any(temperate_mask):
                    temperate_mean = np.mean(means[temperate_mask])
                    report_lines.append(f"  Temperate mean (23.5-66.5°): {temperate_mean:.1f} species")
            
            report_lines.append("")
        
        # Special analysis for EcM fungi
        if 'ecm_fungi' in profiles:
            report_lines.append("SPECIAL ANALYSIS: EcM FUNGI LATITUDINAL DROP")
            report_lines.append("=" * 50)
            
            profile = profiles['ecm_fungi']
            valid_mask = ~np.isnan(profile['mean'])
            
            if np.any(valid_mask):
                lats = lat_centers[valid_mask]
                means = profile['mean'][valid_mask]
                
                # Find the characteristic northern drop
                northern_mask = lats > 40  # Focus on northern latitudes
                if np.any(northern_mask):
                    northern_lats = lats[northern_mask]
                    northern_means = means[northern_mask]
                    
                    # Calculate decline rate in northern regions
                    if len(northern_means) > 2:
                        northern_gradient = np.gradient(northern_means, northern_lats)
                        avg_decline = np.mean(northern_gradient[northern_gradient < 0])
                        
                        report_lines.append(f"  Average decline rate north of 40°N: {avg_decline:.2f} species/degree")
                        
                        # Find where decline is steepest
                        steepest_idx = np.argmin(northern_gradient)
                        steepest_lat = northern_lats[steepest_idx]
                        steepest_rate = northern_gradient[steepest_idx]
                        
                        report_lines.append(f"  Steepest decline at {steepest_lat:.1f}°N: {steepest_rate:.2f} species/degree")
                
                report_lines.append("")
                report_lines.append("ECOLOGICAL INTERPRETATION:")
                report_lines.append("  • EcM fungi are associated with temperate and boreal forests")
                report_lines.append("  • Peak richness typically in temperate regions (30-60°N)")
                report_lines.append("  • Sharp decline at northern tree line (~60-70°N)")
                report_lines.append("  • Limited presence in tropical regions (different fungal communities)")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.output_dir / "latitudinal_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"    Saved analysis report: {report_path}")
        
        # Print key findings to console
        print("\n🔍 KEY LATITUDINAL PATTERNS IDENTIFIED:")
        if 'ecm_fungi' in profiles:
            profile = profiles['ecm_fungi']
            valid_mask = ~np.isnan(profile['mean'])
            if np.any(valid_mask):
                lats = lat_centers[valid_mask]
                means = profile['mean'][valid_mask]
                gradients = np.gradient(means, lats)
                steepest_idx = np.argmin(gradients)
                
                print(f"   • EcM fungi shows characteristic drop at ~{lats[steepest_idx]:.1f}°N")
                print(f"   • Decline rate: {gradients[steepest_idx]:.2f} species per degree latitude")


def main():
    """Main function to run latitudinal analysis."""
    datasets = {
        'plants': '/scratch/yl998/downsampled_rasters/plants-richness_downsampled_0.16667.tif',
        'terrestrial': '/scratch/yl998/downsampled_rasters/terrestrial-richness_downsampled_0.16667.tif',
        'am_fungi': '/scratch/yl998/downsampled_rasters/am-fungi-richness_downsampled_0.16667.tif',
        'ecm_fungi': '/scratch/yl998/downsampled_rasters/ecm-fungi-richness_downsampled_0.16667.tif'
    }
    
    analyzer = LatitudinalAnalyzer()
    analyzer.analyze_latitudinal_patterns(datasets)


if __name__ == "__main__":
    main()
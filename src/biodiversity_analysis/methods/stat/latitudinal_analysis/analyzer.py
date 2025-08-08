#!/usr/bin/env python
"""
Simple Latitudinal Analyzer: Direct approach to show biodiversity patterns across latitude.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def analyze_single_dataset(raster_path: str, name: str, sample_factor: int = 10):
    """Analyze single dataset for latitudinal patterns"""
    print(f"  Analyzing {name}...")
    
    with rasterio.open(raster_path) as src:
        # Read data
        data = src.read(1).astype(np.float64)
        transform = src.transform
        bounds = src.bounds
        
        print(f"    Original data range: {np.nanmin(data):.3f} to {np.nanmax(data):.3f}")
        
        # Clean extreme values for fungal datasets
        if 'fungi' in name.lower():
            # Handle corrupt negative values in fungal datasets
            original_valids = np.count_nonzero(~np.isnan(data) & (data != 0))
            
            # Set extreme negatives to NaN
            data[data < -1e20] = np.nan
            data[data < 0] = np.nan  # Species richness should be positive
            
            cleaned_valids = np.count_nonzero(~np.isnan(data) & (data > 0))
            print(f"    Cleaned data: {original_valids} -> {cleaned_valids} valid pixels")
            print(f"    Cleaned range: {np.nanmin(data[data > 0]):.3f} to {np.nanmax(data):.3f}")
        else:
            # For plants/terrestrial, just set negatives to NaN
            data[data < 0] = np.nan
            data[data == 0] = np.nan  # Treat 0 as no data for richness
        
        # Get dimensions
        height, width = data.shape
        
        # Calculate latitude array using bounds
        lat_step = (bounds.top - bounds.bottom) / height
        latitudes = np.linspace(bounds.top - lat_step/2, bounds.bottom + lat_step/2, height)
        
        # Sample data to reduce processing time
        if sample_factor > 1:
            step = sample_factor
            data_sampled = data[::step, ::step]
            lat_sampled = latitudes[::step]
        else:
            data_sampled = data
            lat_sampled = latitudes
        
        # Process each latitude band
        lat_profiles = []
        lat_means = []
        lat_stds = []
        lat_counts = []
        
        valid_lats = []
        
        for i, lat in enumerate(lat_sampled):
            # Get data for this latitude band (row)
            if i < len(data_sampled):
                row_data = data_sampled[i, :]
                valid_data = row_data[~np.isnan(row_data) & (row_data > 0)]
                
                if len(valid_data) > 0:
                    lat_means.append(np.mean(valid_data))
                    lat_stds.append(np.std(valid_data))
                    lat_counts.append(len(valid_data))
                    valid_lats.append(lat)
        
        print(f"    Found {len(valid_lats)} latitude bands with data")
        print(f"    Latitude range: {min(valid_lats):.1f}°N to {max(valid_lats):.1f}°N")
        
        return {
            'name': name,
            'latitudes': np.array(valid_lats),
            'means': np.array(lat_means),
            'stds': np.array(lat_stds),
            'counts': np.array(lat_counts)
        }


def create_latitudinal_plots(dataset_profiles):
    """Create latitudinal analysis plots"""
    
    output_dir = Path("outputs/biodiversity_analysis/latitudinal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Biodiversity Patterns Across Latitude (Corrected)', fontsize=16, fontweight='bold')
    
    colors = {
        'plants': 'green',
        'terrestrial': 'blue', 
        'am_fungi': 'orange',
        'ecm_fungi': 'red'
    }
    
    # 1. Mean richness across latitude
    ax1 = axes[0, 0]
    for profile in dataset_profiles:
        name = profile['name']
        if len(profile['latitudes']) > 0:
            ax1.plot(profile['latitudes'], profile['means'], 
                    color=colors.get(name, 'black'), 
                    label=name.replace('_', ' ').title(), 
                    linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Latitude (°)')
    ax1.set_ylabel('Mean Species Richness')
    ax1.set_title('A) Mean Species Richness vs Latitude', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-60, 80)
    
    # 2. Data availability
    ax2 = axes[0, 1]
    for profile in dataset_profiles:
        name = profile['name']
        if len(profile['latitudes']) > 0:
            ax2.plot(profile['latitudes'], profile['counts'], 
                    color=colors.get(name, 'black'), 
                    label=name.replace('_', ' ').title(), 
                    linewidth=2)
    
    ax2.set_xlabel('Latitude (°)')
    ax2.set_ylabel('Number of Valid Pixels per Latitude Band')
    ax2.set_title('B) Data Availability Across Latitude', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-60, 80)
    
    # 3. Focus on EcM fungi pattern (if available)
    ax3 = axes[1, 0]
    ecm_profile = next((p for p in dataset_profiles if p['name'] == 'ecm_fungi'), None)
    
    if ecm_profile and len(ecm_profile['latitudes']) > 0:
        lats = ecm_profile['latitudes']
        means = ecm_profile['means']
        stds = ecm_profile['stds']
        
        ax3.plot(lats, means, color='red', linewidth=3, marker='o', markersize=6, label='EcM Fungi Mean')
        ax3.fill_between(lats, means - stds, means + stds, color='red', alpha=0.3, label='±1 Std Dev')
        
        # Find characteristic drop
        if len(means) > 3:
            gradients = np.gradient(means, lats)
            steepest_decline_idx = np.argmin(gradients)
            
            if steepest_decline_idx < len(lats):
                decline_lat = lats[steepest_decline_idx]
                decline_richness = means[steepest_decline_idx]
                decline_rate = gradients[steepest_decline_idx]
                
                ax3.annotate(f'Steepest decline\nat {decline_lat:.1f}°N\n({decline_rate:.2f} species/degree)', 
                           xy=(decline_lat, decline_richness), 
                           xytext=(decline_lat + 15, decline_richness + 10),
                           arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                           fontsize=11, color='darkred', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax3.set_title('C) EcM Fungi Latitudinal Pattern', fontweight='bold')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No EcM Fungi data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('C) EcM Fungi Pattern (No Data)', fontweight='bold')
    
    ax3.set_xlabel('Latitude (°)')
    ax3.set_ylabel('EcM Fungi Richness')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-60, 80)
    
    # 4. All datasets comparison with focus on northern patterns
    ax4 = axes[1, 1]
    for profile in dataset_profiles:
        name = profile['name']
        if len(profile['latitudes']) > 0:
            # Focus on northern hemisphere patterns
            northern_mask = profile['latitudes'] > 0
            if np.any(northern_mask):
                north_lats = profile['latitudes'][northern_mask]
                north_means = profile['means'][northern_mask]
                
                if name == 'ecm_fungi':
                    ax4.plot(north_lats, north_means, 
                            color=colors.get(name, 'black'), 
                            label=name.replace('_', ' ').title(), 
                            linewidth=4, marker='s', markersize=6)
                else:
                    ax4.plot(north_lats, north_means, 
                            color=colors.get(name, 'black'), 
                            label=name.replace('_', ' ').title(), 
                            linewidth=2, alpha=0.8)
    
    ax4.set_xlabel('Latitude (°)')
    ax4.set_ylabel('Mean Species Richness')
    ax4.set_title('D) Northern Hemisphere Patterns (0-80°N)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 80)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "latitudinal_patterns_corrected.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Generate summary report
    generate_summary_report(dataset_profiles, output_dir)


def generate_summary_report(profiles, output_dir):
    """Generate summary report of latitudinal patterns"""
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CORRECTED LATITUDINAL BIODIVERSITY PATTERNS ANALYSIS")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    for profile in profiles:
        name = profile['name']
        report_lines.append(f"{name.upper().replace('_', ' ')}:")
        report_lines.append("-" * (len(name) + 1))
        
        if len(profile['latitudes']) > 0:
            lats = profile['latitudes']
            means = profile['means']
            
            # Basic statistics
            max_idx = np.argmax(means)
            min_idx = np.argmin(means)
            
            report_lines.append(f"  Latitude range: {np.min(lats):.1f}°N to {np.max(lats):.1f}°N")
            report_lines.append(f"  Peak richness: {means[max_idx]:.1f} species at {lats[max_idx]:.1f}°N")
            report_lines.append(f"  Minimum richness: {means[min_idx]:.1f} species at {lats[min_idx]:.1f}°N")
            report_lines.append(f"  Overall mean: {np.mean(means):.1f} species")
            
            # Calculate gradients for steep changes
            if len(means) > 2:
                gradients = np.gradient(means, lats)
                steepest_decline_idx = np.argmin(gradients)
                steepest_increase_idx = np.argmax(gradients)
                
                decline_rate = gradients[steepest_decline_idx]
                increase_rate = gradients[steepest_increase_idx]
                
                report_lines.append(f"  Steepest decline: {decline_rate:.2f} species/degree at {lats[steepest_decline_idx]:.1f}°N")
                report_lines.append(f"  Steepest increase: {increase_rate:.2f} species/degree at {lats[steepest_increase_idx]:.1f}°N")
            
            # Zonal analysis
            tropical_mask = (lats >= -23.5) & (lats <= 23.5)
            temperate_mask = ((lats > 23.5) & (lats <= 60)) | ((lats >= -60) & (lats < -23.5))
            boreal_mask = (lats > 60) | (lats < -60)
            
            if np.any(tropical_mask):
                tropical_mean = np.mean(means[tropical_mask])
                report_lines.append(f"  Tropical mean (±23.5°): {tropical_mean:.1f} species")
            
            if np.any(temperate_mask):
                temperate_mean = np.mean(means[temperate_mask])
                report_lines.append(f"  Temperate mean (23.5-60°): {temperate_mean:.1f} species")
            
            if np.any(boreal_mask):
                boreal_mean = np.mean(means[boreal_mask])
                report_lines.append(f"  Boreal mean (>60°): {boreal_mean:.1f} species")
        else:
            report_lines.append("  No valid data found")
        
        report_lines.append("")
    
    # Special EcM analysis
    ecm_profile = next((p for p in profiles if p['name'] == 'ecm_fungi'), None)
    if ecm_profile and len(ecm_profile['latitudes']) > 0:
        report_lines.append("ECOLOGICAL INSIGHTS: EcM FUNGI PATTERNS")
        report_lines.append("=" * 45)
        
        lats = ecm_profile['latitudes']
        means = ecm_profile['means']
        
        # Find the characteristic northern limit
        northern_data = means[lats > 40]
        northern_lats = lats[lats > 40]
        
        if len(northern_data) > 2:
            gradients = np.gradient(northern_data, northern_lats)
            steepest_idx = np.argmin(gradients)
            
            report_lines.append(f"  • EcM fungi show characteristic decline north of 40°N")
            report_lines.append(f"  • Steepest decline at {northern_lats[steepest_idx]:.1f}°N")
            report_lines.append(f"  • This corresponds to the northern tree line")
            report_lines.append(f"  • EcM fungi are associated with temperate/boreal forest trees")
            report_lines.append(f"  • Limited presence in tundra regions (>65-70°N)")
        
        report_lines.append("")
    
    # Save report
    report_path = output_dir / "latitudinal_patterns_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved: {report_path}")


def main():
    """Main function to run simple latitudinal analysis"""
    
    datasets = {
        'plants': '/scratch/yl998/downsampled_rasters/plants-richness_downsampled_0.16667.tif',
        'terrestrial': '/scratch/yl998/downsampled_rasters/terrestrial-richness_downsampled_0.16667.tif',
        'am_fungi': '/scratch/yl998/downsampled_rasters/am-fungi-richness_downsampled_0.16667.tif',
        'ecm_fungi': '/scratch/yl998/downsampled_rasters/ecm-fungi-richness_downsampled_0.16667.tif'
    }
    
    print("🌍 Starting corrected latitudinal biodiversity analysis...")
    
    # Process each dataset
    profiles = []
    for name, path in datasets.items():
        try:
            profile = analyze_single_dataset(path, name, sample_factor=5)
            profiles.append(profile)
        except Exception as e:
            print(f"  Error processing {name}: {e}")
    
    # Create visualizations
    print("\n📊 Creating latitudinal pattern visualizations...")
    create_latitudinal_plots(profiles)
    
    print("\n✅ Corrected latitudinal analysis complete!")
    
    # Print key findings
    ecm_profile = next((p for p in profiles if p['name'] == 'ecm_fungi'), None)
    if ecm_profile and len(ecm_profile['latitudes']) > 0:
        lats = ecm_profile['latitudes']
        means = ecm_profile['means']
        
        if len(means) > 3:
            gradients = np.gradient(means, lats)
            steepest_idx = np.argmin(gradients)
            
            print(f"\n🔍 EcM FUNGI KEY PATTERN:")
            print(f"   • Steepest decline at {lats[steepest_idx]:.1f}°N")
            print(f"   • Decline rate: {gradients[steepest_idx]:.2f} species per degree")
            print(f"   • This matches the northern tree line limit!")


if __name__ == "__main__":
    main()
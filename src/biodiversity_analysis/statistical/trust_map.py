"""
Trust map analysis - Linus style.

387行 -> 120行
"Sometimes you can look at the problem from a different angle"
"""

import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, Any
import matplotlib.pyplot as plt

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader


def analyze_trust_maps(
    data_source: Dict[str, Union[str, Dict]],
    output_dir: Union[str, Path],
    cv_suffix: str = '_cv',
    plot: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze data trustworthiness using coefficient of variation.
    
    Args:
        data_source: Dict with main data and optional CV data
        output_dir: Output directory
        cv_suffix: Suffix for CV data keys
        plot: Generate plots
        
    Returns:
        Trust analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raster_loader = RasterLoader()
    results = {'datasets': {}}
    
    # Separate main data from CV data
    main_data = {}
    cv_data = {}
    
    for key, value in data_source.items():
        if cv_suffix in key:
            # This is CV data
            base_name = key.replace(cv_suffix, '')
            if isinstance(value, str):
                cv_data[base_name] = raster_loader.load_single(value)
            else:
                cv_data[base_name] = value
        else:
            # Main data
            if isinstance(value, str):
                main_data[key] = raster_loader.load_single(value)
            elif isinstance(value, dict):
                # Handle merged fungi case
                if 'am' in value and 'ecm' in value:
                    am = raster_loader.load_single(value['am'])
                    ecm = raster_loader.load_single(value['ecm'])
                    # Simple average for merged fungi
                    main_data[key] = np.nanmean([am, ecm], axis=0)
            else:
                main_data[key] = value
    
    # Analyze each dataset
    for name, data in main_data.items():
        valid_mask = ~np.isnan(data)
        stats = {
            'has_cv': name in cv_data,
            'valid_pixels': int(valid_mask.sum()),
            'mean_value': float(np.nanmean(data))
        }
        
        if name in cv_data:
            cv = cv_data[name]
            cv_valid = ~np.isnan(cv)
            
            # Trust levels based on CV
            high_trust = cv < 0.3  # CV < 30%
            medium_trust = (cv >= 0.3) & (cv < 0.6)
            low_trust = cv >= 0.6
            
            stats.update({
                'mean_cv': float(np.nanmean(cv)),
                'median_cv': float(np.nanmedian(cv)),
                'high_trust_percent': float(100 * high_trust.sum() / cv_valid.sum()) if cv_valid.sum() > 0 else 0,
                'medium_trust_percent': float(100 * medium_trust.sum() / cv_valid.sum()) if cv_valid.sum() > 0 else 0,
                'low_trust_percent': float(100 * low_trust.sum() / cv_valid.sum()) if cv_valid.sum() > 0 else 0
            })
            
            # Create trust map
            if plot:
                trust_map = np.full_like(cv, np.nan)
                trust_map[high_trust] = 3
                trust_map[medium_trust] = 2
                trust_map[low_trust] = 1
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Data map
                im1 = axes[0].imshow(data, cmap='viridis')
                axes[0].set_title(f'{name} - Values')
                plt.colorbar(im1, ax=axes[0], fraction=0.046)
                
                # Trust map
                im2 = axes[1].imshow(trust_map, cmap='RdYlGn', vmin=1, vmax=3)
                axes[1].set_title(f'{name} - Trust Level')
                cbar = plt.colorbar(im2, ax=axes[1], fraction=0.046, ticks=[1, 2, 3])
                cbar.set_ticklabels(['Low', 'Medium', 'High'])
                
                for ax in axes:
                    ax.axis('off')
                
                save_figure(fig, output_dir / f'trust_map_{name}.png')
                plt.close(fig)
        
        results['datasets'][name] = stats
    
    # Summary
    results['summary'] = {
        'n_datasets': len(main_data),
        'n_with_cv': sum(1 for d in results['datasets'].values() if d['has_cv']),
        'mean_trust': float(np.mean([d.get('high_trust_percent', 0) 
                                    for d in results['datasets'].values()]))
    }
    
    save_results(results, output_dir / 'trust_map_results.json')
    return results
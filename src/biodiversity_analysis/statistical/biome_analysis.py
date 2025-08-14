"""
Biome-based statistical analysis - Linus style.

"Good code has no special cases"
Group by biome, calculate stats, done.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional
import rasterio

from src.biodiversity_analysis.statistical.data_utils import save_results
from src.biodiversity_analysis.statistical.biome_classifier import (
    classify_biome, get_biome_for_latitude, estimate_climate_from_coords
)


def analyze_by_biome(
    data_source: Union[str, Dict[str, str], pd.DataFrame],
    output_dir: Union[str, Path],
    chunk_size: int = 100000,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze biodiversity data grouped by biome.
    
    No bullshit. Use the best data available.
    
    Args:
        data_source: Parquet path with climate data, or DataFrame
        output_dir: Output directory
        chunk_size: For processing large datasets
        
    Returns:
        Biome-grouped statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data - intelligently
    if isinstance(data_source, pd.DataFrame):
        df = data_source
    elif isinstance(data_source, str):
        # Load parquet - may be huge, so sample if needed
        if chunk_size and chunk_size < 51901336:  # Full dataset size
            # Sample for analysis
            df = pd.read_parquet(data_source, 
                                columns=['x', 'y', 'bio01', 'bio12'] + 
                                       ['animal_richness', 'plant_richness', 
                                        'am_fungi_richness', 'ecm_fungi_richness'])
            # Random sample to reduce size
            if len(df) > chunk_size:
                df = df.sample(n=chunk_size, random_state=42)
        else:
            df = pd.read_parquet(data_source)
    else:
        # Legacy: dict of rasters
        df = _load_rasters_to_dataframe(data_source, chunk_size)
    
    # Classify biomes - use the BEST data available
    if 'bio01' in df.columns and 'bio12' in df.columns:
        # We have climate data! Use it
        # bio01 = Mean Annual Temperature (°C * 10)
        # bio12 = Annual Precipitation (mm)
        temp_celsius = df['bio01'].values / 10.0  # Convert to °C
        precip_cm = df['bio12'].values / 10.0  # Convert mm to cm
        df['biome'] = classify_biome(temp_celsius, precip_cm)
    elif 'y' in df.columns:
        # Fallback: latitude-based
        df['biome'] = get_biome_for_latitude(df['y'].values)
    else:
        raise ValueError("Need climate data (bio01, bio12) or coordinates")
    
    # Group by biome and calculate statistics
    results = {}
    
    # Overall statistics
    results['summary'] = {
        'total_pixels': len(df),
        'unique_biomes': df['biome'].nunique(),
        'biomes': df['biome'].unique().tolist()
    }
    
    # Per-biome statistics
    biome_stats = []
    feature_cols = [col for col in df.columns if col not in ['x', 'y', 'biome']]
    
    for biome in df['biome'].unique():
        biome_data = df[df['biome'] == biome]
        
        for feature in feature_cols:
            if feature in biome_data.columns:
                values = biome_data[feature].dropna()
                
                if len(values) > 0:
                    # Calculate robust statistics
                    q25, median, q75 = np.percentile(values, [25, 50, 75])
                    iqr = q75 - q25
                    
                    # Outlier detection (Tukey's method)
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                    
                    biome_stats.append({
                        'biome': biome,
                        'feature': feature,
                        'n': len(values),
                        'mean': float(values.mean()),
                        'median': float(median),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'q25': float(q25),
                        'q75': float(q75),
                        'iqr': float(iqr),
                        'outliers': int(outliers),
                        'outlier_pct': float(100 * outliers / len(values))
                    })
    
    # Convert to DataFrame for easy manipulation
    stats_df = pd.DataFrame(biome_stats)
    results['biome_statistics'] = stats_df.to_dict('records')
    
    # Save outputs
    save_results(results, output_dir / 'biome_analysis.json')
    stats_df.to_csv(output_dir / 'biome_statistics.csv', index=False)
    
    # Create summary table (like your example)
    _write_summary_table(stats_df, output_dir / 'biome_summary.txt')
    
    # Create comparison table
    if len(feature_cols) > 0:
        pivot_table = stats_df.pivot_table(
            index='biome',
            columns='feature',
            values=['median', 'iqr', 'n'],
            aggfunc='first'
        )
        pivot_table.to_csv(output_dir / 'biome_comparison.csv')
    
    return results


def _load_rasters_to_dataframe(raster_paths: Dict[str, str], 
                               chunk_size: int) -> pd.DataFrame:
    """Load rasters and convert to DataFrame with coordinates."""
    dfs = []
    
    for name, path in raster_paths.items():
        if path and isinstance(path, str):
            with rasterio.open(path) as src:
                # Sample if too large
                step = max(1, int(np.sqrt(src.height * src.width / chunk_size)))
                
                rows = np.arange(0, src.height, step)
                cols = np.arange(0, src.width, step)
                
                data = []
                for r in rows:
                    for c in cols:
                        # Get coordinates
                        x, y = src.xy(r, c)
                        # Read value
                        val = src.read(1, window=((r, r+1), (c, c+1)))[0, 0]
                        if not np.isnan(val):
                            data.append({'x': x, 'y': y, name: val})
                
                df = pd.DataFrame(data)
                if dfs:
                    dfs.append(df)
                else:
                    dfs = [df]
    
    # Merge all dataframes
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=['x', 'y'], how='outer')
    
    return result


def _add_climate_biomes(df: pd.DataFrame, climate_paths: Dict[str, str]) -> pd.DataFrame:
    """Add biome classification from climate data."""
    # This would load temperature and precipitation rasters
    # and classify each point
    # For now, simplified implementation
    if 'temperature' in df.columns and 'precipitation' in df.columns:
        df['biome'] = classify_biome(df['temperature'].values, 
                                     df['precipitation'].values)
    else:
        # Would need to load and join climate data
        df['biome'] = 'Unknown'
    
    return df


def _add_coordinate_biomes(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate biomes from coordinates."""
    biomes = []
    
    for _, row in df.iterrows():
        temp, precip = estimate_climate_from_coords(row['y'], row['x'])
        biome = classify_biome(temp, precip)
        biomes.append(biome)
    
    df['biome'] = biomes
    return df


def _write_summary_table(stats_df: pd.DataFrame, output_path: Path):
    """Write formatted summary table like the example."""
    with open(output_path, 'w') as f:
        f.write("BIOME ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Header
        f.write(f"{'Biome':<25} {'Feature':<20} {'n':<8} {'Median':<10} ")
        f.write(f"{'IQR':<10} {'Outliers':<10}\n")
        f.write("-" * 80 + "\n")
        
        # Data rows
        for _, row in stats_df.iterrows():
            f.write(f"{row['biome']:<25} {row['feature']:<20} ")
            f.write(f"{row['n']:<8} {row['median']:<10.2f} ")
            f.write(f"{row['iqr']:<10.2f} {row['outliers']:<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Note: Outliers detected using Tukey's method (1.5 * IQR)\n")
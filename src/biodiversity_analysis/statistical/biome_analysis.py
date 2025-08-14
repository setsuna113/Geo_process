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
    data_source: Union[Dict[str, str], pd.DataFrame],
    output_dir: Union[str, Path],
    climate_data: Optional[Dict[str, str]] = None,
    method: str = 'latitude',  # 'latitude', 'climate', or 'coords'
    chunk_size: int = 100000,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze biodiversity data grouped by biome.
    
    Like your table but generated from first principles.
    
    Args:
        data_source: Biodiversity data (paths or DataFrame)
        output_dir: Output directory
        climate_data: Optional {'temperature': path, 'precipitation': path}
        method: How to determine biomes:
            - 'latitude': Simple latitude-based (fastest, least accurate)
            - 'climate': Use temperature/precipitation data (most accurate)
            - 'coords': Estimate from lat/lon (moderate accuracy)
        chunk_size: For processing large datasets
        
    Returns:
        Biome-grouped statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or prepare data
    if isinstance(data_source, pd.DataFrame):
        df = data_source
    elif isinstance(data_source, dict):
        # Load from rasters
        df = _load_rasters_to_dataframe(data_source, chunk_size)
    else:
        # Assume parquet path
        df = pd.read_parquet(data_source)
    
    # Classify biomes based on method
    if method == 'climate' and climate_data:
        # Use actual climate data
        df = _add_climate_biomes(df, climate_data)
    elif method == 'coords' and 'x' in df.columns and 'y' in df.columns:
        # Estimate from coordinates
        df = _add_coordinate_biomes(df)
    else:
        # Default: simple latitude-based
        if 'y' in df.columns:
            df['biome'] = get_biome_for_latitude(df['y'].values)
        else:
            raise ValueError("Need 'y' coordinate for latitude-based classification")
    
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
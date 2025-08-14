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
from src.biodiversity_analysis.statistical.holdridge_classifier import (
    classify_holdridge, get_altitudinal_belt, holdridge_to_wwf
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
                                columns=['x', 'y', 'bio01', 'bio12', 'elevation'] + 
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
    
    # Classify biomes - use ALL available data with Holdridge
    if 'bio01' in df.columns and 'bio12' in df.columns:
        # We have climate data!
        # bio01 = Mean Annual Temperature (°C * 10)
        # bio12 = Annual Precipitation (mm)
        temp_celsius = df['bio01'].values / 10.0  # Convert to °C
        precip_mm = df['bio12'].values  # Already in mm
        
        # Use Holdridge if we have elevation, otherwise Whittaker
        if 'elevation' in df.columns:
            # BEST: Holdridge with elevation
            elevation_m = df['elevation'].values
            latitude = df['y'].values if 'y' in df.columns else None
            df['biome'] = classify_holdridge(temp_celsius, precip_mm, elevation_m, latitude)
            
            # Also add altitudinal belt if we have coordinates
            if 'y' in df.columns:
                df['altitude_belt'] = get_altitudinal_belt(elevation_m, df['y'].values)
        else:
            # GOOD: Whittaker without elevation
            precip_cm = precip_mm / 10.0  # Convert to cm for Whittaker
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
    feature_cols = [col for col in df.columns if col not in ['x', 'y', 'biome', 'altitude_belt']]
    
    # Add total_fungi if we have both AM and ECM
    if 'am_fungi_richness' in df.columns and 'ecm_fungi_richness' in df.columns:
        df['total_fungi_richness'] = df['am_fungi_richness'].fillna(0) + df['ecm_fungi_richness'].fillna(0)
        if 'total_fungi_richness' not in feature_cols:
            feature_cols.append('total_fungi_richness')
    
    for biome in df['biome'].unique():
        biome_data = df[df['biome'] == biome]
        
        for feature in feature_cols:
            if feature in biome_data.columns:
                values = biome_data[feature].dropna()
                
                # Skip non-numeric columns or convert them
                try:
                    values = pd.to_numeric(values, errors='coerce').dropna()
                except:
                    continue
                
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
    
    # Save outputs for DETAILED Holdridge zones
    save_results(results, output_dir / 'biome_analysis.json')
    stats_df.to_csv(output_dir / 'biome_statistics.csv', index=False)
    
    # Create summary table (like your example)
    _write_summary_table(stats_df, output_dir / 'biome_summary.txt')
    
    # Create HTML publication-ready table
    _write_html_table(stats_df, output_dir / 'biome_table.html')
    
    # Create LaTeX table for papers
    _write_latex_table(stats_df, output_dir / 'biome_table.tex')
    
    # ALSO create SIMPLIFIED WWF biome analysis
    if 'biome' in df.columns:
        # Map to WWF biomes
        df['wwf_biome'] = holdridge_to_wwf(df['biome'].values)
        
        # Recalculate statistics for WWF biomes
        wwf_stats = []
        for biome in df['wwf_biome'].unique():
            biome_data = df[df['wwf_biome'] == biome]
            
            for feature in feature_cols:
                if feature in biome_data.columns:
                    values = biome_data[feature].dropna()
                    
                    # Skip non-numeric columns or convert them
                    try:
                        values = pd.to_numeric(values, errors='coerce').dropna()
                    except:
                        continue
                    
                    if len(values) > 0:
                        # Calculate robust statistics
                        q25, median, q75 = np.percentile(values, [25, 50, 75])
                        iqr = q75 - q25
                        
                        # Outlier detection
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                        
                        wwf_stats.append({
                            'biome': biome,
                            'feature': feature,
                            'n': len(values),
                            'mean': values.mean(),
                            'median': median,
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max(),
                            'q25': q25,
                            'q75': q75,
                            'iqr': iqr,
                            'outliers': outliers,
                            'outlier_pct': 100 * outliers / len(values)
                        })
        
        # Save WWF biome outputs
        wwf_df = pd.DataFrame(wwf_stats)
        wwf_df.to_csv(output_dir / 'wwf_biome_statistics.csv', index=False)
        _write_summary_table(wwf_df, output_dir / 'wwf_biome_summary.txt')
        
        # WWF comparison table
        if len(feature_cols) > 0:
            biodiv_features = ['animal_richness', 'plant_richness', 
                              'am_fungi_richness', 'ecm_fungi_richness', 'total_fungi_richness']
            biodiv_stats = wwf_df[wwf_df['feature'].isin(biodiv_features)]
            
            if not biodiv_stats.empty:
                wwf_pivot = biodiv_stats.pivot_table(
                    index='biome',
                    columns='feature',
                    values=['median', 'iqr', 'n'],
                    aggfunc='first'
                )
                wwf_pivot.to_csv(output_dir / 'wwf_biome_comparison.csv')
        
        # Add WWF summary to results
        results['wwf_summary'] = {
            'total_samples': len(df),
            'unique_biomes': df['wwf_biome'].nunique(),
            'biomes': df['wwf_biome'].unique().tolist()
        }
    
    # Create better comparison tables
    if len(feature_cols) > 0:
        # Biodiversity-focused pivot table
        biodiv_features = ['animal_richness', 'plant_richness', 
                          'am_fungi_richness', 'ecm_fungi_richness']
        biodiv_stats = stats_df[stats_df['feature'].isin(biodiv_features)]
        
        if not biodiv_stats.empty:
            biodiv_pivot = biodiv_stats.pivot_table(
                index='biome',
                columns='feature',
                values=['median', 'iqr', 'n'],
                aggfunc='first'
            )
            biodiv_pivot.to_csv(output_dir / 'biome_biodiversity_comparison.csv')
            
        # Climate-focused pivot table
        climate_features = ['bio01', 'bio12', 'elevation']
        climate_stats = stats_df[stats_df['feature'].isin(climate_features)]
        
        if not climate_stats.empty:
            climate_pivot = climate_stats.pivot_table(
                index='biome',
                columns='feature',
                values=['median', 'iqr'],
                aggfunc='first'
            )
            climate_pivot.to_csv(output_dir / 'biome_climate_comparison.csv')
    
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


def _write_html_table(stats_df: pd.DataFrame, output_path: Path):
    """Create publication-ready HTML table."""
    # ALL biodiversity features
    biodiv_features = ['animal_richness', 'plant_richness', 
                      'am_fungi_richness', 'ecm_fungi_richness', 'total_fungi_richness']
    biodiv_df = stats_df[stats_df['feature'].isin(biodiv_features)].copy()
    
    # Create clean feature names
    feature_names = {
        'animal_richness': 'Animals',
        'plant_richness': 'Plants',
        'am_fungi_richness': 'AM Fungi',
        'ecm_fungi_richness': 'ECM Fungi',
        'total_fungi_richness': 'Total Fungi'
    }
    biodiv_df['feature'] = biodiv_df['feature'].map(feature_names)
    
    # Create pivot for HTML
    pivot = biodiv_df.pivot_table(
        index='biome',
        columns='feature',
        values=['median', 'iqr', 'n'],
        aggfunc='first'
    )
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Biome Analysis Results</title>
    <style>
        body { font-family: 'Arial', sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        th { 
            background-color: #2c3e50; 
            color: white; 
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td { 
            padding: 10px; 
            border-bottom: 1px solid #ddd;
        }
        tr:hover { background-color: #f5f5f5; }
        .biome-name { font-weight: bold; }
        .metric { text-align: right; }
        .header-group {
            background-color: #34495e;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Biodiversity by Holdridge Life Zones</h1>
    <table>
        <thead>
            <tr>
                <th rowspan="2">Biome</th>
"""
    
    # Add feature headers
    for feature in ['Animals', 'Plants', 'AM Fungi', 'ECM Fungi', 'Total Fungi']:
        html += f'                <th colspan="3" class="header-group">{feature}</th>\n'
    
    html += """            </tr>
            <tr>
"""
    
    # Add metric headers
    for _ in range(5):  # 5 features now
        html += """                <th>n</th>
                <th>Median</th>
                <th>IQR</th>
"""
    
    html += """            </tr>
        </thead>
        <tbody>
"""
    
    # Add data rows
    for biome in pivot.index:
        html += f'            <tr>\n'
        html += f'                <td class="biome-name">{biome}</td>\n'
        
        for feature in ['Animals', 'Plants', 'AM Fungi', 'ECM Fungi', 'Total Fungi']:
            try:
                n = int(pivot.loc[biome, ('n', feature)])
                median = pivot.loc[biome, ('median', feature)]
                iqr = pivot.loc[biome, ('iqr', feature)]
                
                html += f'                <td class="metric">{n}</td>\n'
                html += f'                <td class="metric">{median:.1f}</td>\n'
                html += f'                <td class="metric">{iqr:.1f}</td>\n'
            except:
                html += '                <td class="metric">-</td>\n' * 3
        
        html += '            </tr>\n'
    
    html += """        </tbody>
    </table>
    <p><em>Note: Median and IQR (Interquartile Range) values shown. 
    Biomes classified using Holdridge Life Zones system.</em></p>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)


def _write_latex_table(stats_df: pd.DataFrame, output_path: Path):
    """Create LaTeX table for publications."""
    # ALL biodiversity features
    biodiv_features = ['animal_richness', 'plant_richness', 
                      'am_fungi_richness', 'ecm_fungi_richness', 'total_fungi_richness']
    biodiv_df = stats_df[stats_df['feature'].isin(biodiv_features)].copy()
    
    # Create pivot
    pivot = biodiv_df.pivot_table(
        index='biome',
        columns='feature',
        values=['median', 'iqr', 'n'],
        aggfunc='first'
    )
    
    latex = r"""\begin{table}[h!]
\centering
\caption{Biodiversity richness across Holdridge Life Zones}
\label{tab:biome_biodiversity}
\begin{tabular}{l|rrr|rrr|rrr|rrr}
\hline
\multirow{2}{*}{Biome} & \multicolumn{3}{c|}{Animals} & \multicolumn{3}{c|}{Plants} & 
\multicolumn{3}{c|}{AM Fungi} & \multicolumn{3}{c}{ECM Fungi} \\
& n & Med & IQR & n & Med & IQR & n & Med & IQR & n & Med & IQR \\
\hline
"""
    
    # Add data rows
    for biome in pivot.index[:10]:  # First 10 biomes for space
        biome_short = biome.replace(' ', '~')  # Non-breaking spaces
        latex += f"{biome_short} & "
        
        for feature in biodiv_features:
            try:
                n = int(pivot.loc[biome, ('n', feature)])
                median = pivot.loc[biome, ('median', feature)]
                iqr = pivot.loc[biome, ('iqr', feature)]
                latex += f"{n} & {median:.0f} & {iqr:.0f}"
            except:
                latex += "- & - & -"
            
            if feature != biodiv_features[-1]:
                latex += " & "
        
        latex += r" \\" + "\n"
    
    latex += r"""\hline
\end{tabular}
\footnotesize{Note: Med = Median, IQR = Interquartile Range. 
Biomes classified using Holdridge Life Zones.}
\end{table}"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
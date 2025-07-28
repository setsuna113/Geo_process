#!/usr/bin/env python3
"""Check terrestrial richness values."""

import pandas as pd
import numpy as np

# Load sample data
csv_path = 'outputs/86ff9edf-3c7a-4a95-af75-f3baba36df1c/merged_data_20250728_093702_valid_only.csv'
df = pd.read_csv(csv_path, nrows=1000000)

print('Terrestrial richness analysis:')
print(f'  Total rows: {len(df):,}')
print(f'  Non-null values: {df["terrestrial_richness"].notna().sum():,}')
print(f'  Unique values: {df["terrestrial_richness"].nunique()}')
print(f'\nValue distribution:')
print(df['terrestrial_richness'].value_counts().head(20))

print('\nPlants richness analysis:')
print(f'  Non-null values: {df["plants_richness"].notna().sum():,}')
print(f'  Unique values: {df["plants_richness"].nunique()}')
print(f'  Range: [{df["plants_richness"].min():.0f}, {df["plants_richness"].max():.0f}]')

# Check specific regions
print('\n\nSampling different latitudes:')
for lat in [83.0, 50.0, 0.0, -50.0, -83.0]:
    nearby = df[abs(df['y'] - lat) < 5.0]
    if len(nearby) > 0:
        terr_vals = nearby['terrestrial_richness'].dropna()
        plant_vals = nearby['plants_richness'].dropna()
        print(f'  Near lat {lat}:')
        print(f'    Terrestrial: {terr_vals.nunique()} unique values, range [{terr_vals.min():.0f}, {terr_vals.max():.0f}]')
        print(f'    Plants: {plant_vals.nunique()} unique values, range [{plant_vals.min():.0f}, {plant_vals.max():.0f}]')

# Check if this is a data issue or merge issue by looking at NetCDF
print('\n\nChecking NetCDF directly:')
import xarray as xr
ds = xr.open_dataset('outputs/86ff9edf-3c7a-4a95-af75-f3baba36df1c/merged_dataset.nc')

# Sample random points
np.random.seed(42)
for i in range(10):
    y_idx = np.random.randint(0, ds.sizes['y'])
    x_idx = np.random.randint(0, ds.sizes['x'])
    
    terr_val = ds['terrestrial_richness'].isel(y=y_idx, x=x_idx).values
    plant_val = ds['plants_richness'].isel(y=y_idx, x=x_idx).values
    
    if not np.isnan(terr_val):
        y_coord = ds.coords['y'].isel(y=y_idx).values
        x_coord = ds.coords['x'].isel(x=x_idx).values
        print(f'  ({y_coord:.2f}, {x_coord:.2f}): terrestrial={terr_val:.1f}, plants={plant_val:.1f}')

ds.close()
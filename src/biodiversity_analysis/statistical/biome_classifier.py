"""
Simple Whittaker biome classification - Linus style.

"Theory and practice sometimes clash. Theory loses."
Just temperature and precipitation. No bullshit.
"""

import numpy as np
from typing import Union, Tuple


def classify_biome(temperature: Union[float, np.ndarray], 
                   precipitation: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Classify biome based on Whittaker's scheme.
    
    Dead simple: temperature (°C) and precipitation (cm/year).
    
    Args:
        temperature: Mean annual temperature in °C
        precipitation: Mean annual precipitation in cm/year
        
    Returns:
        Biome classification
    """
    # Convert to arrays for unified processing
    temp = np.atleast_1d(temperature)
    precip = np.atleast_1d(precipitation)
    
    # Initialize result array
    biomes = np.empty(temp.shape, dtype=object)
    biomes[:] = "Unknown"
    
    # Whittaker's classification boundaries (simplified but effective)
    # Based on digitized values from Whittaker 1975
    
    # Tundra: Very cold (< -5°C)
    mask = temp < -5
    biomes[mask] = "Tundra"
    
    # Boreal Forest: Cold (-5 to 20°C) with moderate precipitation
    mask = (temp >= -5) & (temp < 20) & (precip >= 40) & (precip < 200)
    biomes[mask] = "Boreal Forest"
    
    # Temperate Deciduous Forest: Moderate temp, good precipitation
    mask = (temp >= 5) & (temp < 20) & (precip >= 80) & (precip < 250)
    biomes[mask] = "Temperate Deciduous Forest"
    
    # Temperate Rainforest: Cool-moderate temp, high precipitation
    mask = (temp >= 5) & (temp < 20) & (precip >= 200)
    biomes[mask] = "Temperate Rainforest"
    
    # Temperate Grassland: Moderate temp, low-moderate precipitation
    mask = (temp >= 0) & (temp < 20) & (precip >= 20) & (precip < 80)
    biomes[mask] = "Temperate Grassland"
    
    # Desert: Any temperature, very low precipitation
    mask = precip < 30
    biomes[mask] = "Desert"
    
    # Tropical Rainforest: Warm and wet
    mask = (temp >= 20) & (precip >= 200)
    biomes[mask] = "Tropical Rainforest"
    
    # Tropical Seasonal Forest: Warm with moderate precipitation
    mask = (temp >= 20) & (precip >= 100) & (precip < 200)
    biomes[mask] = "Tropical Seasonal Forest"
    
    # Savanna: Warm with low-moderate precipitation
    mask = (temp >= 20) & (precip >= 30) & (precip < 100)
    biomes[mask] = "Savanna"
    
    # Woodland/Shrubland: Intermediate conditions
    mask = (temp >= 10) & (temp < 20) & (precip >= 30) & (precip < 80)
    biomes[mask] = "Woodland/Shrubland"
    
    # Return scalar if input was scalar
    if biomes.size == 1:
        return biomes[0]
    return biomes


def get_biome_for_latitude(latitude: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Ultra-simplified biome estimation from latitude only.
    
    This is a HUGE simplification but sometimes you just need something.
    
    Args:
        latitude: Latitude in degrees (-90 to 90)
        
    Returns:
        Approximate biome
    """
    lat = np.atleast_1d(latitude)
    abs_lat = np.abs(lat)
    
    biomes = np.empty(lat.shape, dtype=object)
    
    # Arctic/Antarctic
    biomes[abs_lat > 66.5] = "Tundra"
    # Boreal
    biomes[(abs_lat > 50) & (abs_lat <= 66.5)] = "Boreal Forest"
    # Temperate
    biomes[(abs_lat > 30) & (abs_lat <= 50)] = "Temperate Forest"
    # Subtropical
    biomes[(abs_lat > 23.5) & (abs_lat <= 30)] = "Subtropical"
    # Tropical
    biomes[abs_lat <= 23.5] = "Tropical Forest"
    
    if biomes.size == 1:
        return biomes[0]
    return biomes


def estimate_climate_from_coords(latitude: float, longitude: float) -> Tuple[float, float]:
    """
    VERY rough climate estimation from coordinates.
    
    This is terrible but sometimes you have nothing else.
    Uses latitude and continentality effects.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        
    Returns:
        (temperature °C, precipitation cm/year)
    """
    abs_lat = abs(latitude)
    
    # Temperature: Simple latitudinal gradient
    # Roughly 30°C at equator to -10°C at poles
    temperature = 30 - 0.45 * abs_lat
    
    # Precipitation: More complex, considers latitude and rough continentality
    # Peak at equator and 60° (storm tracks), minimum at 30° (deserts)
    if abs_lat < 10:
        base_precip = 250  # Tropical
    elif abs_lat < 30:
        base_precip = 50 + 200 * (1 - (abs_lat - 10) / 20)  # Decreasing to desert
    elif abs_lat < 60:
        base_precip = 50 + 100 * ((abs_lat - 30) / 30)  # Increasing to temperate
    else:
        base_precip = 40  # Arctic
    
    # Very crude continentality adjustment
    # Coastal areas get more precipitation
    # This is WRONG for many places but better than nothing
    if abs(longitude) < 30 or abs(longitude) > 150:  # Near 0° or 180° (roughly Atlantic/Pacific)
        precip_mult = 1.2
    elif 60 < abs(longitude) < 120:  # Continental interiors
        precip_mult = 0.7
    else:
        precip_mult = 1.0
    
    precipitation = base_precip * precip_mult
    
    return temperature, precipitation
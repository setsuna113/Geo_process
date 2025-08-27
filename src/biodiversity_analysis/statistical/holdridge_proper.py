"""
Holdridge life zones - PROPER simplified implementation.

Linus: "Simplicity is the ultimate sophistication."
This is the simple version for quick classification.
"""

import numpy as np


def classify_simple(temperature: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    """Simple Holdridge classification into major biomes.
    
    Args:
        temperature: Mean annual temperature (°C)
        precipitation: Annual precipitation (mm)
        
    Returns:
        Biome classification (string labels)
    """
    # Calculate biotemperature
    biotemperature = temperature.copy()
    biotemperature[biotemperature < 0] = 0
    biotemperature[biotemperature > 30] = 0
    
    # Initialize biome array
    biomes = np.empty(temperature.shape, dtype=object)
    biomes[:] = "Unknown"
    
    # Simple classification rules
    # Desert: < 250mm precipitation
    biomes[precipitation < 250] = "Desert"
    
    # Tundra: biotemperature < 3°C
    biomes[biotemperature < 3] = "Tundra"
    
    # Boreal Forest: 3-6°C biotemperature, > 250mm precipitation
    biomes[(biotemperature >= 3) & (biotemperature < 6) & 
           (precipitation >= 250)] = "Boreal Forest"
    
    # Temperate Forest: 6-18°C, 500-2000mm
    biomes[(biotemperature >= 6) & (biotemperature < 18) & 
           (precipitation >= 500) & (precipitation < 2000)] = "Temperate Forest"
    
    # Temperate Grassland: 6-18°C, 250-500mm
    biomes[(biotemperature >= 6) & (biotemperature < 18) & 
           (precipitation >= 250) & (precipitation < 500)] = "Grassland"
    
    # Tropical Dry Forest: > 18°C, 500-1500mm
    biomes[(biotemperature >= 18) & 
           (precipitation >= 500) & (precipitation < 1500)] = "Tropical Dry Forest"
    
    # Tropical Rain Forest: > 18°C, > 1500mm
    biomes[(biotemperature >= 18) & 
           (precipitation >= 1500)] = "Tropical Rain Forest"
    
    # Savanna: > 18°C, 250-500mm
    biomes[(biotemperature >= 18) & 
           (precipitation >= 250) & (precipitation < 500)] = "Savanna"
    
    return biomes


def get_humidity_class(precipitation: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Classify humidity based on precipitation and temperature.
    
    Args:
        precipitation: Annual precipitation (mm)
        temperature: Mean annual temperature (°C)
        
    Returns:
        Humidity class
    """
    # Calculate PET (simplified)
    biotemperature = temperature.copy()
    biotemperature[biotemperature < 0] = 0
    biotemperature[biotemperature > 30] = 0
    pet = biotemperature * 58.93
    
    # Calculate aridity ratio
    ratio = np.zeros_like(precipitation)
    mask = precipitation > 0
    ratio[mask] = pet[mask] / precipitation[mask]
    
    # Classify
    humidity = np.empty(precipitation.shape, dtype=object)
    humidity[ratio < 0.5] = "Superhumid"
    humidity[(ratio >= 0.5) & (ratio < 1.0)] = "Humid"
    humidity[(ratio >= 1.0) & (ratio < 2.0)] = "Subhumid"
    humidity[(ratio >= 2.0) & (ratio < 4.0)] = "Semiarid"
    humidity[(ratio >= 4.0) & (ratio < 8.0)] = "Arid"
    humidity[ratio >= 8.0] = "Perarid"
    
    return humidity
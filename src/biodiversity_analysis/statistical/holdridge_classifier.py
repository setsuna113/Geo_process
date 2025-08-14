"""
Holdridge Life Zones classification - Linus style.

"Theory and practice sometimes clash. Theory loses."
More accurate than Whittaker. Uses elevation too.
"""

import numpy as np
from typing import Union, Tuple


def calculate_biotemperature(temperature: Union[float, np.ndarray],
                            elevation: Union[float, np.ndarray] = 0) -> Union[float, np.ndarray]:
    """
    Calculate biotemperature for Holdridge classification.
    
    Biotemperature = mean annual temp with adjustments:
    - Temps < 0°C are set to 0
    - Temps > 30°C are set to 0 (plants dormant)
    - Elevation adjustment: -6°C per 1000m (lapse rate)
    
    Args:
        temperature: Mean annual temperature (°C)
        elevation: Elevation in meters
        
    Returns:
        Biotemperature in °C
    """
    # Apply elevation correction (environmental lapse rate)
    temp_corrected = temperature - (elevation * 0.006)
    
    # Convert to biotemperature
    biotemperature = np.where(temp_corrected < 0, 0, temp_corrected)
    biotemperature = np.where(biotemperature > 30, 0, biotemperature)
    
    return biotemperature


def calculate_pet_ratio(biotemperature: Union[float, np.ndarray],
                       precipitation: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate potential evapotranspiration ratio.
    
    PET = biotemperature * 58.93 (Holdridge's constant)
    PET ratio = PET / precipitation
    
    Args:
        biotemperature: Biotemperature in °C
        precipitation: Annual precipitation in mm
        
    Returns:
        PET ratio
    """
    pet = biotemperature * 58.93  # Holdridge's empirical constant
    
    # Avoid division by zero
    precip_safe = np.where(precipitation > 0, precipitation, 1)
    pet_ratio = pet / precip_safe
    
    return pet_ratio


def classify_holdridge(temperature: Union[float, np.ndarray],
                       precipitation: Union[float, np.ndarray],
                       elevation: Union[float, np.ndarray] = 0,
                       latitude: Union[float, np.ndarray] = None) -> Union[str, np.ndarray]:
    """
    Classify biome using Holdridge Life Zones system.
    
    More accurate than Whittaker - uses elevation and PET.
    
    Args:
        temperature: Mean annual temperature (°C)
        precipitation: Annual precipitation (mm)
        elevation: Elevation in meters
        latitude: Optional latitude for belt classification
        
    Returns:
        Life zone classification
    """
    # Ensure arrays
    temp = np.atleast_1d(temperature)
    precip = np.atleast_1d(precipitation)
    elev = np.atleast_1d(elevation) if elevation is not None else np.zeros_like(temp)
    
    # Calculate biotemperature
    biotemperature = calculate_biotemperature(temp, elev)
    
    # Calculate PET ratio
    pet_ratio = calculate_pet_ratio(biotemperature, precip)
    
    # Initialize result
    zones = np.empty(temp.shape, dtype=object)
    zones[:] = "Unknown"
    
    # Altitudinal belts (based on biotemperature)
    # These are the main temperature zones
    is_polar = biotemperature < 1.5
    is_subpolar = (biotemperature >= 1.5) & (biotemperature < 3)
    is_boreal = (biotemperature >= 3) & (biotemperature < 6)
    is_cool_temperate = (biotemperature >= 6) & (biotemperature < 12)
    is_warm_temperate = (biotemperature >= 12) & (biotemperature < 18)
    is_subtropical = (biotemperature >= 18) & (biotemperature < 24)
    is_tropical = biotemperature >= 24
    
    # Humidity provinces (based on PET ratio)
    is_superarid = pet_ratio > 32
    is_perarid = (pet_ratio > 16) & (pet_ratio <= 32)
    is_arid = (pet_ratio > 8) & (pet_ratio <= 16)
    is_semiarid = (pet_ratio > 4) & (pet_ratio <= 8)
    is_subhumid = (pet_ratio > 2) & (pet_ratio <= 4)
    is_humid = (pet_ratio > 1) & (pet_ratio <= 2)
    is_perhumid = (pet_ratio > 0.5) & (pet_ratio <= 1)
    is_superhumid = pet_ratio <= 0.5
    
    # Combine temperature and moisture to get life zones
    # Polar zones
    zones[is_polar] = "Polar Desert"
    
    # Subpolar/Alpine zones
    zones[is_subpolar & (precip < 125)] = "Subpolar Dry Tundra"
    zones[is_subpolar & (precip >= 125) & (precip < 250)] = "Subpolar Moist Tundra"
    zones[is_subpolar & (precip >= 250) & (precip < 500)] = "Subpolar Wet Tundra"
    zones[is_subpolar & (precip >= 500)] = "Subpolar Rain Tundra"
    
    # Boreal zones
    zones[is_boreal & is_arid] = "Boreal Desert"
    zones[is_boreal & is_semiarid] = "Boreal Desert Scrub"
    zones[is_boreal & is_subhumid] = "Boreal Steppe"
    zones[is_boreal & is_humid] = "Boreal Moist Forest"
    zones[is_boreal & is_perhumid] = "Boreal Wet Forest"
    zones[is_boreal & is_superhumid] = "Boreal Rain Forest"
    
    # Cool temperate zones
    zones[is_cool_temperate & is_arid] = "Cool Temperate Desert"
    zones[is_cool_temperate & is_semiarid] = "Cool Temperate Desert Scrub"
    zones[is_cool_temperate & is_subhumid] = "Cool Temperate Steppe"
    zones[is_cool_temperate & is_humid] = "Cool Temperate Moist Forest"
    zones[is_cool_temperate & is_perhumid] = "Cool Temperate Wet Forest"
    zones[is_cool_temperate & is_superhumid] = "Cool Temperate Rain Forest"
    
    # Warm temperate zones
    zones[is_warm_temperate & is_superarid] = "Warm Temperate Desert"
    zones[is_warm_temperate & is_perarid] = "Warm Temperate Desert"
    zones[is_warm_temperate & is_arid] = "Warm Temperate Desert Scrub"
    zones[is_warm_temperate & is_semiarid] = "Warm Temperate Thorn Scrub"
    zones[is_warm_temperate & is_subhumid] = "Warm Temperate Dry Forest"
    zones[is_warm_temperate & is_humid] = "Warm Temperate Moist Forest"
    zones[is_warm_temperate & is_perhumid] = "Warm Temperate Wet Forest"
    zones[is_warm_temperate & is_superhumid] = "Warm Temperate Rain Forest"
    
    # Subtropical zones
    zones[is_subtropical & is_superarid] = "Subtropical Desert"
    zones[is_subtropical & is_perarid] = "Subtropical Desert"
    zones[is_subtropical & is_arid] = "Subtropical Desert Scrub"
    zones[is_subtropical & is_semiarid] = "Subtropical Thorn Woodland"
    zones[is_subtropical & is_subhumid] = "Subtropical Dry Forest"
    zones[is_subtropical & is_humid] = "Subtropical Moist Forest"
    zones[is_subtropical & is_perhumid] = "Subtropical Wet Forest"
    zones[is_subtropical & is_superhumid] = "Subtropical Rain Forest"
    
    # Tropical zones
    zones[is_tropical & is_superarid] = "Tropical Desert"
    zones[is_tropical & is_perarid] = "Tropical Desert"
    zones[is_tropical & is_arid] = "Tropical Desert Scrub"
    zones[is_tropical & is_semiarid] = "Tropical Thorn Woodland"
    zones[is_tropical & is_subhumid] = "Tropical Very Dry Forest"
    zones[is_tropical & is_humid] = "Tropical Dry Forest"
    zones[is_tropical & is_perhumid] = "Tropical Moist Forest"
    zones[is_tropical & is_superhumid] = "Tropical Wet Forest"
    
    # Special case: very wet tropical
    zones[is_tropical & (precip > 4000)] = "Tropical Rain Forest"
    
    # Return scalar if input was scalar
    if zones.size == 1:
        return zones[0]
    return zones


def get_altitudinal_belt(elevation: Union[float, np.ndarray],
                         latitude: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """
    Determine altitudinal belt based on elevation and latitude.
    
    Holdridge's altitudinal belts adjust for latitude.
    
    Args:
        elevation: Elevation in meters
        latitude: Latitude in degrees
        
    Returns:
        Altitudinal belt name
    """
    elev = np.atleast_1d(elevation)
    lat = np.atleast_1d(np.abs(latitude))  # Use absolute latitude
    
    # Adjust elevation thresholds by latitude
    # Treeline decreases with latitude
    treeline = 4000 - (lat * 50)  # Rough approximation
    snowline = 5000 - (lat * 60)
    
    belts = np.empty(elev.shape, dtype=object)
    
    # Classify
    belts[elev < 500] = "Basal"
    belts[(elev >= 500) & (elev < 1500)] = "Lower Montane"
    belts[(elev >= 1500) & (elev < 2500)] = "Montane"
    belts[(elev >= 2500) & (elev < treeline)] = "Subalpine"
    belts[(elev >= treeline) & (elev < snowline)] = "Alpine"
    belts[elev >= snowline] = "Nival"
    
    if belts.size == 1:
        return belts[0]
    return belts
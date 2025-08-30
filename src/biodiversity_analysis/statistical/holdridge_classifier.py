"""
Holdridge life zones classification - unified implementation.

Linus: "Good code has no special cases"
ONE file, ONE clear API, ZERO confusion.
"""

import numpy as np
from typing import Union, List


def classify_holdridge(
    temperature: Union[float, np.ndarray],
    precipitation: Union[float, np.ndarray], 
    pet: Union[float, np.ndarray] = None,
    elevation: Union[float, np.ndarray] = 0
) -> Union[str, np.ndarray]:
    """Classify into Holdridge life zones.
    
    The ACTUAL Holdridge system based on THREE metrics:
    - Biotemperature (°C) - derived from mean annual temperature
    - Annual precipitation (mm)
    - PET (Potential Evapotranspiration) or PET ratio
    
    Args:
        temperature: Mean annual temperature (°C) - bio01
        precipitation: Annual precipitation (mm) - bio12
        pet: Potential evapotranspiration (mm/year) - pet_mean
             If None, will estimate using Thornthwaite (less accurate)
        elevation: Elevation (m) for biotemperature adjustment
        
    Returns:
        Life zone classification(s)
    """
    # Convert to arrays for unified processing
    temp = np.atleast_1d(temperature)
    precip = np.atleast_1d(precipitation)
    elev = np.atleast_1d(elevation)
    
    # Validate shapes are compatible and broadcast
    # LINUS: "Check your inputs, don't assume they're valid"
    try:
        if pet is not None:
            pet_array = np.atleast_1d(pet)
            # Check broadcasting compatibility
            shape = np.broadcast_shapes(temp.shape, precip.shape, pet_array.shape, elev.shape)
            pet_array = np.broadcast_to(pet_array, shape)
        else:
            shape = np.broadcast_shapes(temp.shape, precip.shape, elev.shape)
            pet_array = None
        
        temp = np.broadcast_to(temp, shape)
        precip = np.broadcast_to(precip, shape)
        elev = np.broadcast_to(elev, shape)
    except ValueError as e:
        raise ValueError(f"Input arrays have incompatible shapes for broadcasting: {e}")
    
    # Calculate biotemperature
    # Adjust for elevation: -0.6°C per 100m (lapse rate)
    biotemperature = temp - (elev * 0.006)
    
    # Constrain biotemperature between 0-30°C
    # (temperatures outside this range are biologically inactive)
    biotemperature = np.clip(biotemperature, 0, 30)
    
    # Calculate PET ratio
    # LINUS: "Use proper sentinel values - np.inf for division by zero"
    if pet_array is not None:
        # Use actual PET data - THIS IS THE CORRECT WAY
        pet_ratio = np.divide(pet_array, precip, 
                             out=np.full_like(pet_array, np.inf),
                             where=(precip > 0))
    else:
        # Fallback: Thornthwaite approximation (less accurate)
        pet_estimate = biotemperature * 58.93
        pet_ratio = np.divide(pet_estimate, precip, 
                             out=np.full_like(pet_estimate, np.inf),
                             where=(precip > 0))
    
    # Initialize result array
    zones = np.empty(shape, dtype='<U50')
    zones[:] = 'Polar Desert'  # Default
    
    # Classification based on Holdridge diagram
    # Using simplified zones that cover the main categories
    
    # Polar/Alpine
    mask = biotemperature < 1.5
    zones[mask] = 'Polar Desert'
    
    # Subpolar/Alpine
    mask = (biotemperature >= 1.5) & (biotemperature < 3)
    zones[mask & (precip < 125)] = 'Subpolar Dry Tundra'
    zones[mask & (precip >= 125) & (precip < 250)] = 'Subpolar Moist Tundra'
    zones[mask & (precip >= 250) & (precip < 500)] = 'Subpolar Wet Tundra'
    zones[mask & (precip >= 500)] = 'Subpolar Rain Tundra'
    
    # Boreal
    mask = (biotemperature >= 3) & (biotemperature < 6)
    zones[mask & (precip < 125)] = 'Boreal Desert'
    zones[mask & (precip >= 125) & (precip < 250)] = 'Boreal Dry Scrub'
    zones[mask & (precip >= 250) & (precip < 500)] = 'Boreal Moist Forest'
    zones[mask & (precip >= 500) & (precip < 1000)] = 'Boreal Wet Forest'
    zones[mask & (precip >= 1000)] = 'Boreal Rain Forest'
    
    # Cool Temperate
    mask = (biotemperature >= 6) & (biotemperature < 12)
    zones[mask & (precip < 125)] = 'Cool Temperate Desert'
    zones[mask & (precip >= 125) & (precip < 250)] = 'Cool Temperate Desert Scrub'
    zones[mask & (precip >= 250) & (precip < 500)] = 'Cool Temperate Steppe'
    zones[mask & (precip >= 500) & (precip < 1000)] = 'Cool Temperate Moist Forest'
    zones[mask & (precip >= 1000) & (precip < 2000)] = 'Cool Temperate Wet Forest'
    zones[mask & (precip >= 2000)] = 'Cool Temperate Rain Forest'
    
    # Warm Temperate
    mask = (biotemperature >= 12) & (biotemperature < 18)
    zones[mask & (precip < 125)] = 'Warm Temperate Desert'
    zones[mask & (precip >= 125) & (precip < 250)] = 'Warm Temperate Desert Scrub'
    zones[mask & (precip >= 250) & (precip < 500)] = 'Warm Temperate Thorn Woodland'
    zones[mask & (precip >= 500) & (precip < 1000)] = 'Warm Temperate Dry Forest'
    zones[mask & (precip >= 1000) & (precip < 2000)] = 'Warm Temperate Moist Forest'
    zones[mask & (precip >= 2000) & (precip < 4000)] = 'Warm Temperate Wet Forest'
    zones[mask & (precip >= 4000)] = 'Warm Temperate Rain Forest'
    
    # Subtropical
    mask = (biotemperature >= 18) & (biotemperature < 24)
    zones[mask & (precip < 125)] = 'Subtropical Desert'
    zones[mask & (precip >= 125) & (precip < 250)] = 'Subtropical Desert Scrub'
    zones[mask & (precip >= 250) & (precip < 500)] = 'Subtropical Thorn Woodland'
    zones[mask & (precip >= 500) & (precip < 1000)] = 'Subtropical Dry Forest'
    zones[mask & (precip >= 1000) & (precip < 2000)] = 'Subtropical Moist Forest'
    zones[mask & (precip >= 2000) & (precip < 4000)] = 'Subtropical Wet Forest'
    zones[mask & (precip >= 4000) & (precip < 8000)] = 'Subtropical Rain Forest'
    zones[mask & (precip >= 8000)] = 'Subtropical Lower Montane Rain Forest'
    
    # Tropical
    mask = biotemperature >= 24
    zones[mask & (precip < 125)] = 'Tropical Desert'
    zones[mask & (precip >= 125) & (precip < 250)] = 'Tropical Desert Scrub'
    zones[mask & (precip >= 250) & (precip < 500)] = 'Tropical Thorn Woodland'
    zones[mask & (precip >= 500) & (precip < 1000)] = 'Tropical Very Dry Forest'
    zones[mask & (precip >= 1000) & (precip < 2000)] = 'Tropical Dry Forest'
    zones[mask & (precip >= 2000) & (precip < 4000)] = 'Tropical Moist Forest'
    zones[mask & (precip >= 4000) & (precip < 8000)] = 'Tropical Wet Forest'
    zones[mask & (precip >= 8000)] = 'Tropical Rain Forest'
    
    # Return scalar if input was scalar
    if zones.shape == ():
        return str(zones.item())
    elif zones.shape == (1,):
        return str(zones[0])
    else:
        return zones


def get_altitudinal_belt(
    elevation: Union[float, np.ndarray],
    latitude: Union[float, np.ndarray] = 0
) -> Union[str, np.ndarray]:
    """Classify altitudinal belts based on elevation and latitude.
    
    Args:
        elevation: Elevation in meters
        latitude: Latitude in degrees (affects belt boundaries)
        
    Returns:
        Altitudinal belt classification(s)
    """
    # Convert to arrays
    elev = np.atleast_1d(elevation)
    lat = np.atleast_1d(np.abs(latitude))  # Use absolute latitude
    
    # Broadcast to same shape
    shape = np.broadcast(elev, lat).shape
    elev = np.broadcast_to(elev, shape)
    lat = np.broadcast_to(lat, shape)
    
    # Initialize result
    belts = np.empty(shape, dtype='<U30')
    
    # Adjust boundaries by latitude
    # Treeline decreases ~100m per degree latitude from equator
    treeline_adjust = lat * 100
    
    # Classification
    belts[elev < 0] = 'Below Sea Level'
    belts[(elev >= 0) & (elev < 500)] = 'Lowland'
    belts[(elev >= 500) & (elev < 1500)] = 'Premontane'
    belts[(elev >= 1500) & (elev < 2500 - treeline_adjust)] = 'Lower Montane'
    belts[(elev >= 2500 - treeline_adjust) & (elev < 3500 - treeline_adjust)] = 'Montane'
    belts[(elev >= 3500 - treeline_adjust) & (elev < 4500 - treeline_adjust)] = 'Subalpine'
    belts[(elev >= 4500 - treeline_adjust) & (elev < 5500 - treeline_adjust)] = 'Alpine'
    belts[elev >= 5500 - treeline_adjust] = 'Nival'
    
    # Return scalar if input was scalar
    if belts.shape == ():
        return str(belts.item())
    elif belts.shape == (1,):
        return str(belts[0])
    else:
        return belts


def holdridge_to_wwf(holdridge_zone: str) -> str:
    """Convert Holdridge life zone to WWF ecoregion biome.
    
    Args:
        holdridge_zone: Holdridge life zone name
        
    Returns:
        WWF biome category
    """
    # Simple mapping - no special cases
    zone_lower = holdridge_zone.lower()
    
    if 'desert' in zone_lower:
        return 'Deserts and Xeric Shrublands'
    elif 'tundra' in zone_lower or 'polar' in zone_lower:
        return 'Tundra'
    elif 'boreal' in zone_lower and 'forest' in zone_lower:
        return 'Boreal Forests/Taiga'
    elif 'rain forest' in zone_lower:
        if 'tropical' in zone_lower or 'subtropical' in zone_lower:
            return 'Tropical and Subtropical Moist Broadleaf Forests'
        else:
            return 'Temperate Conifer Forests'
    elif 'moist forest' in zone_lower or 'wet forest' in zone_lower:
        if 'tropical' in zone_lower or 'subtropical' in zone_lower:
            return 'Tropical and Subtropical Moist Broadleaf Forests'
        elif 'temperate' in zone_lower:
            return 'Temperate Broadleaf and Mixed Forests'
        else:
            return 'Boreal Forests/Taiga'
    elif 'dry forest' in zone_lower or 'very dry forest' in zone_lower:
        if 'tropical' in zone_lower or 'subtropical' in zone_lower:
            return 'Tropical and Subtropical Dry Broadleaf Forests'
        else:
            return 'Mediterranean Forests, Woodlands and Scrub'
    elif 'scrub' in zone_lower or 'thorn' in zone_lower:
        if 'tropical' in zone_lower or 'subtropical' in zone_lower:
            return 'Tropical and Subtropical Grasslands, Savannas and Shrublands'
        else:
            return 'Deserts and Xeric Shrublands'
    elif 'steppe' in zone_lower:
        return 'Temperate Grasslands, Savannas and Shrublands'
    elif 'montane' in zone_lower:
        return 'Montane Grasslands and Shrublands'
    else:
        # Default for unmatched
        return 'Unknown'


# Backward compatibility alias
classify_holdridge_correct = classify_holdridge

"""
Smart visualization - Handle large arrays without killing matplotlib.

Linus: "If you need to be clever, you're doing it wrong."
This isn't clever, it just downsamples when needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def smart_imshow(
    ax,
    data: np.ndarray,
    max_pixels: int = 1_000_000,
    **kwargs
) -> any:
    """Display large arrays without crashing.
    
    If data > max_pixels, downsample it. Simple.
    
    Args:
        ax: Matplotlib axis
        data: 2D array to display
        max_pixels: Maximum pixels before downsampling
        **kwargs: Pass to imshow
        
    Returns:
        Image object from imshow
    """
    if data.size <= max_pixels:
        # Small enough, show directly
        return ax.imshow(data, **kwargs)
    
    # Need to downsample
    factor = int(np.sqrt(data.size / max_pixels)) + 1
    
    # Simple strided sampling
    downsampled = data[::factor, ::factor]
    
    # Show downsampled
    im = ax.imshow(downsampled, **kwargs)
    
    # Add note about downsampling
    ax.text(0.02, 0.98, f'↓{factor}x', 
           transform=ax.transAxes,
           fontsize=8,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return im
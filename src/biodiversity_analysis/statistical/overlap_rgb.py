"""
RGB overlap map - Visualize multiple biodiversity layers as RGB.

Linus: "Show me something useful, not fancy."
RGB: Red for plants, Green for animals, Blue for fungi. Simple.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def create_rgb_overlap_map(
    data: Dict[str, np.ndarray],
    output_path: Union[str, Path]
) -> Path:
    """Create RGB overlap visualization.
    
    Aboveground (plants/animals) vs Belowground (fungi).
    
    Args:
        data: Dictionary with biodiversity arrays
        output_path: Where to save
        
    Returns:
        Path to saved figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract layers
    plants = data.get('plants', np.zeros((100, 100)))
    animals = data.get('animals', np.zeros_like(plants))
    
    # Fungi - merged or separate
    if 'fungi' in data:
        fungi = data['fungi']
    elif 'merged_fungi' in data:
        fungi = data['merged_fungi']
    elif 'am_fungi' in data and 'ecm_fungi' in data:
        fungi = data['am_fungi'] + data['ecm_fungi']
    elif 'am_fungi' in data:
        fungi = data['am_fungi']
    elif 'ecm_fungi' in data:
        fungi = data['ecm_fungi']
    else:
        fungi = np.zeros_like(plants)
    
    # Ensure same shape
    min_shape = (
        min(plants.shape[0], animals.shape[0], fungi.shape[0]),
        min(plants.shape[1], animals.shape[1], fungi.shape[1])
    )
    plants = plants[:min_shape[0], :min_shape[1]]
    animals = animals[:min_shape[0], :min_shape[1]]
    fungi = fungi[:min_shape[0], :min_shape[1]]
    
    # Normalize each layer to 0-1
    def normalize(arr):
        """Normalize array to 0-1, handling NaN."""
        arr = np.nan_to_num(arr, 0)
        vmin, vmax = np.percentile(arr[arr > 0], [2, 98]) if np.any(arr > 0) else (0, 1)
        normalized = (arr - vmin) / (vmax - vmin + 1e-10)
        return np.clip(normalized, 0, 1)
    
    # Create RGB image
    rgb = np.zeros((*min_shape, 3))
    rgb[:, :, 0] = normalize(plants)     # Red channel
    rgb[:, :, 1] = normalize(animals)    # Green channel  
    rgb[:, :, 2] = normalize(fungi)      # Blue channel
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # RGB composite
    axes[0].imshow(rgb, extent=[-180, 180, -90, 90], aspect='auto')
    axes[0].set_title('Biodiversity RGB Composite\n(R=Plants, G=Animals, B=Fungi)', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(True, alpha=0.3)
    
    # Legend/explanation
    axes[1].axis('off')
    explanation = """
    RGB Composite Interpretation:
    
    • Red: Plant diversity
    • Green: Animal diversity  
    • Blue: Fungal diversity
    
    Color Combinations:
    • Yellow (R+G): Plants + Animals
    • Magenta (R+B): Plants + Fungi
    • Cyan (G+B): Animals + Fungi
    • White: All three groups
    • Black: Low diversity
    
    Brighter colors = Higher diversity
    """
    axes[1].text(0.1, 0.5, explanation, fontsize=11, verticalalignment='center')
    
    # Save
    plt.suptitle('Biodiversity Overlap Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Created RGB overlap map: {output_path}")
    return output_path
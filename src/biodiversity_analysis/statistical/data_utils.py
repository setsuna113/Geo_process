from src.abstractions import get_logger
"""
Data utilities for statistical methods.

Linus: "Don't hide your light under a bushel"
Simple utilities that do one thing well.
"""

import json
from pathlib import Path
from typing import Any, Dict

logger = get_logger(__name__)


def save_results(data: Dict[str, Any], output_path: Path) -> None:
    """Save results to JSON file.
    
    Simple JSON dump. No magic.
    
    Args:
        data: Dictionary to save
        output_path: Where to save it
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Saved results to {output_path}")


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load results from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def save_figure(fig, output_path: Path, dpi: int = 150) -> None:
    """Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        output_path: Where to save it
        dpi: Resolution
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
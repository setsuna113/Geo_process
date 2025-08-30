from src.abstractions import get_logger
"""Base reporter interface for CP-WOPT visualization and reporting.

This module provides the abstract base class for all CP-WOPT reporters,
ensuring consistent interface and shared functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = get_logger(__name__)


class BaseReporter(ABC):
    """Abstract base class for all CP-WOPT reporters.
    
    Provides common functionality and ensures consistent interface
    across all reporter implementations.
    """
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize base reporter.
        
        Args:
            output_dir: Directory to save outputs
            config: Reporting configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Common configuration
        self.plot_format = config.get('plot_format', 'png')
        self.plot_dpi = config.get('plot_dpi', 150)
        self.colormap = config.get('colormap', 'viridis')
        self.generate_plots = config.get('generate_plots', True)
        
        # Setup matplotlib style
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configure matplotlib plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = self.plot_dpi
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   tight_layout: bool = True) -> Path:
        """Save a matplotlib figure with consistent settings.
        
        Args:
            fig: Figure to save
            filename: Name of the file (without extension)
            tight_layout: Whether to apply tight layout
            
        Returns:
            Path to saved file
        """
        if tight_layout:
            fig.tight_layout()
        
        filepath = self.output_dir / f"{filename}.{self.plot_format}"
        fig.savefig(filepath, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved figure: {filepath}")
        return filepath
    
    @abstractmethod
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report from CP-WOPT results.
        
        Args:
            results: CP-WOPT analysis results
            
        Returns:
            Dictionary with report outputs (paths to files, metrics, etc.)
        """
        pass
    
    def _validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate that results contain required fields.
        
        Args:
            results: Results dictionary to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_fields = self.get_required_fields()
        
        for field in required_fields:
            if field not in results:
                raise ValueError(f"Missing required field: {field}")
        
        return True
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get list of required fields in results dictionary.
        
        Returns:
            List of required field names
        """
        pass
    
    def _create_colorbar(self, ax, mappable, label: str = ""):
        """Add a properly formatted colorbar to a plot.
        
        Args:
            ax: Matplotlib axis
            mappable: The mappable object (e.g., from imshow)
            label: Colorbar label
        """
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        if label:
            cbar.set_label(label, rotation=270, labelpad=15)
        return cbar
    
    def _apply_smoothing(self, data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to data.
        
        Args:
            data: Input data array
            sigma: Smoothing parameter
            
        Returns:
            Smoothed data
        """
        if sigma <= 0:
            return data
            
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(data, sigma=sigma)
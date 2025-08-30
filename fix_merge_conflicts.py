#!/usr/bin/env python3
"""Fix remaining merge conflicts in statistical files."""

import re

def fix_overlap_py():
    """Fix overlap.py merge conflicts."""
    with open('src/biodiversity_analysis/statistical/overlap.py', 'r') as f:
        content = f.read()
    
    # Fix the first conflict - choose the cleaner version without redundant import
    content = content.replace(
        """<<<<<<< HEAD
            # create_rgb_overlap_map(data, rgb_output)
=======
            _create_cartopy_overlap_map(data, rgb_output)
>>>>>>> origin/main""",
        """            _create_cartopy_overlap_map(data, rgb_output)"""
    )
    
    # Fix the second conflict - remove redundant logger import
    content = content.replace(
        """<<<<<<< HEAD
        import logging
        logger = get_logger(__name__)
=======
        # LINUS: "Don't reimport what's already imported"
>>>>>>> origin/main""",
        """        # LINUS: "Don't reimport what's already imported"""""
    )
    
    # Add proper logger import at top if not there
    if "logger = logging.getLogger(__name__)" in content:
        content = content.replace("logger = logging.getLogger(__name__)", "logger = get_logger(__name__)")
    
    # Ensure we have the right import at the top
    if "from src.abstractions import get_logger" not in content:
        # Add it after the docstring
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '"""' in line and i > 5:  # Found end of docstring
                lines.insert(i+1, "\nfrom src.abstractions import get_logger")
                break
        content = '\n'.join(lines)
    
    with open('src/biodiversity_analysis/statistical/overlap.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed overlap.py")


def fix_partial_correlation_py():
    """Fix partial_correlation.py merge conflicts."""
    with open('src/biodiversity_analysis/statistical/partial_correlation.py', 'r') as f:
        content = f.read()
    
    # Remove the conflict markers and keep only data_utils imports
    content = content.replace(
        """<<<<<<< HEAD
=======
from src.biodiversity_analysis.statistical.result_types import MethodResult
from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
>>>>>>> origin/main""",
        """from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure"""
    )
    
    with open('src/biodiversity_analysis/statistical/partial_correlation.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed partial_correlation.py")


def fix_trust_map_py():
    """Fix trust_map.py merge conflicts."""
    with open('src/biodiversity_analysis/statistical/trust_map.py', 'r') as f:
        content = f.read()
    
    # Fix first conflict - remove smart_visualization import
    content = content.replace(
        """<<<<<<< HEAD
=======
from .smart_visualization import smart_imshow
>>>>>>> origin/main""",
        ""
    )
    
    # Fix second conflict - choose the multiband version
    content = content.replace(
        """<<<<<<< HEAD
    # NO DOWNSAMPLING - plt.imshow handles it automatically
=======
    # Extract multiband raster - NO SPECIAL CASES
    multiband = data['multiband']
    band_names = data['band_names']
>>>>>>> origin/main""",
        """    # Extract multiband raster - NO SPECIAL CASES
    multiband = data['multiband']
    band_names = data['band_names']"""
    )
    
    # Fix third major conflict block
    old_block = """<<<<<<< HEAD
            # Create plots - plt.imshow handles large data automatically
            _plot_trust_analysis(
                data_array, cv_array, name, 
                output_dir, trust_metrics
            )
            
            results[name] = trust_metrics
        else:
            # No uncertainty data available
            basic_metrics = _compute_basic_metrics(data_array)
            results[name] = basic_metrics
            
            # Simple visualization - plt.imshow handles large data
            _plot_basic_map(data_array, name, output_dir, basic_metrics)
=======
>>>>>>> origin/main"""
    
    # Remove the conflicted block entirely (it's handled differently now)
    content = content.replace(old_block, "")
    
    # Fix fourth conflict
    content = content.replace(
        """<<<<<<< HEAD
def _compute_trust_metrics(data: np.ndarray, cv: np.ndarray) -> Dict[str, Any]:
    \"\"\"Compute trust metrics from data and CV.\"\"\"
    # Remove invalid pixels
    valid_mask = ~np.isnan(data) & ~np.isnan(cv) & (data > 0)
=======
def _compute_trust_metrics_rasterio(
    data_source: Union[str, rasterio.DatasetReader],
    cv_source: Union[str, rasterio.DatasetReader]
) -> Dict[str, Any]:
    \"\"\"Compute trust metrics using rasterio block-wise processing.\"\"\"
>>>>>>> origin/main""",
        """def _compute_trust_metrics_rasterio(
    data_source: Union[str, rasterio.DatasetReader],
    cv_source: Union[str, rasterio.DatasetReader]
) -> Dict[str, Any]:
    \"\"\"Compute trust metrics using rasterio block-wise processing.\"\"\""""
    )
    
    # Fix fifth conflict
    content = content.replace(
        """<<<<<<< HEAD
def _compute_basic_metrics(data: np.ndarray) -> Dict[str, Any]:
    \"\"\"Compute basic metrics when no CV available.\"\"\"
    valid_mask = ~np.isnan(data)
=======
def _compute_basic_trust_metrics(
    multiband: rasterio.DatasetReader,
    band_idx: int,
    band_name: str
) -> Dict[str, Any]:
    \"\"\"Compute basic trust metrics for a band.
>>>>>>> origin/main""",
        """def _compute_basic_trust_metrics(
    multiband: rasterio.DatasetReader,
    band_idx: int,
    band_name: str
) -> Dict[str, Any]:
    \"\"\"Compute basic trust metrics for a band."""
    )
    
    with open('src/biodiversity_analysis/statistical/trust_map.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed trust_map.py")


def main():
    """Fix all merge conflicts."""
    print("Fixing merge conflicts...")
    
    fix_overlap_py()
    fix_partial_correlation_py()
    fix_trust_map_py()
    
    print("\nAll conflicts resolved!")


if __name__ == '__main__':
    main()
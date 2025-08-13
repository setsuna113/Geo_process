"""Fungi richness computation with H2O support.

Handles the creation of fungi_richness column from am_fungi_richness and ecm_fungi_richness.
Works directly with H2O frames.
"""

import logging
import h2o
from typing import Optional

logger = logging.getLogger(__name__)


def get_fungi_richness(am_fungi: h2o.H2OFrame, ecm_fungi: h2o.H2OFrame) -> h2o.H2OFrame:
    """Get fungi_richness column from AM and ECM fungi columns.
    
    Creates fungi_richness as sum of am_fungi_richness and ecm_fungi_richness.
    
    Args:
        am_fungi: H2O column with AM fungi richness
        ecm_fungi: H2O column with ECM fungi richness
        
    Returns:
        H2O frame with single column 'fungi_richness'
    """
    # Create fungi_richness as new frame
    logger.info("Computing fungi_richness...")
    fungi_richness = am_fungi + ecm_fungi
    fungi_richness.set_name(0, 'fungi_richness')
    
    # Log statistics
    mean_val = fungi_richness.mean()
    std_val = fungi_richness.sd()
    logger.info(f"Created fungi_richness: mean={mean_val:.2f}, std={std_val:.2f}")
    
    return fungi_richness
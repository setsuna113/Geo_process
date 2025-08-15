"""Improved spatial weights computation with better parallelization.

Uses multiprocessing and chunking for efficient parallel computation.
"""

import numpy as np
import logging
import os
from multiprocessing import Pool, cpu_count
from functools import partial

logger = logging.getLogger(__name__)


def _haversine_chunk(args):
    """Compute haversine distances for a chunk of coordinates.
    
    This function computes distances for a subset of rows against all columns.
    """
    coords1_chunk, coords2_rad, start_idx = args
    
    n1 = coords1_chunk.shape[0]
    n2 = coords2_rad.shape[0]
    distances = np.empty((n1, n2), dtype=np.float32)
    
    for i in range(n1):
        lat1 = coords1_chunk[i, 0]
        lon1 = coords1_chunk[i, 1]
        
        # Vectorized computation for all coords2
        lat2 = coords2_rad[:, 0]
        lon2 = coords2_rad[:, 1]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        distances[i, :] = c
    
    return start_idx, distances


def haversine_distance_km_chunked(coords1: np.ndarray, coords2: np.ndarray = None, n_jobs: int = None) -> np.ndarray:
    """Compute haversine distances using multiprocessing with chunking.
    
    This implementation uses true multiprocessing for better CPU utilization.
    
    Args:
        coords1: Coordinates (n1, 2) with [lon, lat] in degrees
        coords2: Optional second set of coordinates (n2, 2)
        n_jobs: Number of parallel jobs (default: from env or all CPUs)
        
    Returns:
        Distance matrix in km
    """
    # Get number of workers
    if n_jobs is None:
        n_jobs = int(os.environ.get('MESF_N_JOBS', '120'))
        if n_jobs <= 0:
            n_jobs = cpu_count()
    
    logger.info(f"Using {n_jobs} processes for parallel haversine computation")
    
    # Convert to radians and swap to [lat, lon] order
    coords1_rad = np.radians(coords1[:, [1, 0]]).astype(np.float32)
    
    if coords2 is None:
        coords2_rad = coords1_rad
    else:
        coords2_rad = np.radians(coords2[:, [1, 0]]).astype(np.float32)
    
    n1, n2 = len(coords1_rad), len(coords2_rad)
    logger.info(f"Computing {n1:,} x {n2:,} = {n1*n2:,} distances using {n_jobs} processes")
    
    # Determine chunk size - aim for at least 100 chunks for good load balancing
    # but not too small to avoid overhead
    min_chunk_size = max(10, n1 // (n_jobs * 10))
    chunk_size = min(1000, max(min_chunk_size, n1 // 100))
    n_chunks = (n1 + chunk_size - 1) // chunk_size
    
    logger.info(f"Processing in {n_chunks} chunks of size ~{chunk_size}")
    
    # Prepare chunks
    chunks = []
    for i in range(0, n1, chunk_size):
        end_idx = min(i + chunk_size, n1)
        chunk = coords1_rad[i:end_idx]
        chunks.append((chunk, coords2_rad, i))
    
    # Process in parallel
    distances_rad = np.empty((n1, n2), dtype=np.float32)
    
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_haversine_chunk, chunks)
        
        # Assemble results
        for start_idx, chunk_distances in results:
            end_idx = start_idx + chunk_distances.shape[0]
            distances_rad[start_idx:end_idx, :] = chunk_distances
    
    # Convert from radians to km (Earth radius = 6371 km)
    return distances_rad * 6371


# Make this the default for the module
haversine_distance_km = haversine_distance_km_chunked
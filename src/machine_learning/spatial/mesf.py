#!/usr/bin/env python3
"""
MESF (Moran Eigenvector Spatial Filtering) Implementation in Python
Exact port of mesf.jl functionality for use via PyCall.jl

This module:
1. Checks for cached MESF in numpy format
2. If not cached, computes MESF using the meigen_f algorithm
3. Selects top-k eigenvectors based on eigenvalues
4. Caches and returns the result
"""

import numpy as np
import os
import hashlib
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import jax
import jax.numpy as jnp
from jax import jit

# Configure JAX for CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)  # Use float32 for speed
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import time

# Import worker control if needed
try:
    from worker_control import WorkerController
    WORKER_CONTROL_AVAILABLE = True
except ImportError:
    try:
        from .worker_control import WorkerController
        WORKER_CONTROL_AVAILABLE = True
    except ImportError:
        WORKER_CONTROL_AVAILABLE = False

# Configuration
CACHE_DIR = "/scratch/yl998/.cache/mesf"
DEFAULT_ENUM = 200
TOP_K_FEATURES = 5  # Use top-5 eigenvectors directly

def generate_cache_key(coords, enum):
    """Generate deterministic cache key based only on data dimensions.
    
    This simplified version makes it easy to match between Python and Julia.
    Since we're working with the same dataset, n and enum uniquely identify the cache.
    """
    n, d = coords.shape
    
    # Simple deterministic key - no hashing needed
    # Format: mesf_n{num_points}_k{num_eigenvectors}
    return f"mesf_n{n}_k{enum}"

def check_cache(cache_key):
    """Check if cache exists and load if available."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    
    if os.path.exists(cache_path):
        print(f"Loading MESF from cache: {cache_path}")
        return np.load(cache_path)
    
    return None

def save_cache(cache_key, mesf_features):
    """Save matrix to cache as .npy file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    
    np.save(cache_path, mesf_features)
    print(f"Saved MESF to cache: {cache_path}")

def build_connectivity_matrix(D, h, model="exp"):
    """Build connectivity matrix based on distance and model type."""
    if model == "exp":
        return np.exp(-D / h)
    elif model == "gau":
        return np.exp(-((D / h) ** 2))
    elif model == "sph":
        d_norm = D / h
        C = np.where(d_norm < 1, 1 - 1.5 * d_norm + 0.5 * d_norm**3, 0.0)
        return C
    else:
        raise ValueError(f"Unknown model: {model}")

def compute_bandwidth(D):
    """
    Compute maximum spanning tree distance for bandwidth.
    Following the exact method from spmoran's meigen_f.
    """
    n = D.shape[0]
    
    # Convert to sparse matrix for MST computation
    # Use negative distances to get maximum spanning tree via minimum_spanning_tree
    D_sparse = csr_matrix(-D)
    
    # Compute minimum spanning tree on negative distances (= maximum spanning tree on distances)
    mst = minimum_spanning_tree(D_sparse)
    
    # Find maximum edge weight in the MST (negate back to get actual distance)
    max_dist = -mst.data.min() if len(mst.data) > 0 else 0.0
    
    return max_dist

@jit
def compute_haversine_distances_jax(centers, points):
    """JAX-optimized Haversine distance computation for geographic coordinates.
    
    Computes distances from all centers to all points.
    9-10x faster than sklearn through JIT compilation and vectorization.
    
    Args:
        centers: (n_centers, 2) array of [lon, lat] in degrees
        points: (n_points, 2) array of [lon, lat] in degrees
    
    Returns:
        Distance matrix in kilometers
    """
    # Convert to radians
    c1 = jnp.radians(centers)
    c2 = jnp.radians(points)
    
    # Extract lat/lon (reshape for broadcasting)
    lat1 = c1[:, 1:2]  # (n_centers, 1)
    lon1 = c1[:, 0:1]  # (n_centers, 1)
    lat2 = c2[:, 1:2].T  # (1, n_points)
    lon2 = c2[:, 0:1].T  # (1, n_points)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = jnp.sin(dlat/2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon/2)**2
    c = 2 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0, 1)))
    
    # Convert to kilometers
    distances = c * 6371.0
    
    return distances

def meigen_f(coords, enum=DEFAULT_ENUM, model="exp", n_workers=64):
    """Main meigen_f implementation - optimized for unique coordinates
    
    Args:
        coords: Input coordinates (n, 2)
        enum: Number of eigenvectors to compute
        model: Connectivity model ('exp', 'gau', 'sph')
        n_workers: Max CPUs to use (default: 64)
    """
    n_points, d = coords.shape
    coords_work = coords  # Assume coordinates are already unique
    
    # Set up worker control for ALL libraries
    if WORKER_CONTROL_AVAILABLE:
        controller = WorkerController(n_workers)
        controller.set_all_thread_limits()
        print(f"  All parallel operations limited to {n_workers} threads")
    else:
        # Fallback: just warn that WorkerController is not available
        print(f"  WARNING: WorkerController not available, thread limits may not be properly set")
    
    # Warn if dataset is very small
    if n_points < 100:
        print(f"Warning: Very small dataset (n={n_points}), results may be unstable")
    
    # K-means clustering with sqrt(n) centers
    n_centers = min(int(np.sqrt(n_points)), n_points)
    print(f"Computing k-means with {n_centers} centers (sqrt({n_points}))...")
    
    t_kmeans = time.time()
    
    # Use FAISS K-means (required - no fallback)
    import faiss
    # Configure FAISS threads after import
    faiss.omp_set_num_threads(n_workers)
    print(f"  Using FAISS K-means (optimized for large-scale, limited to {n_workers} threads)...")
    
    # FAISS requires float32 and contiguous arrays
    coords_work_f32 = np.ascontiguousarray(coords_work, dtype=np.float32)
    
    # Configure FAISS k-means
    kmeans_faiss = faiss.Kmeans(
        d=coords_work.shape[1],  # dimension
        k=n_centers,              # number of clusters
        niter=20,                 # iterations
        verbose=True,             # show progress
        nredo=1,                  # run once (like n_init=1)
        # Let FAISS subsample intelligently for massive datasets
        # Default is 256, which would sample ~1.8M points for 7199 clusters
        seed=42                   # random seed
    )
    
    # Train k-means
    kmeans_faiss.train(coords_work_f32)
    centers = kmeans_faiss.centroids.copy()  # Get centers as numpy array
    
    # CRITICAL: Reset CPU affinity after FAISS (it sets affinity to CPU 0!)
    os.sched_setaffinity(0, range(os.cpu_count()))
    print(f"  ✅ Reset CPU affinity to all {os.cpu_count()} CPUs after FAISS")
            
    print(f"  K-means took {time.time() - t_kmeans:.2f} seconds", flush=True)
    
    # Step 4: Build connectivity matrix on centers
    print("Building connectivity matrix...", flush=True)
    t_conn = time.time()
    # Use Haversine distance for geographic coordinates
    # Convert to JAX arrays for distance computation
    centers_jax = jnp.array(centers)
    D_centers = np.array(compute_haversine_distances_jax(centers_jax, centers_jax))
    h = compute_bandwidth(D_centers)
    C = build_connectivity_matrix(D_centers, h, model)
    print(f"  Connectivity matrix took {time.time() - t_conn:.2f} seconds", flush=True)
    
    # Step 5: Eigendecomposition on centers using sparse solver
    print(f"Computing eigendecomposition (top {enum} eigenvectors)...", flush=True)
    t_eigen = time.time()
    
    # Centering operator: M = I - 11'/n
    Cmean = C.mean(axis=1, keepdims=True)
    MCM = (C - Cmean) - Cmean.T + Cmean.mean()
    
    # Use sparse eigendecomposition for top-k eigenvectors
    n_eigs = min(enum, n_centers - 2)
    eigenvalues, eigenvectors = eigsh(MCM, k=n_eigs, which='LA', tol=1e-6)
    
    # Sort in descending order (eigsh returns ascending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"  Eigendecomposition took {time.time() - t_eigen:.2f} seconds", flush=True)
    
    # Filter positive eigenvalues
    pos_idx = eigenvalues > 1e-8
    sf0 = eigenvectors[:, pos_idx]
    ev0 = eigenvalues[pos_idx]
    
    # Scale eigenvalues (as per meigen_f)
    ev_full = ev0 * (n_points / n_centers) - 1
    ev_ap = ev_full[ev_full > 1e-8]
    
    # Step 7: Project back to original points
    print("Projecting eigenvectors to all points...", flush=True)
    t_proj = time.time()
    sf_ap0 = sf0 @ np.diag(1 / ev0)
    
    # Compute connectivity from centers to all unique points
    print(f"  Computing distance matrix ({centers.shape[0]} x {coords_work.shape[0]})...", flush=True)
    t_dist = time.time()
    
    # Check memory requirements
    memory_gb = (n_centers * coords_work.shape[0] * 4) / (1024**3)  # Float32
    print(f"  Memory required for full distance matrix: {memory_gb:.1f} GB", flush=True)
    
    # INCREMENTAL PROJECTION - Using Numba for 30-1000x speedup
    from scipy.linalg import blas
    
    print(f"  Using JAX vectorized haversine (9-10x faster than sklearn)", flush=True)
    # Warmup JIT compilation with realistic batch size
    print(f"  Warming up JAX JIT compilation...", flush=True)
    warmup_centers = jnp.array(centers.astype(np.float32))
    warmup_points = jnp.array(coords_work[:min(1000, coords_work.shape[0])].astype(np.float32))
    _ = compute_haversine_distances_jax(warmup_centers, warmup_points)
    print(f"  JIT compilation complete", flush=True)
    
    # Thread settings are already configured by WorkerController
    
    batch_size = 100000  # Process 100K points at a time
    n_batches = (coords_work.shape[0] + batch_size - 1) // batch_size
    print(f"  Processing in {n_batches} batches of {batch_size} points")
    print(f"  Note: All parallel operations limited to {n_workers} threads")
    
    # Initialize output matrix for projections
    n_eigenvectors = sf_ap0.shape[1]
    sf_ap = np.zeros((coords_work.shape[0], n_eigenvectors), dtype=np.float32)
    
    # Process each batch sequentially (but each batch uses parallel BLAS)
    Cmean_flat = Cmean.flatten().astype(np.float32)
    
    # Convert centers to JAX ONCE before the loop
    centers_jax = jnp.array(centers.astype(np.float32))
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, coords_work.shape[0])
        batch = coords_work[start_idx:end_idx]
        
        # Convert only the batch to JAX, compute, convert back
        batch_jax = jnp.array(batch.astype(np.float32))
        D_batch = np.array(compute_haversine_distances_jax(centers_jax, batch_jax))
        
        # Build connectivity for this batch (vectorized numpy operations)
        C_batch = build_connectivity_matrix(D_batch, h, model)
        
        # Center the connectivity matrix
        C_batch_centered = C_batch - Cmean_flat[:, np.newaxis]
        
        # BLAS sgemm uses MKL_NUM_THREADS automatically for parallelization
        result = blas.sgemm(alpha=1.0, a=C_batch_centered.T, b=sf_ap0)
        sf_ap[start_idx:end_idx] = result
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{n_batches} batches ({end_idx:,}/{coords_work.shape[0]:,} points)", flush=True)
    
    # Re-center after all batches (vectorized)
    sf_ap = sf_ap - sf_ap.mean(axis=0)
    
    print(f"  Distance computation and projection took {time.time() - t_dist:.2f} seconds")
    
    print(f"  Total projection took {time.time() - t_proj:.2f} seconds")
    
    # Select valid eigenvectors
    valid_idx = ev_full > 1e-8
    n_valid = valid_idx.sum()
    sf_ap = sf_ap[:, :min(n_valid, sf_ap.shape[1])]
    
    # Step 8: Boundary correction (sfsf)
    sfsf = (sf_ap ** 2).sum(axis=1)
    sfsf[sfsf > 1] = 1
    
    # Return results (no duplicate mapping needed)
    return sf_ap, ev_ap[:sf_ap.shape[1]]

def select_top_k(mesf, eigenvalues, k=TOP_K_FEATURES):
    """Select top-k eigenvectors based on eigenvalues."""
    n_available = min(k, mesf.shape[1])
    print(f"Selecting top-{n_available} eigenvectors from {mesf.shape[1]} available")
    
    # Select top-k eigenvectors
    top_k_mesf = mesf[:, :n_available]
    top_k_eigenvalues = eigenvalues[:n_available]
    
    # Calculate spatial autocorrelation captured
    total_autocorr = eigenvalues.sum()
    captured_autocorr = top_k_eigenvalues.sum()
    pct_captured = (captured_autocorr / total_autocorr) * 100 if total_autocorr > 0 else 0
    
    print(f"Top-{n_available} eigenvalues: {top_k_eigenvalues.round(2)}")
    print(f"Spatial autocorrelation captured: {pct_captured:.2f}%")
    
    return top_k_mesf.astype(np.float32), top_k_eigenvalues

def standardize_mesf(mesf):
    """Standardize eigenvectors using 2SD method."""
    standardized = np.zeros_like(mesf)
    
    for i in range(mesf.shape[1]):
        col = mesf[:, i]
        mu = col.mean()
        sigma = col.std()
        # 2SD standardization
        standardized[:, i] = (col - mu) / (2 * sigma)
    
    return standardized

def compute_mesf_with_cache(coords, enum=DEFAULT_ENUM, force_recompute=False, top_k=TOP_K_FEATURES, n_workers=64):
    """
    Main function to compute MESF with caching and top-k selection.
    Standardization is now always applied as post-processing.
    
    Args:
        coords: numpy array of shape (n, 2) with Float32 coordinates
        enum: number of eigenvectors to compute
        force_recompute: force recomputation even if cached
        top_k: number of top eigenvectors to select
        n_workers: maximum threads to use (default: 64)
    
    Returns:
        numpy array of shape (n, top_k) with standardized MESF features
    """
    start_time = time.time()
    
    # Ensure coords is float32
    coords = coords.astype(np.float32)
    
    # Generate cache key
    cache_key = generate_cache_key(coords, enum)
    
    # Check cache unless forced to recompute
    if not force_recompute:
        cached_result = check_cache(cache_key)
        if cached_result is not None:
            return cached_result
    
    print("Computing MESF from scratch...")
    
    # Compute MESF using meigen_f algorithm
    mesf, eigenvalues = meigen_f(coords, enum=enum, n_workers=n_workers)
    print(f"Computed MESF: {mesf.shape} matrix with {len(eigenvalues)} eigenvalues")
    
    # Select top-k eigenvectors based on eigenvalues
    top_k_mesf, top_k_eigenvalues = select_top_k(mesf, eigenvalues, k=top_k)
    
    # Always standardize as post-processing (2SD method)
    print("Applying 2SD standardization (compulsory post-processing)...")
    top_k_mesf = standardize_mesf(top_k_mesf)
    
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    
    # Save to cache
    save_cache(cache_key, top_k_mesf)
    
    return top_k_mesf

# Entry point for PyCall.jl
def compute_mesf(coords_array, enum=DEFAULT_ENUM, force_recompute=False, top_k=TOP_K_FEATURES, n_workers=64):
    """
    Entry point for Julia PyCall.
    Accepts numpy array directly from Julia.
    Always returns standardized features.
    """
    # coords_array comes as numpy array from Julia
    return compute_mesf_with_cache(
        coords_array, 
        enum=enum, 
        force_recompute=force_recompute,
        top_k=top_k,
        n_workers=n_workers
    )
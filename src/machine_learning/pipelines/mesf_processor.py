#!/usr/bin/env python
"""MESF processor for Julia hybrid pipeline integration.

Direct CLI interface for Julia to compute MESF eigenvectors.
Can be called from command line or imported as a module.

Usage:
    python mesf_processor.py <data_path> [options]
    
Example:
    python mesf_processor.py /data/bio.parquet --spatial-step 10 --n-eigenvectors 10
"""

import sys
import os

# No environment variable control - let taskset handle CPU limiting
# Libraries will auto-detect available cores within taskset limits

# NOW import other libraries
import argparse
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path BEFORE any project imports  
# mesf_processor.py is at src/machine_learning/pipelines/mesf_processor.py
# Go up 3 levels: pipelines -> machine_learning -> src -> [project_root]
project_root = Path(__file__).resolve().parents[3]  # Use parents[3] for clarity
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

# Now import spatial modules
from src.machine_learning.spatial.nystrom_eigensolver import NystromEigensolver
from src.machine_learning.spatial.mesf_cache_manager import MESFCacheManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MESFProcessor:
    """Clean Nystrom-only MESF processor optimized for large datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None, n_jobs: Optional[int] = None):
        """Initialize MESF processor.
        
        Args:
            cache_dir: Optional cache directory for results
            n_jobs: Number of parallel jobs (default: from config or -1)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'mesf'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs  # Will be set from config in get_mesf_eigenvectors
        
        # Initialize robust cache manager (always use it)
        self.cache_manager = MESFCacheManager(cache_dir=str(self.cache_dir))
        logger.info(f"MESF processor initialized with robust cache at {self.cache_dir}")
    
    def get_mesf_eigenvectors(
        self,
        data_path: str,
        spatial_grid_step: Optional[int] = None,
        n_eigenvectors: int = 10,
        distance_threshold: float = 100.0,
        n_jobs: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Compute MESF eigenvectors using Nystrom approximation.
        
        This is the main entry point called by the orchestrator.
        
        Args:
            data_path: Path to arrow file
            spatial_grid_step: Spatial sampling step (must match orchestrator)
            n_eigenvectors: Number of eigenvectors to compute (default 10)
            distance_threshold: Distance threshold in km for spatial weights
            n_jobs: Number of parallel jobs (default: use instance setting)
            
        Returns:
            H2O frame with ME1, ME2, ..., ME10 columns matching input rows
        """
        logger.info("="*60)
        logger.info("MESF NYSTROM COMPUTATION")
        logger.info("="*60)
        
        # Use provided n_jobs or fall back to instance setting
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        if n_jobs is None:
            n_jobs = -1  # Default to all cores if not specified
        
        # No thread control - taskset will limit CPU cores
        n_cores = n_jobs if n_jobs != -1 else 120
        logger.info(f"Configured for {n_cores} CPU cores (enforced by taskset)")
        
        # Use robust cache manager
        mesf_params = {
            'n_eigenvectors': n_eigenvectors,
            'distance_threshold': distance_threshold
        }
        
        def compute_mesf_fn(file_path, grid_step, params):
            # Compute MESF and return eigenvectors + metadata
            eigenvectors = self._compute_mesf_internal(
                file_path, grid_step, params['n_eigenvectors'], 
                params['distance_threshold'], n_jobs
            )
            metadata = {
                'n_eigenvectors': params['n_eigenvectors'],
                'distance_threshold': params['distance_threshold'],
                'spatial_grid_step': grid_step,
                'computed_at': str(np.datetime64('now')),
                'n_samples': eigenvectors.shape[0] if eigenvectors is not None else 0
            }
            return eigenvectors, metadata
        
        # Get or compute MESF using robust cache
        return self.cache_manager.get_or_compute_mesf(
            data_path, spatial_grid_step or 1, mesf_params, compute_mesf_fn
        )
        
        # Step 1: Count total rows efficiently
        n_total = self._count_total_rows(data_path, spatial_grid_step)
        if n_total == 0:
            logger.error("No data found")
            return None
        
        # Step 2: Calculate Nystrom parameters using 100×√n rule
        n_landmark, n_sample = self._calculate_nystrom_params(n_total)
        
        # Step 3: Select and load sample coordinates (includes landmarks)
        sample_indices, landmark_indices = self._select_indices(n_total, n_landmark, n_sample)
        coords_sample = self._load_coordinates(data_path, sample_indices, spatial_grid_step)
        
        if coords_sample is None:
            return None
        
        # Extract landmark coordinates from samples
        # Landmarks are the first n_landmark rows of samples
        coords_landmark = coords_sample[:n_landmark]
        
        # Step 4: Compute eigenvectors on landmarks
        eigenvectors_landmark = self._compute_landmark_eigenvectors(
            coords_landmark, n_eigenvectors, distance_threshold
        )
        
        if eigenvectors_landmark is None:
            return None
        
        # Step 5: Since landmarks = samples, no extension needed
        if n_sample == n_landmark:
            eigenvectors_sample = eigenvectors_landmark
            logger.info("Using landmark eigenvectors directly (landmarks = samples)")
        else:
            # Fallback for old configuration
            eigenvectors_sample = self._extend_to_samples(
                coords_sample, coords_landmark, eigenvectors_landmark
            )
        
        # Step 6: Interpolate to full dataset using RFF
        eigenvectors_full = self._interpolate_to_full_rff(
            data_path, spatial_grid_step, n_total,
            coords_sample, eigenvectors_sample, n_jobs
        )
        
        if eigenvectors_full is None:
            return None
        
        logger.info(f"MESF computation complete: {n_total:,} rows × {n_eigenvectors} eigenvectors")
        return eigenvectors_full
    
    def _compute_mesf_internal(self, data_path: str, spatial_grid_step: Optional[int], 
                              n_eigenvectors: int, distance_threshold: float, 
                              n_jobs: Optional[int]) -> Optional[np.ndarray]:
        """Internal MESF computation without caching, for use by cache manager.
        
        Args:
            data_path: Path to data file
            spatial_grid_step: Spatial sampling step
            n_eigenvectors: Number of eigenvectors
            distance_threshold: Distance threshold in km
            n_jobs: Number of parallel jobs
            
        Returns:
            MESF eigenvectors or None if error
        """
        # Count total rows
        n_total = self._count_total_rows(data_path, spatial_grid_step)
        if n_total == 0:
            logger.error("No data found")
            return None
        
        # Calculate Nystrom parameters
        n_landmark, n_sample = self._calculate_nystrom_params(n_total)
        
        # Select indices
        sample_indices, landmark_indices = self._select_indices(n_total, n_landmark, n_sample)
        
        # Load coordinates
        coords_sample = self._load_coordinates(data_path, sample_indices, spatial_grid_step)
        if coords_sample is None:
            return None
        
        coords_landmark = coords_sample[:n_landmark]
        
        # Compute eigenvectors on landmarks
        eigenvectors_landmark = self._compute_landmark_eigenvectors(
            coords_landmark, n_eigenvectors, distance_threshold
        )
        if eigenvectors_landmark is None:
            return None
        
        # Extend to samples
        eigenvectors_sample = self._extend_to_samples(
            coords_sample, coords_landmark, eigenvectors_landmark
        )
        if eigenvectors_sample is None:
            return None
        
        # Interpolate to full dataset
        eigenvectors_full = self._interpolate_to_full_rff(
            data_path, spatial_grid_step, n_total,
            coords_sample, eigenvectors_sample,
            n_jobs or self.n_jobs or -1
        )
        
        return eigenvectors_full
    
    def _count_total_rows(self, data_path: str, spatial_grid_step: Optional[int]) -> int:
        """Count total rows efficiently using Polars.
        
        Args:
            data_path: Path to arrow file
            spatial_grid_step: Sampling step
            
        Returns:
            Total number of rows after sampling
        """
        try:
            # Use Polars to load Arrow format
            df = pl.scan_ipc(data_path)  # IPC = Arrow format in Polars
            total_rows = df.select(pl.len()).collect()[0, 0]
            
            if spatial_grid_step and spatial_grid_step > 1:
                # Calculate sampled count - match H2O's range(0, n, step) behavior
                # This includes partial last group, so we need ceiling division
                sampled_rows = (total_rows + spatial_grid_step - 1) // spatial_grid_step
                logger.info(f"Data: {total_rows:,} rows → {sampled_rows:,} after 1/{spatial_grid_step} sampling")
                return sampled_rows
            else:
                logger.info(f"Data: {total_rows:,} rows (no sampling)")
                return total_rows
                
        except Exception as e:
            logger.error(f"Failed to count rows: {e}")
            return 0
    
    def _calculate_nystrom_params(self, n_total: int) -> Tuple[int, int]:
        """Calculate optimal Nystrom parameters.
        
        Uses 4×√n landmarks and 200×√n samples for RFF.
        
        Args:
            n_total: Total number of data points
            
        Returns:
            (n_landmark, n_sample) tuple
        """
        # 1×√n landmarks for eigendecomposition (reduced for feasibility)
        # With 51M points: 1×7204 = 7,204 landmarks (manageable)
        n_landmark = int(1 * np.sqrt(n_total))
        
        # 50×√n samples for RFF reference points (reduced for speed)
        # With 51M points: 50×7204 = 360,200 samples
        n_sample = int(50 * np.sqrt(n_total))
        
        # Apply reasonable bounds
        n_landmark = max(1000, min(10000, n_landmark))  # Cap at 10k for feasibility
        n_sample = max(10000, min(500000, n_sample))  # Cap at 500k for speed
        
        logger.info(f"Nystrom parameters: {n_landmark:,} landmarks (1×√{n_total:,}), "
                   f"{n_sample:,} samples (50×√{n_total:,}) for RFF")
        
        return n_landmark, n_sample
    
    def _select_indices(
        self, 
        n_total: int, 
        n_landmark: int, 
        n_sample: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select sample and landmark indices.
        
        Ensures landmarks ⊂ samples for mathematical consistency.
        
        Args:
            n_total: Total number of points
            n_landmark: Number of landmarks
            n_sample: Number of samples
            
        Returns:
            (sample_indices, landmark_indices) where landmarks ⊂ samples
        """
        # Select samples uniformly
        # Ensure we dont sample more than available
        n_sample = min(n_sample, n_total)
        sample_indices = np.random.choice(n_total, n_sample, replace=False)
        
        # Landmarks are the first n_landmark samples
        # This ensures landmarks ⊂ samples
        landmark_indices = sample_indices[:n_landmark]
        
        logger.info(f"Selected {n_sample:,} samples (including {n_landmark:,} landmarks)")
        
        return sample_indices, landmark_indices
    
    def _load_coordinates(
        self,
        data_path: str,
        indices: np.ndarray,
        spatial_grid_step: Optional[int]
    ) -> Optional[np.ndarray]:
        """Load x,y coordinates for specified indices.
        
        Args:
            data_path: Path to arrow file
            indices: Row indices to load (accounts for spatial sampling)
            spatial_grid_step: Sampling step
            
        Returns:
            Coordinates array (n_samples, 2) or None if error
        """
        try:
            # Load all x,y coordinates with Polars
            coords_df = pl.read_ipc(
                data_path,
                columns=['x', 'y'],
                memory_map=True
            )
            
            # Remove nulls
            coords_df = coords_df.drop_nulls()
            
            # Select only the sample indices using iloc-style indexing
            coords_df = coords_df[indices]
            
            # Convert to numpy float32 for efficiency
            coords_np = coords_df.to_numpy().astype(np.float32)
            
            logger.info(f"Loaded {len(coords_np):,} coordinate pairs")
            return coords_np
            
        except Exception as e:
            logger.error(f"Failed to load coordinates: {e}")
            return None
    
    def _compute_landmark_eigenvectors(
        self,
        coords_landmark: np.ndarray,
        n_eigenvectors: int,
        distance_threshold: float
    ) -> Optional[np.ndarray]:
        """Compute eigenvectors on landmark points.
        
        Args:
            coords_landmark: Landmark coordinates (n_landmark, 2)
            n_eigenvectors: Number of eigenvectors to compute
            distance_threshold: Distance threshold in km
            
        Returns:
            Eigenvectors (n_landmark, n_eigenvectors) or None if error
        """
        try:
            # Use abstracted Nystrom eigensolver
            eigensolver = NystromEigensolver(
                n_eigenvectors=n_eigenvectors,
                distance_threshold=distance_threshold
            )
            
            eigenvalues, eigenvectors = eigensolver.compute_landmark_eigenvectors(coords_landmark)
            return eigenvectors
            
        except Exception as e:
            logger.error(f"Eigendecomposition failed: {e}")
            return None
    
    def _extend_to_samples(
        self,
        coords_sample: np.ndarray,
        coords_landmark: np.ndarray,
        eigenvectors_landmark: np.ndarray
    ) -> np.ndarray:
        """Extend eigenvectors from landmarks to all samples.
        
        Args:
            coords_sample: All sample coordinates (n_sample, 2)
            coords_landmark: Landmark coordinates (n_landmark, 2)
            eigenvectors_landmark: Landmark eigenvectors (n_landmark, n_eigenvectors)
            
        Returns:
            Sample eigenvectors (n_sample, n_eigenvectors)
        """
        # Use abstracted Nystrom eigensolver
        eigensolver = NystromEigensolver()
        eigenvectors_sample = eigensolver.extend_to_samples(
            coords_sample,
            coords_landmark,
            eigenvectors_landmark
        )
        return eigenvectors_sample
    
    def _interpolate_to_full_rff(
        self,
        data_path: str,
        spatial_grid_step: Optional[int],
        n_total: int,
        coords_sample: np.ndarray,
        eigenvectors_sample: np.ndarray,
        n_jobs: int = -1
    ) -> Optional[np.ndarray]:
        """Interpolate eigenvectors to full dataset using Random Fourier Features.
        
        Uses JAX-optimized RFF with precomputed projection matrix.
        
        Args:
            data_path: Path to arrow file
            spatial_grid_step: Sampling step
            n_total: Total number of points
            coords_sample: Sample coordinates (n_sample, 2)
            eigenvectors_sample: Sample eigenvectors (n_sample, n_eigenvectors)
            
        Returns:
            Full eigenvectors (n_total, n_eigenvectors) or None if error
        """
        try:
            import psutil
            import jax
            import jax.numpy as jnp
            from jax import jit
            
            logger.info(f"Interpolating to {n_total:,} points using JAX-optimized RFF...")
            
            # JAX should already be configured from the top of the file
            # Don't reconfigure here as it causes nested threading
            
            # RFF parameters
            n_features = 5000  # Number of random features
            bandwidth = 100.0  # Gaussian kernel bandwidth in km
            
            # Convert to float32 for speed
            coords_sample = coords_sample.astype(np.float32)
            eigenvectors_sample = eigenvectors_sample.astype(np.float32)
            n_eigenvectors = eigenvectors_sample.shape[1]
            
            # Step 1: Generate random Fourier features (once!)
            logger.info(f"Generating {n_features} random Fourier features...")
            np.random.seed(42)  # For reproducibility
            D = np.random.randn(2, n_features).astype(np.float32) / bandwidth
            
            # Convert to JAX immediately for ALL operations
            D_jax = jnp.array(D)
            coords_sample_jax = jnp.array(coords_sample)
            eigenvectors_sample_jax = jnp.array(eigenvectors_sample)
            
            # Step 2: Compute reference features using JAX (PARALLELIZED!)
            logger.info(f"Computing RFF for {len(coords_sample):,} reference points using JAX...")
            proj_ref = coords_sample_jax @ D_jax  # JAX auto-parallelizes this
            
            # Use sin/cos features for lower variance (JAX operations)
            cos_features = jnp.cos(proj_ref)
            sin_features = jnp.sin(proj_ref)
            Z_ref = jnp.concatenate([cos_features, sin_features], axis=1)
            Z_ref = Z_ref * jnp.sqrt(1.0 / n_features)
            
            # Step 3: Precompute projection matrix W using JAX
            logger.info("Precomputing projection matrix W with JAX...")
            W_jax = Z_ref.T @ eigenvectors_sample_jax  # JAX parallelizes this too
            logger.info(f"Projection matrix W shape: {W_jax.shape}")
            
            # JIT-compile the RFF computation
            @jit
            def compute_rff_batch(batch_coords):
                """JIT-compiled RFF computation for a batch."""
                # Project coordinates
                proj = batch_coords @ D_jax
                
                # Compute sin/cos features
                cos_features = jnp.cos(proj)
                sin_features = jnp.sin(proj)
                Z = jnp.concatenate([cos_features, sin_features], axis=1)
                Z = Z * jnp.sqrt(1.0 / n_features)
                
                # Apply projection matrix
                return Z @ W_jax
            
            # Step 4: Load all coordinates
            logger.info("Loading full coordinates...")
            coords_df = pl.read_ipc(
                data_path,
                columns=['x', 'y'],
                memory_map=True
            )
            
            coords_df = coords_df.drop_nulls()
            coords_full = coords_df.to_numpy().astype(np.float32)  # Use float32 for efficiency
            
            # Step 5: Process in smaller batches to avoid memory issues
            available_memory = psutil.virtual_memory().available
            batch_size = 2_000_000  # Reduced to 2M points per batch for stability
            
            logger.info(f"Processing {n_total:,} points in batches of {batch_size:,}")
            logger.info(f"Available memory: {available_memory/(1024**3):.1f}GB")
            
            # Process batches and accumulate results
            eigenvectors_batches = []
            
            for start_idx in range(0, n_total, batch_size):
                end_idx = min(start_idx + batch_size, n_total)
                batch_size_actual = end_idx - start_idx
                
                # Get batch of coordinates (no JAX conversion yet)
                batch_coords_np = coords_full[start_idx:end_idx]
                
                # Convert to JAX, compute, and immediately convert back
                batch_coords_jax = jnp.array(batch_coords_np)
                eigenvectors_batch_jax = compute_rff_batch(batch_coords_jax)
                eigenvectors_batch_np = np.array(eigenvectors_batch_jax)
                
                # Free JAX memory immediately
                del batch_coords_jax, eigenvectors_batch_jax
                
                # Store numpy result
                eigenvectors_batches.append(eigenvectors_batch_np)
                
                # Log progress
                logger.info(f"  Processed {end_idx:,}/{n_total:,} points ({100*end_idx/n_total:.1f}%)")
                
                # Force garbage collection every few batches
                if len(eigenvectors_batches) % 5 == 0:
                    import gc
                    gc.collect()
            
            # Concatenate all batches
            logger.info("Concatenating batch results...")
            eigenvectors_full = np.vstack(eigenvectors_batches)
            
            logger.info(f"RFF interpolation complete: {n_total:,} × {n_eigenvectors} matrix")
            return eigenvectors_full
            
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            return None
    
    def get_mesf_eigenvectors_for_dataframe(
        self,
        df: 'pd.DataFrame',
        n_eigenvectors: int = 10,
        distance_threshold: float = 100.0
    ) -> Optional[np.ndarray]:
        """Compute MESF eigenvectors for a pandas DataFrame.
        
        This method is for Julia integration - takes a DataFrame with x,y columns
        and returns numpy array of eigenvectors.
        
        Args:
            df: DataFrame with x, y columns
            n_eigenvectors: Number of eigenvectors
            distance_threshold: Distance threshold in km
            
        Returns:
            Numpy array of eigenvectors (n_rows, n_eigenvectors) or None
        """
        import pandas as pd
        
        if not isinstance(df, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return None
        
        if 'x' not in df.columns or 'y' not in df.columns:
            logger.error("DataFrame must have 'x' and 'y' columns")
            return None
        
        # Extract coordinates
        coords = df[['x', 'y']].values.astype(np.float32)
        n_total = len(coords)
        
        logger.info(f"Computing MESF for DataFrame with {n_total:,} rows")
        
        # Use simplified computation for DataFrame input
        n_landmark = min(5000, n_total)  # Limit landmarks for memory
        n_sample = min(50000, n_total)   # Limit samples
        
        # Random sampling
        if n_sample < n_total:
            sample_idx = np.random.choice(n_total, n_sample, replace=False)
            coords_sample = coords[sample_idx]
        else:
            coords_sample = coords
        
        # Landmarks are subset of samples
        coords_landmark = coords_sample[:n_landmark]
        
        # Compute eigenvectors on landmarks
        eigenvectors_landmark = self._compute_landmark_eigenvectors(
            coords_landmark, n_eigenvectors, distance_threshold
        )
        
        if eigenvectors_landmark is None:
            return None
        
        # Extend to samples if needed
        if n_sample > n_landmark:
            eigenvectors_sample = self._extend_to_samples(
                coords_sample, coords_landmark, eigenvectors_landmark
            )
        else:
            eigenvectors_sample = eigenvectors_landmark
        
        # Interpolate to full dataset using k-NN
        if n_total > n_sample:
            from sklearn.neighbors import NearestNeighbors
            
            # Fit k-NN on samples
            knn = NearestNeighbors(n_neighbors=min(10, n_sample))
            knn.fit(coords_sample)
            
            # Find nearest neighbors for all points
            distances, indices = knn.kneighbors(coords)
            
            # Weighted average of eigenvectors
            weights = 1.0 / (distances + 1e-10)
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            eigenvectors_full = np.zeros((n_total, n_eigenvectors), dtype=np.float32)
            for i in range(n_total):
                eigenvectors_full[i] = (weights[i:i+1] @ eigenvectors_sample[indices[i]]).squeeze()
        else:
            eigenvectors_full = eigenvectors_sample
        
        logger.info(f"MESF computation complete: {eigenvectors_full.shape}")
        return eigenvectors_full
    
    def _create_numpy_array(self, eigenvectors: np.ndarray) -> np.ndarray:
        """Return numpy array with proper shape.
        
        For Julia integration, we return numpy arrays instead of H2O frames.
        
        Args:
            eigenvectors: Eigenvectors array (n_total, n_eigenvectors)
            
        Returns:
            H2O frame with ME1, ME2, ..., ME10 columns
        """
        import pandas as pd
        import tempfile
        import os
        
        n_eigenvectors = eigenvectors.shape[1]
        col_names = [f'ME{i+1}' for i in range(n_eigenvectors)]
        
        # FAST PATH: Use pandas->parquet->H2O for parallel import
        logger.info(f"Converting {eigenvectors.shape[0]:,} MESF eigenvectors to H2O using parallel import...")
        
        # Create temporary parquet file
        with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Convert to pandas DataFrame (fast)
            df = pd.DataFrame(eigenvectors, columns=col_names)
            
            # Write to parquet (fast, compressed)
            df.to_parquet(tmp_path, compression='snappy', engine='pyarrow')
            logger.info(f"Wrote MESF to temporary parquet: {os.path.getsize(tmp_path)/1e9:.2f} GB")
            
            # Import using H2O's parallel reader (USES ALL CORES!)
            mesf_h2o = h2o.import_file(tmp_path)
            mesf_h2o.set_names(col_names)
            logger.info("MESF imported to H2O successfully using parallel import")
            
            return mesf_h2o
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        # Delegate to cache manager
        return self.cache_manager.get_cache_info()


def compute_mesf_cli(
    data_path: str,
    spatial_grid_step: int = 1,
    n_eigenvectors: int = 10,
    n_jobs: int = 120,
    output_dir: str = 'results/mesf_cache',
    filename_step: int = None
) -> str:
    """CLI-friendly MESF computation for Julia integration.
    
    Args:
        data_path: Path to arrow file
        spatial_grid_step: Spatial sampling (1 = no sampling)
        n_eigenvectors: Number of MESF eigenvectors
        n_jobs: Number of CPU cores to use
        output_dir: Directory to save results
        filename_step: Step to use in filename (if different from spatial_grid_step)
        
    Returns:
        Path to saved MESF file
    """
    print("="*60)
    print(f"COMPUTING MESF WITH SPATIAL_STEPS={spatial_grid_step}")
    print("="*60)
    print(f"Data: {data_path}")
    print(f"Parameters: n_eigenvectors={n_eigenvectors}, n_jobs={n_jobs}")
    
    # No thread control here - taskset enforces CPU limits
    logger.info(f"Configured for {n_jobs} cores (taskset enforced)")
    
    # Initialize processor with output directory as cache directory
    processor = MESFProcessor(n_jobs=n_jobs, cache_dir=output_dir)
    
    # Compute MESF (will use cache if available)
    mesf_eigenvectors = processor.get_mesf_eigenvectors(
        data_path=data_path,
        spatial_grid_step=spatial_grid_step,
        n_eigenvectors=n_eigenvectors,
        distance_threshold=100.0,
        n_jobs=n_jobs
    )
    
    if mesf_eigenvectors is not None:
        # Save result
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use filename_step if provided, otherwise use spatial_grid_step
        step_for_filename = filename_step if filename_step is not None else spatial_grid_step
        output_file = output_dir / f'mesf_step{step_for_filename}_n{n_eigenvectors}.npy'
        np.save(output_file, mesf_eigenvectors)
        
        print(f"\n✓ MESF computation complete!")
        print(f"✓ Saved to: {output_file}")
        print(f"Shape: {mesf_eigenvectors.shape}")
        print(f"File size: {output_file.stat().st_size / 1024**3:.2f} GB")
        
        # Print the path for Julia to read
        print(output_file)  # Last line is the file path
        return str(output_file)
    else:
        print("ERROR: MESF computation failed")
        sys.exit(1)


def main():
    """Main entry point for CLI usage from Julia."""
    parser = argparse.ArgumentParser(
        description='Compute MESF eigenvectors for Julia hybrid pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python mesf_processor.py /data/bio.parquet
  
  # With spatial sampling and custom parameters
  python mesf_processor.py /data/bio.parquet --spatial-step 10 --n-eigenvectors 15
  
  # Specify output directory
  python mesf_processor.py /data/bio.parquet --output-dir /cache/mesf
        """
    )
    
    parser.add_argument('data_path', help='Path to parquet data file')
    parser.add_argument('--spatial-step', type=int, default=1,
                       help='Spatial grid sampling step (default: 1)')
    parser.add_argument('--n-eigenvectors', type=int, default=10,
                       help='Number of eigenvectors (default: 10)')
    parser.add_argument('--n-jobs', type=int, default=120,
                       help='Number of CPU cores (default: 120)')
    parser.add_argument('--output-dir', default='results/mesf_cache',
                       help='Output directory (default: results/mesf_cache)')
    
    args = parser.parse_args()
    
    # Compute MESF (no sampling in Python, Julia handles it)
    output_file = compute_mesf_cli(
        data_path=args.data_path,
        spatial_grid_step=1,  # No sampling in Python
        n_eigenvectors=args.n_eigenvectors,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        filename_step=args.spatial_step  # Use provided step for filename
    )


if __name__ == "__main__":
    main()
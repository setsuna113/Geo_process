"""Clean MESF processor with Nystrom-only implementation.

Key design decisions:
1. Always use Nystrom (no fallback to exact MESF)
2. Use 100×√n samples for optimal quality/memory balance
3. Load only samples' x,y coordinates with Polars
4. Efficient k-NN interpolation to full dataset
5. No redundant configs or logic
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import h2o
import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

# Import abstracted spatial computation modules
from src.machine_learning.spatial.nystrom_eigensolver import NystromEigensolver
# No spatial_weights - we inline the simple computation

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
        logger.info(f"MESF processor initialized with cache at {self.cache_dir}")
    
    def get_mesf_eigenvectors(
        self,
        data_path: str,
        spatial_grid_step: Optional[int] = None,
        n_eigenvectors: int = 10,
        distance_threshold: float = 100.0,
        n_jobs: Optional[int] = None
    ) -> Optional[h2o.H2OFrame]:
        """Compute MESF eigenvectors using Nystrom approximation.
        
        This is the main entry point called by the orchestrator.
        
        Args:
            data_path: Path to parquet file
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
        
        # Use configured core limit from n_jobs
        import os
        if n_jobs == -1:
            n_cores = os.cpu_count() or 150
        else:
            n_cores = n_jobs  # Use the configured value
        
        # Set JAX to use limited cores
        os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_cores}'
        
        logger.info(f"Using {n_cores} CPU cores for parallel computations")
        
        # Generate cache key
        cache_key = self._get_cache_key(data_path, spatial_grid_step, n_eigenvectors, distance_threshold)
        logger.info(f"Generated cache key: {cache_key}")
        
        # Try cache first
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            logger.info("✅ Successfully loaded MESF eigenvectors from cache!")
            return cached
        else:
            logger.info("Cache not found or failed to load, will compute MESF")
        
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
        
        # Convert to H2O frame with proper column names
        mesf_h2o = self._create_h2o_frame(eigenvectors_full)
        
        # Cache the result
        self._save_to_cache(cache_key, eigenvectors_full)
        
        logger.info(f"MESF computation complete: {n_total:,} rows × {n_eigenvectors} eigenvectors")
        return mesf_h2o
    
    def _count_total_rows(self, data_path: str, spatial_grid_step: Optional[int]) -> int:
        """Count total rows efficiently using Polars.
        
        Args:
            data_path: Path to parquet file
            spatial_grid_step: Sampling step
            
        Returns:
            Total number of rows after sampling
        """
        try:
            # Use Polars lazy evaluation to count rows efficiently
            df = pl.scan_parquet(data_path)
            total_rows = df.select(pl.count()).collect()[0, 0]
            
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
        # 4×√n landmarks for eigendecomposition
        # With 51M points: 4×7204 = 28,816 landmarks
        n_landmark = int(4 * np.sqrt(n_total))
        
        # 200×√n samples for RFF reference points
        # With 51M points: 200×7204 = 1,440,800 samples
        n_sample = int(200 * np.sqrt(n_total))
        
        # Apply reasonable bounds
        n_landmark = max(1000, min(50000, n_landmark))  # Cap at 50k for memory
        n_sample = max(10000, min(2000000, n_sample))  # Cap at 2M for memory
        
        logger.info(f"Nystrom parameters: {n_landmark:,} landmarks (4×√{n_total:,}), "
                   f"{n_sample:,} samples (200×√{n_total:,}) for RFF")
        
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
            data_path: Path to parquet file
            indices: Row indices to load (accounts for spatial sampling)
            spatial_grid_step: Sampling step
            
        Returns:
            Coordinates array (n_samples, 2) or None if error
        """
        try:
            # Load all x,y coordinates with Polars
            coords_df = pl.read_parquet(
                data_path,
                columns=['x', 'y'],
                use_pyarrow=True,
                memory_map=True
            )
            
            # Apply spatial sampling if needed
            if spatial_grid_step and spatial_grid_step > 1:
                # For DataFrame, use slice with step
                coords_df = coords_df[::spatial_grid_step]
            
            # Remove nulls
            coords_df = coords_df.drop_nulls()
            
            # Select only the sample indices using iloc-style indexing
            coords_df = coords_df[indices]
            
            # Convert to numpy
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
            data_path: Path to parquet file
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
            
            # Configure JAX
            import os
            n_cores = self.n_jobs if self.n_jobs else (os.cpu_count() or 150)
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_cores}'
            
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
            coords_df = pl.read_parquet(
                data_path,
                columns=['x', 'y'],
                use_pyarrow=True,
                memory_map=True
            )
            
            if spatial_grid_step and spatial_grid_step > 1:
                coords_df = coords_df[::spatial_grid_step]
            
            coords_df = coords_df.drop_nulls()
            coords_full = coords_df.to_numpy().astype(np.float32)
            
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
    
    def _create_h2o_frame(self, eigenvectors: np.ndarray) -> h2o.H2OFrame:
        """Create H2O frame with proper column names.
        
        LINUS FIX: Use parquet->H2O for parallel import instead of slow numpy->H2O.
        
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
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
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
    
    def _get_cache_key(
        self,
        data_path: str,
        spatial_grid_step: Optional[int],
        n_eigenvectors: int,
        distance_threshold: float
    ) -> str:
        """Generate unique cache key.
        
        Args:
            data_path: Path to data file
            spatial_grid_step: Sampling parameter
            n_eigenvectors: Number of eigenvectors
            distance_threshold: Distance threshold
            
        Returns:
            Unique cache key string
        """
        key_data = {
            'file': str(Path(data_path).resolve()),
            'sampling': spatial_grid_step or 0,
            'n_eigen': n_eigenvectors,
            'dist_thresh': distance_threshold
        }
        
        # Add file modification time
        try:
            stat = Path(data_path).stat()
            key_data['mtime'] = stat.st_mtime
            key_data['size'] = stat.st_size
        except:
            pass
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[h2o.H2OFrame]:
        """Load cached result if exists.
        
        Args:
            cache_key: Cache key
            
        Returns:
            H2O frame or None if not cached
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"
        logger.info(f"Looking for cache file: {cache_file}")
        
        if not cache_file.exists():
            logger.info(f"Cache file does not exist: {cache_file}")
            return None
        
        try:
            logger.info(f"Loading cache file: {cache_file} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
            data = np.load(cache_file)
            eigenvectors = data['eigenvectors']
            logger.info(f"Loaded numpy array from cache: {eigenvectors.shape}")
            h2o_frame = self._create_h2o_frame(eigenvectors)
            logger.info(f"Created H2O frame: {h2o_frame.shape}")
            return h2o_frame
        except Exception as e:
            logger.error(f"Cache load failed: {e}", exc_info=True)
            return None
    
    def _save_to_cache(self, cache_key: str, eigenvectors: np.ndarray) -> None:
        """Save result to cache.
        
        Args:
            cache_key: Cache key
            eigenvectors: Eigenvectors to cache
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.npz"
            np.savez_compressed(cache_file, eigenvectors=eigenvectors)
            logger.info(f"Cached result: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        cache_files = list(self.cache_dir.glob("*.npz"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024)
        }
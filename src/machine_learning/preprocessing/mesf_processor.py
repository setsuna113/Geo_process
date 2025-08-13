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
from src.machine_learning.spatial.spatial_weights import compute_knn_weights

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
        
        logger.info(f"Using {n_jobs if n_jobs != -1 else 'all'} CPU cores for k-NN")
        
        # Generate cache key
        cache_key = self._get_cache_key(data_path, spatial_grid_step, n_eigenvectors, distance_threshold)
        
        # Try cache first
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            logger.info("Loaded MESF eigenvectors from cache")
            return cached
        
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
        
        # Step 5: Extend eigenvectors to all samples
        eigenvectors_sample = self._extend_to_samples(
            coords_sample, coords_landmark, eigenvectors_landmark
        )
        
        # Step 6: Interpolate to full dataset using k-NN
        eigenvectors_full = self._interpolate_to_full(
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
                # Calculate sampled count
                sampled_rows = total_rows // spatial_grid_step
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
        
        Uses √n rule for landmarks and 100× for samples.
        
        Args:
            n_total: Total number of data points
            
        Returns:
            (n_landmark, n_sample) tuple
        """
        # √n rule for landmarks
        n_landmark = int(np.sqrt(n_total))
        
        # 100× landmarks for samples (optimal balance)
        n_sample = min(100 * n_landmark, n_total // 2)
        
        # Apply reasonable bounds
        n_landmark = max(1000, min(50000, n_landmark))
        n_sample = max(10000, min(1000000, n_sample))
        
        logger.info(f"Nystrom parameters: {n_landmark:,} landmarks (√{n_total:,}), "
                   f"{n_sample:,} samples ({n_sample/n_landmark:.0f}× landmarks)")
        
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
                coords_df = coords_df.take_every(spatial_grid_step)
            
            # Remove nulls
            coords_df = coords_df.drop_nulls()
            
            # Select only the sample indices
            coords_df = coords_df.take(indices)
            
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
    
    def _interpolate_to_full(
        self,
        data_path: str,
        spatial_grid_step: Optional[int],
        n_total: int,
        coords_sample: np.ndarray,
        eigenvectors_sample: np.ndarray,
        n_jobs: int = -1
    ) -> Optional[np.ndarray]:
        """Interpolate eigenvectors to full dataset using k-NN.
        
        Process in batches to avoid memory issues.
        
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
            logger.info(f"Interpolating to {n_total:,} points using k-NN...")
            
            # Build k-NN index from samples
            k = min(50, len(coords_sample))
            coords_sample_rad = np.radians(coords_sample[:, [1, 0]])  # [lat, lon] for sklearn
            
            # Use configured n_jobs for parallelization
            knn = NearestNeighbors(n_neighbors=k, metric='haversine', n_jobs=n_jobs)
            knn.fit(coords_sample_rad)
            logger.info(f"Built k-NN index with {k} neighbors using {n_jobs if n_jobs != -1 else 'all'} cores")
            
            # Process in batches
            batch_size = 100000
            n_eigenvectors = eigenvectors_sample.shape[1]
            eigenvectors_full = np.zeros((n_total, n_eigenvectors), dtype=np.float32)
            
            # Load all coordinates for interpolation
            coords_df = pl.read_parquet(
                data_path,
                columns=['x', 'y'],
                use_pyarrow=True,
                memory_map=True
            )
            
            if spatial_grid_step and spatial_grid_step > 1:
                coords_df = coords_df.take_every(spatial_grid_step)
            
            coords_df = coords_df.drop_nulls()
            coords_full = coords_df.to_numpy().astype(np.float32)
            coords_full_rad = np.radians(coords_full[:, [1, 0]])
            
            # Process in batches
            for start_idx in range(0, n_total, batch_size):
                end_idx = min(start_idx + batch_size, n_total)
                batch_coords = coords_full_rad[start_idx:end_idx]
                
                # Find nearest samples
                distances, indices = knn.kneighbors(batch_coords)
                
                # Use abstracted weight computation
                weights = compute_knn_weights(distances, method='inverse_distance')
                
                # Weighted interpolation
                for i in range(n_eigenvectors):
                    eigenvectors_full[start_idx:end_idx, i] = np.sum(
                        weights * eigenvectors_sample[indices, i], axis=1
                    )
                
                if (end_idx - start_idx) >= batch_size:
                    logger.info(f"  Processed {end_idx:,}/{n_total:,} points...")
            
            logger.info(f"Interpolation complete: {n_total:,} × {n_eigenvectors} matrix")
            return eigenvectors_full
            
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            return None
    
    def _create_h2o_frame(self, eigenvectors: np.ndarray) -> h2o.H2OFrame:
        """Create H2O frame with proper column names.
        
        Args:
            eigenvectors: Eigenvectors array (n_total, n_eigenvectors)
            
        Returns:
            H2O frame with ME1, ME2, ..., ME10 columns
        """
        n_eigenvectors = eigenvectors.shape[1]
        
        # Create H2O frame
        mesf_h2o = h2o.H2OFrame(eigenvectors)
        
        # Set column names
        col_names = [f'ME{i+1}' for i in range(n_eigenvectors)]
        mesf_h2o.set_names(col_names)
        
        return mesf_h2o
    
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
        
        if not cache_file.exists():
            return None
        
        try:
            data = np.load(cache_file)
            eigenvectors = data['eigenvectors']
            logger.info(f"Loaded cache: {eigenvectors.shape}")
            return self._create_h2o_frame(eigenvectors)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
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
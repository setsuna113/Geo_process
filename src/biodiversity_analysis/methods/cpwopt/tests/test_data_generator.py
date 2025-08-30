"""Synthetic test data generator for CP-WOPT testing."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from src.abstractions import TensorData


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 1000
    n_species: int = 20
    lat_range: Tuple[float, float] = (-45, 45)
    lon_range: Tuple[float, float] = (-180, 180)
    rank: int = 5
    noise_level: float = 0.1
    sparsity: float = 0.8
    missing_rate: float = 0.1
    seed: Optional[int] = None


class SyntheticBiodiversityGenerator:
    """Generate synthetic biodiversity data with known structure."""
    
    def __init__(self, config: SyntheticDataConfig):
        """Initialize generator with configuration."""
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
    
    def generate_structured_tensor(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate tensor with known CP structure.
        
        Returns:
            Tuple of (tensor, true_factors)
        """
        # Define tensor shape
        lat_bins = 20
        lon_bins = 20
        n_features = self.config.n_species
        
        # Generate true factor matrices
        true_factors = []
        
        # Latitude factors - different patterns
        lat_factor = np.zeros((lat_bins, self.config.rank))
        for r in range(self.config.rank):
            if r == 0:  # Latitudinal gradient
                lat_factor[:, r] = np.linspace(0, 1, lat_bins)
            elif r == 1:  # Tropical peak
                center = lat_bins // 2
                lat_factor[:, r] = np.exp(-0.5 * ((np.arange(lat_bins) - center) / 5)**2)
            elif r == 2:  # Temperate peaks
                lat_factor[:, r] = np.sin(np.linspace(0, 2*np.pi, lat_bins))
            else:  # Random patterns
                lat_factor[:, r] = np.random.rand(lat_bins)
        
        # Normalize
        for r in range(self.config.rank):
            lat_factor[:, r] /= np.linalg.norm(lat_factor[:, r])
        true_factors.append(lat_factor)
        
        # Longitude factors - different patterns
        lon_factor = np.zeros((lon_bins, self.config.rank))
        for r in range(self.config.rank):
            if r == 0:  # East-west gradient
                lon_factor[:, r] = np.linspace(1, 0, lon_bins)
            elif r == 1:  # Coastal preference
                lon_factor[:, r] = 0.5 * (
                    np.exp(-0.1 * np.arange(lon_bins)) + 
                    np.exp(-0.1 * (lon_bins - np.arange(lon_bins)))
                )
            else:  # Random patterns
                lon_factor[:, r] = np.random.rand(lon_bins)
        
        # Normalize
        for r in range(self.config.rank):
            lon_factor[:, r] /= np.linalg.norm(lon_factor[:, r])
        true_factors.append(lon_factor)
        
        # Feature (species) factors - ecological niches
        feature_factor = np.zeros((n_features, self.config.rank))
        for r in range(self.config.rank):
            # Each component represents different ecological niche
            if r < n_features:
                # Some species load strongly on specific components
                feature_factor[r::self.config.rank, r] = np.random.exponential(2, 
                    size=len(feature_factor[r::self.config.rank, r]))
            else:
                feature_factor[:, r] = np.random.exponential(0.5, n_features)
        
        # Normalize
        for r in range(self.config.rank):
            if np.linalg.norm(feature_factor[:, r]) > 0:
                feature_factor[:, r] /= np.linalg.norm(feature_factor[:, r])
        true_factors.append(feature_factor)
        
        # Construct tensor from factors
        tensor = np.zeros((lat_bins, lon_bins, n_features))
        for r in range(self.config.rank):
            # Outer product of factor columns
            component = np.outer(lat_factor[:, r], lon_factor[:, r])
            for f in range(n_features):
                tensor[:, :, f] += feature_factor[f, r] * component
        
        # Scale to reasonable count range
        tensor *= 100
        
        # Add noise
        noise = np.random.randn(*tensor.shape) * self.config.noise_level * np.mean(tensor)
        tensor += noise
        
        # Ensure non-negative
        tensor = np.maximum(tensor, 0)
        
        return tensor, true_factors
    
    def apply_sparsity(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sparsity pattern to tensor.
        
        Returns:
            Tuple of (sparse_tensor, weight_mask)
        """
        # Create sparsity mask
        mask = np.random.rand(*tensor.shape) > self.config.sparsity
        
        # Apply mask
        sparse_tensor = tensor.copy()
        sparse_tensor[~mask] = 0
        
        # Weight mask (1 where observed, 0 where missing)
        weights = mask.astype(float)
        
        return sparse_tensor, weights
    
    def generate_dataframe(self) -> pd.DataFrame:
        """Generate DataFrame format biodiversity data."""
        # Generate random sampling locations
        latitudes = np.random.uniform(
            self.config.lat_range[0], 
            self.config.lat_range[1], 
            self.config.n_samples
        )
        longitudes = np.random.uniform(
            self.config.lon_range[0],
            self.config.lon_range[1],
            self.config.n_samples
        )
        
        # Create base DataFrame
        df = pd.DataFrame({
            'latitude': latitudes,
            'longitude': longitudes
        })
        
        # Generate species counts based on location
        for i in range(self.config.n_species):
            # Different species have different spatial patterns
            if i % 5 == 0:  # Widespread species
                base_rate = 10
                counts = np.random.poisson(base_rate, self.config.n_samples)
            elif i % 5 == 1:  # Tropical species
                base_rate = 5 * np.exp(-0.1 * np.abs(latitudes))
                counts = np.random.poisson(base_rate)
            elif i % 5 == 2:  # Temperate species
                base_rate = 5 * np.exp(-0.05 * (np.abs(latitudes) - 30)**2)
                counts = np.random.poisson(np.maximum(base_rate, 0.1))
            elif i % 5 == 3:  # Rare species
                counts = np.random.poisson(0.5, self.config.n_samples)
            else:  # Endemic species (localized)
                # Create hotspots
                n_hotspots = np.random.randint(1, 4)
                counts = np.zeros(self.config.n_samples)
                for _ in range(n_hotspots):
                    center_lat = np.random.uniform(*self.config.lat_range)
                    center_lon = np.random.uniform(*self.config.lon_range)
                    distances = np.sqrt(
                        (latitudes - center_lat)**2 + 
                        (longitudes - center_lon)**2
                    )
                    local_counts = np.random.poisson(
                        10 * np.exp(-0.1 * distances)
                    )
                    counts += local_counts
            
            # Apply sparsity
            zero_mask = np.random.rand(self.config.n_samples) < self.config.sparsity
            counts[zero_mask] = 0
            
            # Add missing values
            missing_mask = np.random.rand(self.config.n_samples) < self.config.missing_rate
            counts = counts.astype(float)
            counts[missing_mask] = np.nan
            
            df[f'species_{i:03d}'] = counts
        
        return df
    
    def generate_tensor_data(self, lat_bins: int = 20, 
                           lon_bins: int = 20) -> TensorData:
        """Generate TensorData object with known structure."""
        # Generate structured tensor
        tensor, true_factors = self.generate_structured_tensor()
        
        # Apply sparsity
        sparse_tensor, weights = self.apply_sparsity(tensor)
        
        # Create feature names
        feature_names = [f'species_{i:03d}' for i in range(self.config.n_species)]
        
        # Create metadata
        metadata = {
            'true_rank': self.config.rank,
            'true_factors': true_factors,
            'generation_config': {
                'n_samples': self.config.n_samples,
                'noise_level': self.config.noise_level,
                'sparsity': self.config.sparsity
            },
            'lat_bins': lat_bins,
            'lon_bins': lon_bins,
            'lat_range': self.config.lat_range,
            'lon_range': self.config.lon_range
        }
        
        # Create TensorData
        tensor_data = TensorData(
            tensor=sparse_tensor,
            weights=weights,
            coordinates=[],  # Not used for grid data
            feature_names=feature_names,
            metadata=metadata,
            is_sparse=self.config.sparsity > 0.5
        )
        
        return tensor_data
    
    def generate_challenging_cases(self) -> Dict[str, TensorData]:
        """Generate challenging test cases."""
        cases = {}
        
        # Case 1: Extremely sparse data
        config_sparse = SyntheticDataConfig(
            n_species=50,
            rank=3,
            sparsity=0.95,
            noise_level=0.2,
            seed=42
        )
        gen_sparse = SyntheticBiodiversityGenerator(config_sparse)
        cases['extreme_sparsity'] = gen_sparse.generate_tensor_data()
        
        # Case 2: High noise
        config_noisy = SyntheticDataConfig(
            n_species=20,
            rank=5,
            sparsity=0.7,
            noise_level=0.5,
            seed=43
        )
        gen_noisy = SyntheticBiodiversityGenerator(config_noisy)
        cases['high_noise'] = gen_noisy.generate_tensor_data()
        
        # Case 3: Many components
        config_complex = SyntheticDataConfig(
            n_species=30,
            rank=10,
            sparsity=0.8,
            noise_level=0.1,
            seed=44
        )
        gen_complex = SyntheticBiodiversityGenerator(config_complex)
        cases['many_components'] = gen_complex.generate_tensor_data()
        
        # Case 4: Imbalanced modes
        config_imbalanced = SyntheticDataConfig(
            n_species=100,  # Many more species than spatial bins
            rank=5,
            sparsity=0.85,
            noise_level=0.15,
            seed=45
        )
        gen_imbalanced = SyntheticBiodiversityGenerator(config_imbalanced)
        cases['imbalanced_modes'] = gen_imbalanced.generate_tensor_data(
            lat_bins=10, lon_bins=10
        )
        
        # Case 5: No structure (pure noise)
        config_noise_only = SyntheticDataConfig(
            n_species=20,
            rank=0,  # Will create random tensor
            sparsity=0.8,
            noise_level=1.0,
            seed=46
        )
        gen_noise = SyntheticBiodiversityGenerator(config_noise_only)
        noise_tensor = np.random.rand(20, 20, 20)
        noise_sparse, noise_weights = gen_noise.apply_sparsity(noise_tensor)
        cases['no_structure'] = TensorData(
            tensor=noise_sparse,
            weights=noise_weights,
            coordinates=[],
            feature_names=[f'species_{i:03d}' for i in range(20)],
            metadata={'description': 'Pure noise, no CP structure'},
            is_sparse=True
        )
        
        return cases


def create_validation_dataset() -> Tuple[TensorData, CPFactors]:
    """Create dataset with known ground truth for validation.
    
    Returns:
        Tuple of (tensor_data, true_factors)
    """
    from src.abstractions import CPFactors
    
    # Create simple, interpretable structure
    lat_bins = 15
    lon_bins = 15
    n_species = 10
    rank = 3
    
    # Factor 1: North-South gradient
    lat_factor_1 = np.linspace(0, 1, lat_bins)
    lon_factor_1 = np.ones(lon_bins)
    species_factor_1 = np.array([1, 1, 1, 0.5, 0.5, 0, 0, 0, 0, 0])
    
    # Factor 2: East-West gradient  
    lat_factor_2 = np.ones(lat_bins)
    lon_factor_2 = np.linspace(1, 0, lon_bins)
    species_factor_2 = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1, 0, 0])
    
    # Factor 3: Central hotspot
    lat_factor_3 = np.exp(-0.5 * ((np.arange(lat_bins) - lat_bins//2) / 3)**2)
    lon_factor_3 = np.exp(-0.5 * ((np.arange(lon_bins) - lon_bins//2) / 3)**2)
    species_factor_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    
    # Combine into factor matrices
    lat_factors = np.column_stack([lat_factor_1, lat_factor_2, lat_factor_3])
    lon_factors = np.column_stack([lon_factor_1, lon_factor_2, lon_factor_3])
    species_factors = np.column_stack([species_factor_1, species_factor_2, species_factor_3])
    
    # Normalize
    for i in range(rank):
        lat_factors[:, i] /= np.linalg.norm(lat_factors[:, i])
        lon_factors[:, i] /= np.linalg.norm(lon_factors[:, i])
        species_factors[:, i] /= np.linalg.norm(species_factors[:, i])
    
    # Construct tensor
    tensor = np.zeros((lat_bins, lon_bins, n_species))
    for r in range(rank):
        component = np.outer(lat_factors[:, r], lon_factors[:, r])
        for s in range(n_species):
            tensor[:, :, s] += species_factors[s, r] * component * 50
    
    # Add small noise
    tensor += np.random.randn(*tensor.shape) * 0.5
    tensor = np.maximum(tensor, 0)
    
    # Create sparse version
    mask = np.random.rand(*tensor.shape) > 0.7  # 70% sparse
    sparse_tensor = tensor.copy()
    sparse_tensor[~mask] = 0
    
    # Create TensorData
    tensor_data = TensorData(
        tensor=sparse_tensor,
        weights=mask.astype(float),
        coordinates=[],
        feature_names=[f'species_{i}' for i in range(n_species)],
        metadata={
            'lat_bins': lat_bins,
            'lon_bins': lon_bins,
            'true_rank': rank
        },
        is_sparse=True
    )
    
    # Create true CPFactors
    true_factors = CPFactors(
        factors=[lat_factors, lon_factors, species_factors],
        rank=rank,
        mode_names=['latitude', 'longitude', 'features'],
        feature_names=tensor_data.feature_names,
        metadata={'ground_truth': True}
    )
    
    return tensor_data, true_factors


if __name__ == '__main__':
    # Example usage
    config = SyntheticDataConfig(
        n_samples=5000,
        n_species=30,
        rank=5,
        sparsity=0.85,
        seed=42
    )
    
    generator = SyntheticBiodiversityGenerator(config)
    
    # Generate DataFrame
    df = generator.generate_dataframe()
    print(f"Generated DataFrame with shape: {df.shape}")
    print(f"Sparsity: {(df.iloc[:, 2:] == 0).sum().sum() / (df.shape[0] * (df.shape[1] - 2)):.2%}")
    
    # Generate TensorData
    tensor_data = generator.generate_tensor_data()
    print(f"\nGenerated TensorData with shape: {tensor_data.shape}")
    print(f"Tensor sparsity: {(tensor_data.tensor == 0).sum() / tensor_data.tensor.size:.2%}")
    
    # Generate challenging cases
    cases = generator.generate_challenging_cases()
    print(f"\nGenerated {len(cases)} challenging test cases:")
    for name, data in cases.items():
        print(f"  {name}: shape={data.shape}, sparsity={(data.tensor == 0).sum() / data.tensor.size:.2%}")
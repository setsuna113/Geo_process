# Sparse Clustering Implementation Guide

## 1. Tensor Factorization (CP-WOPT)

### Algorithm Structure
- **Core**: CANDECOMP/PARAFAC (CP) decomposition with weighted optimization
- **Objective**: min ||W ⊙ (X - [[A, B, C]])||² where W is weight tensor (0 for missing, 1 for observed)
- **Optimization**: First-order methods (L-BFGS-B) or alternating least squares

### Key Implementation Points
```python
import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np

# Data preparation
# Shape: (n_lat, n_lon, n_features=4)
tensor = np.zeros((n_lat, n_lon, 4))
weights = np.zeros_like(tensor)  # 0 for missing, 1 for observed

# CP decomposition with missing values
def cp_wopt(tensor, weights, rank, n_iter=100, tol=1e-6):
    # Initialize factors
    factors = tl.random.cp(tensor.shape, rank)
    
    # Optimization loop
    for iteration in range(n_iter):
        for mode in range(tensor.ndim):
            # Update factor for mode
            factors[mode] = weighted_update(tensor, weights, factors, mode)
        
        # Check convergence
        if check_convergence(factors, tol):
            break
    
    return factors
```

### Required Parameters
- `rank`: Number of components (typically 5-50)
- `n_iter`: Maximum iterations (100-1000)
- `tol`: Convergence tolerance (1e-6)
- `init`: Initialization method ('random', 'svd')
- `weights`: Binary mask for missing values

### Libraries
```bash
pip install tensorly
# Alternative: MATLAB Tensor Toolbox
# Alternative: sparse_dot_mkl for large sparse tensors
```

---

## 2. Nonnegative Spatial Factorization (NSF)

### Algorithm Structure
- **Core**: NMF with Gaussian Process (GP) prior on spatial factors
- **Objective**: Poisson/NB likelihood with spatial smoothness
- **Model**: Y ~ Poisson(L @ F.T) where L has GP prior

### Key Implementation Points
```python
import tensorflow as tf
import tensorflow_probability as tfp

class NSF:
    def __init__(self, n_factors, spatial_coords, likelihood='poisson'):
        self.n_factors = n_factors
        self.coords = spatial_coords
        self.likelihood = likelihood
        
    def build_model(self, Y_sparse):
        # Spatial factors with GP prior
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
            length_scale=self.length_scale
        )
        
        # Loadings matrix L with spatial structure
        L = tfp.distributions.GaussianProcess(
            kernel=kernel,
            index_points=self.coords
        )
        
        # Transform to non-negative
        L_positive = tf.nn.softplus(L)
        
        # Factor matrix F (non-spatial)
        F = tf.Variable(
            tf.random.uniform([self.n_factors, n_features], 0, 1)
        )
        
        # Likelihood
        if self.likelihood == 'poisson':
            Y_pred = tf.matmul(L_positive, F)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=Y_sparse, logits=tf.log(Y_pred)
            )
        
        return loss
```

### Required Parameters
- `n_factors`: Number of spatial factors (10-100)
- `length_scale`: GP kernel length scale (tune based on spatial extent)
- `likelihood`: 'poisson' or 'negative_binomial'
- `spatial_coords`: (n_locations, 2) array of lat/lon
- `learning_rate`: For optimization (0.001-0.1)
- `inducing_points`: For scalability (subset of locations)

### Libraries
```bash
# From GitHub
git clone https://github.com/willtownes/nsf-paper.git
pip install tensorflow tensorflow-probability

# Dependencies
pip install numpy scipy pandas matplotlib
```

---

## Integration Strategy

### Data Preprocessing
```python
# Common preprocessing for both methods
def prepare_biodiversity_data(df):
    # df columns: lat, lon, species1, species2, species3, species4
    
    # For Tensor Factorization
    lat_bins = np.linspace(-90, 90, n_lat)
    lon_bins = np.linspace(-180, 180, n_lon)
    tensor = create_3d_tensor(df, lat_bins, lon_bins)
    weights = (tensor != missing_value).astype(float)
    
    # For NSF
    coords = df[['lat', 'lon']].values
    features = df[['species1', 'species2', 'species3', 'species4']].values
    sparse_matrix = create_sparse_matrix(coords, features)
    
    return tensor, weights, sparse_matrix, coords
```

### Hybrid Approach
```python
def hybrid_clustering(data, n_clusters):
    # Step 1: Tensor factorization for global patterns
    factors = cp_wopt(data.tensor, data.weights, rank=n_clusters*2)
    
    # Step 2: Extract dense regions
    dense_mask = data.weights.sum(axis=2) > threshold
    
    # Step 3: Apply NSF to dense regions
    nsf_model = NSF(n_factors=n_clusters, 
                    spatial_coords=data.coords[dense_mask])
    nsf_factors = nsf_model.fit(data.features[dense_mask])
    
    # Step 4: Combine results
    global_clusters = reconstruct_clusters(factors)
    local_refinement = assign_nsf_clusters(nsf_factors)
    
    return merge_clustering(global_clusters, local_refinement)
```

### Key Hyperparameters to Tune
| Parameter | Tensor Factorization | NSF |
|-----------|---------------------|-----|
| Main | rank (5-50) | n_factors (10-100) |
| Spatial | grid_resolution | length_scale |
| Optimization | learning_rate, n_iter | learning_rate, inducing_points |
| Regularization | l1_reg, l2_reg | kernel_variance |

### Performance Considerations
- **Tensor**: Use sparse tensor libraries for >1M points
- **NSF**: Use inducing points approximation for >10K points
- **Memory**: Both methods can use mini-batching
- **GPU**: Both benefit from GPU acceleration
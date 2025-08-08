# Integrated KMeans-Hierarchical Implementation Plan

## Overview
Transform the deprecated k-means module into a two-stage kmeans-hierarchical clustering method that handles 44M biodiversity data points efficiently.

## Existing Resources to Utilize

### 1. **From K-means Module** (`src/biodiversity_analysis/methods/kmeans/`)
- `analyzer.py` - Base analyzer with data loading, preprocessing pipeline
- `analyzer_adapter.py` - IAnalyzer interface adapter
- `core.py` - Core k-means implementation (will extend for two-stage)
- `kmeans_config.py` - Configuration (will extend with hierarchical params)
- `visualization.py` - Visualization utilities (will extend for dendrograms)
- `adaptive_distance.py` - Distance calculations (reuse for hierarchical)
- `weight_calculator.py` - Feature weighting (applicable to both stages)

### 2. **Shared Infrastructure**
- `src/biodiversity_analysis/shared/base_analyzer.py` - Base analyzer class
- `src/biodiversity_analysis/shared/data.py` - ParquetLoader for data loading
- `src/biodiversity_analysis/preprocessing/pipeline.py` - SharedPreprocessor
- `src/config/biodiversity_config.py` - Configuration management
- `src/config/method_registry.py` - Method registration

### 3. **Data Pipeline**
- Already handles parquet files with 44M rows
- Existing preprocessing with missing data handling (85% complete cases)
- Coordinate tracking for geographic visualization

## Implementation Steps

### Step 1: Rename and Restructure Module
```bash
# Rename module directory
mv src/biodiversity_analysis/methods/kmeans \
   src/biodiversity_analysis/methods/kmeans_hierarchical

# Update module imports
```

### Step 2: Extend Configuration (`kmeans_hierarchical_config.py`)
```python
from dataclasses import dataclass
from .kmeans_config import KMeansConfig

@dataclass
class KMeansHierarchicalConfig(KMeansConfig):
    """Configuration for two-stage kmeans-hierarchical clustering."""
    
    # Stage 1: K-means parameters (inherited from KMeansConfig)
    # n_clusters becomes n_pre_clusters for clarity
    n_pre_clusters: int = 750  # Determined by variance preservation
    variance_threshold: float = 0.95  # For determining n_pre_clusters
    
    # Stage 2: Hierarchical parameters
    linkage_method: str = 'ward'  # ward, complete, average, single
    distance_metric_hierarchical: str = 'euclidean'  # Required for ward
    final_k_range: tuple = (4, 8)  # Range for final clusters
    
    # Cluster selection
    selection_method: str = 'bootstrap'  # bootstrap, silhouette, davies_bouldin
    n_bootstrap: int = 30  # For bootstrap stability
    
    # Computational optimization
    use_minibatch: bool = True  # For k-means stage
    batch_size: int = 10000
    max_memory_gb: float = 8.0  # Memory limit for hierarchical stage
    
    # Visualization
    generate_dendrogram: bool = True
    generate_geographic_maps: bool = True
    generate_3d_plots: bool = True
    generate_validation_plots: bool = True
```

### Step 3: Modify Core Implementation (`core.py` → `kmeans_hierarchical_core.py`)
```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from .core import BiodiversityKMeans  # Reuse existing k-means

class KMeansHierarchical:
    """Two-stage clustering: K-means → Hierarchical."""
    
    def __init__(self, config: KMeansHierarchicalConfig):
        self.config = config
        self.kmeans = None
        self.linkage_matrix = None
        self.final_labels = None
        self.scaler = RobustScaler()
        
    def fit(self, features: np.ndarray, coordinates: np.ndarray):
        """Fit two-stage clustering."""
        
        # Stage 0: Handle complete cases only
        complete_mask = ~np.isnan(features).any(axis=1)
        features_complete = features[complete_mask]
        coords_complete = coordinates[complete_mask]
        
        # Stage 0.5: Scale data
        features_scaled = self.scaler.fit_transform(features_complete)
        
        # Stage 1: Determine optimal k if needed
        if self.config.n_pre_clusters == 'auto':
            n_pre_clusters = self._determine_kmeans_k(
                features_scaled, 
                self.config.variance_threshold
            )
        else:
            n_pre_clusters = self.config.n_pre_clusters
            
        # Stage 1: K-means clustering
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_pre_clusters,
            batch_size=self.config.batch_size,
            init='k-means++',
            n_init=3,
            random_state=42
        )
        pre_labels = self.kmeans.fit_predict(features_scaled)
        
        # Stage 2: Hierarchical clustering on centroids
        centroids = self.kmeans.cluster_centers_
        self.linkage_matrix = linkage(
            centroids, 
            method=self.config.linkage_method
        )
        
        # Stage 3: Determine optimal final k
        if self.config.selection_method == 'bootstrap':
            optimal_k = self._bootstrap_selection(
                centroids, 
                self.config.final_k_range
            )
        else:
            optimal_k = self._metric_selection(
                features_scaled, 
                pre_labels,
                self.config.final_k_range
            )
        
        # Stage 4: Get final clusters
        hierarchical_labels = fcluster(
            self.linkage_matrix, 
            optimal_k, 
            criterion='maxclust'
        )
        
        # Map back to original points
        self.final_labels = hierarchical_labels[pre_labels]
        
        # Handle missing data points
        full_labels = np.full(len(features), -1)
        full_labels[complete_mask] = self.final_labels
        self.final_labels = full_labels
        
        return self
        
    def _determine_kmeans_k(self, features, variance_threshold):
        """Determine k that preserves variance_threshold."""
        # Implementation from our guide
        k_values = [100, 250, 500, 750, 1000, 1250, 1500]
        for k in k_values:
            # ... variance preservation logic
            pass
        return 750  # Placeholder
        
    def _bootstrap_selection(self, centroids, k_range):
        """Bootstrap stability test for optimal k."""
        # Implementation from our guide
        return 6  # Placeholder
```

### Step 4: Update Analyzer (`analyzer.py` → `kmeans_hierarchical_analyzer.py`)
```python
from .kmeans_hierarchical_core import KMeansHierarchical
from .kmeans_hierarchical_config import KMeansHierarchicalConfig
from .enhanced_visualization import KMeansHierarchicalVisualizer

class KMeansHierarchicalAnalyzer(BaseBiodiversityAnalyzer):
    """Two-stage kmeans-hierarchical analyzer."""
    
    def __init__(self):
        super().__init__(method_name='kmeans_hierarchical', version='2.0.0')
        self.clusterer = None
        self.visualizer = KMeansHierarchicalVisualizer()
        
    def analyze(self, data_path: str, **kwargs) -> AnalysisResult:
        """Perform two-stage clustering analysis."""
        
        # 1. Load data (reuse existing)
        biodiv_data = self.load_data(data_path)
        
        # 2. Preprocess (reuse existing pipeline)
        preprocessing_pipeline = BiodiversityPreprocessingPipeline(
            compositional_transform='hurdle',
            use_pmem=True,
            standardize=True,
            standardization_method='robust'
        )
        features_processed, feature_names, metadata = \
            preprocessing_pipeline.fit_transform(
                biodiv_data.features,
                biodiv_data.coordinates,
                biodiv_data.feature_names
            )
        
        # 3. Create config
        config = KMeansHierarchicalConfig(**kwargs)
        
        # 4. Fit two-stage clustering
        self.clusterer = KMeansHierarchical(config)
        self.clusterer.fit(features_processed, biodiv_data.coordinates)
        
        # 5. Calculate metrics (reuse + extend)
        quality_metrics = self._calculate_quality_metrics(
            features_processed,
            self.clusterer.final_labels,
            biodiv_data.coordinates
        )
        
        # 6. Generate heavy visualizations
        visualizations = self._generate_visualizations(
            biodiv_data,
            features_processed,
            self.clusterer,
            quality_metrics
        )
        
        # 7. Create result
        return self._create_result(
            biodiv_data,
            self.clusterer,
            quality_metrics,
            visualizations
        )
        
    def _generate_visualizations(self, biodiv_data, features, 
                                clusterer, metrics):
        """Generate comprehensive visualization suite."""
        viz = {}
        
        if self.config.generate_dendrogram:
            viz['dendrogram'] = self.visualizer.plot_dendrogram(
                clusterer.linkage_matrix,
                clusterer.config.final_k
            )
            
        if self.config.generate_geographic_maps:
            viz['geographic_map'] = self.visualizer.plot_geographic_clusters(
                biodiv_data.coordinates,
                clusterer.final_labels
            )
            viz['density_maps'] = self.visualizer.plot_cluster_density_maps(
                biodiv_data.coordinates,
                clusterer.final_labels
            )
            
        if self.config.generate_3d_plots:
            viz['3d_clusters'] = self.visualizer.plot_3d_clusters(
                features[:, :3],  # First 3 features
                clusterer.final_labels
            )
            
        if self.config.generate_validation_plots:
            viz['validation_metrics'] = self.visualizer.plot_validation_suite(
                metrics,
                clusterer.stability_scores
            )
            
        # Reuse existing visualizations
        viz['cluster_profiles'] = self.visualizer.plot_cluster_profiles(
            features,
            clusterer.final_labels,
            biodiv_data.feature_names
        )
        
        viz['parallel_coordinates'] = self.visualizer.plot_parallel_coordinates(
            features,
            clusterer.final_labels,
            biodiv_data.feature_names
        )
        
        return viz
```

### Step 5: Extend Visualization (`visualization.py` → `enhanced_visualization.py`)
```python
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from .visualization import KMeansVisualizer  # Inherit existing

class KMeansHierarchicalVisualizer(KMeansVisualizer):
    """Extended visualizer for two-stage clustering."""
    
    def plot_dendrogram(self, linkage_matrix, optimal_k):
        """Generate hierarchical dendrogram."""
        fig, ax = plt.subplots(figsize=(15, 8))
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=30,
            orientation='top',
            ax=ax
        )
        ax.axhline(y=self._get_cut_height(linkage_matrix, optimal_k),
                  color='r', linestyle='--', 
                  label=f'Cut for {optimal_k} clusters')
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Distance (Ward)')
        ax.legend()
        return fig
        
    def plot_geographic_clusters(self, coordinates, labels):
        """Interactive geographic cluster map."""
        import pandas as pd
        
        # Sample for performance
        n_sample = min(100000, len(coordinates))
        idx = np.random.choice(len(coordinates), n_sample, replace=False)
        
        df = pd.DataFrame({
            'lon': coordinates[idx, 1],
            'lat': coordinates[idx, 0],
            'cluster': labels[idx]
        })
        
        fig = px.scatter_geo(
            df,
            lon='lon',
            lat='lat',
            color='cluster',
            title='Global Biodiversity Clusters',
            color_discrete_sequence=px.colors.qualitative.Set1,
            projection='natural earth'
        )
        
        return fig
        
    def plot_3d_clusters(self, features_3d, labels):
        """Interactive 3D cluster visualization."""
        # Sample for performance
        n_sample = min(50000, len(features_3d))
        idx = np.random.choice(len(features_3d), n_sample, replace=False)
        
        fig = px.scatter_3d(
            x=features_3d[idx, 0],
            y=features_3d[idx, 1],
            z=features_3d[idx, 2],
            color=labels[idx],
            title='3D Biodiversity Clusters',
            labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3'}
        )
        
        return fig
        
    def plot_validation_suite(self, metrics, stability_scores):
        """Comprehensive validation metrics visualization."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Silhouette Scores', 'Davies-Bouldin Scores',
                          'Bootstrap Stability', 'Cluster Sizes')
        )
        
        # Add traces for each metric
        # ... implementation details
        
        return fig
        
    def plot_cluster_density_maps(self, coordinates, labels):
        """Density heatmaps per cluster."""
        # Implementation for geographic density visualization
        pass
        
    def plot_parallel_coordinates(self, features, labels, feature_names):
        """Parallel coordinates for feature relationships."""
        # Reuse and enhance existing implementation
        pass
```

### Step 6: Update Adapter (`analyzer_adapter.py` → `kmeans_hierarchical_adapter.py`)
```python
from .kmeans_hierarchical_analyzer import KMeansHierarchicalAnalyzer

class KMeansHierarchicalAnalyzerAdapter(IAnalyzer):
    """Adapter for KMeansHierarchicalAnalyzer."""
    
    def __init__(self, config: Any, db: Any):
        self._analyzer = KMeansHierarchicalAnalyzer()
        self._config = config
        self._db = db
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'n_pre_clusters': 'auto',
            'variance_threshold': 0.95,
            'linkage_method': 'ward',
            'final_k_range': (4, 8),
            'selection_method': 'bootstrap',
            'n_bootstrap': 30,
            'generate_dendrogram': True,
            'generate_geographic_maps': True,
            'save_results': True,
            'output_dir': './outputs/kmeans_hierarchical_results'
        }
        
    # Rest of adapter implementation...
```

### Step 7: Register Method in Configuration
```python
# In src/config/__init__.py or initialization
from src.biodiversity_analysis.methods.kmeans_hierarchical import (
    KMeansHierarchicalAnalyzerAdapter
)
from src.config.method_registry import MethodConfigRegistry

# Register the new method
MethodConfigRegistry.register(
    'kmeans_hierarchical',
    KMeansHierarchicalConfigLoader
)

# Add to available methods
AVAILABLE_METHODS = [
    'som', 
    'cpwopt', 
    'kmeans_hierarchical',  # Replaces 'kmeans'
    'gwpca',
    'maxp',
    'stat'
]
```

### Step 8: Create Config Loader
```python
# src/config/kmeans_hierarchical/kmeans_hierarchical_config_loader.py
from src.abstractions.interfaces.method_config import IMethodConfig

class KMeansHierarchicalConfigLoader(IMethodConfig):
    """Config loader for kmeans-hierarchical method."""
    
    def load_config(self, config_name=None, preset=None):
        """Load configuration for kmeans-hierarchical."""
        base_config = {
            'method_params': {
                'n_pre_clusters': 'auto',
                'variance_threshold': 0.95,
                'linkage_method': 'ward',
                'final_k_range': (4, 8),
                'selection_method': 'bootstrap',
                'n_bootstrap': 30
            },
            'data_processing': {
                'handle_missing': 'complete_cases',
                'normalization': 'robust',
                'remove_outliers': False
            },
            'visualization': {
                'generate_dendrogram': True,
                'generate_geographic_maps': True,
                'generate_3d_plots': True,
                'generate_validation_plots': True
            }
        }
        
        # Apply presets if provided
        if preset == 'fast':
            base_config['method_params']['n_bootstrap'] = 10
            base_config['visualization']['generate_3d_plots'] = False
        elif preset == 'publication':
            base_config['method_params']['n_bootstrap'] = 100
            base_config['visualization']['dpi'] = 300
            
        return base_config
```

## File Structure After Implementation

```
src/biodiversity_analysis/methods/kmeans_hierarchical/
├── __init__.py                          # Module exports
├── analyzer_adapter.py                  # IAnalyzer adapter (modified)
├── kmeans_hierarchical_analyzer.py      # Main analyzer (renamed/modified)
├── kmeans_hierarchical_core.py          # Two-stage clustering core (new)
├── kmeans_hierarchical_config.py        # Extended configuration (new)
├── enhanced_visualization.py            # Extended visualizations (new)
├── adaptive_distance.py                 # Distance calculations (reused)
├── weight_calculator.py                 # Feature weighting (reused)
└── sparse_optimizer.py                  # Optimization utilities (reused)
```

## Integration Points

### 1. **Data Loading**
- Reuses existing ParquetLoader
- Handles 44M rows efficiently
- Maintains coordinate tracking

### 2. **Preprocessing**
- Uses BiodiversityPreprocessingPipeline
- RobustScaler for outliers
- Complete cases handling (85% retention)

### 3. **Configuration**
- Integrates with biodiversity_config.py
- Registered in method_registry
- Supports presets and overrides

### 4. **Visualization**
- Extends existing KMeansVisualizer
- Adds hierarchical-specific plots
- Heavy visualization focus as requested

### 5. **Results**
- Compatible with AnalysisResult interface
- Saves to parquet/CSV formats
- Generates interpretation reports

## Testing Strategy

```python
# Test file: tests/test_kmeans_hierarchical.py
def test_two_stage_clustering():
    """Test two-stage clustering on sample data."""
    analyzer = KMeansHierarchicalAnalyzer()
    result = analyzer.analyze(
        'data/test_biodiversity.parquet',
        n_pre_clusters=100,  # Smaller for testing
        final_k_range=(3, 5)
    )
    assert result.labels is not None
    assert len(np.unique(result.labels)) <= 5
    
def test_variance_preservation():
    """Test k selection by variance preservation."""
    # Test implementation
    
def test_bootstrap_stability():
    """Test bootstrap stability selection."""
    # Test implementation
```

## Migration Path

1. **Phase 1**: Copy kmeans directory to kmeans_hierarchical
2. **Phase 2**: Implement two-stage core algorithm
3. **Phase 3**: Extend configuration and visualization
4. **Phase 4**: Update registration and adapters
5. **Phase 5**: Test with sample data
6. **Phase 6**: Full-scale test with 44M points
7. **Phase 7**: Deprecate original kmeans module

## Performance Considerations

- **Memory**: ~8-10 GB for 44M points
- **Time**: ~30-60 minutes total
  - K-means stage: 15-30 minutes
  - Hierarchical stage: 5-10 minutes
  - Visualization: 10-20 minutes
- **Storage**: ~2 GB for outputs

## Advantages of This Approach

1. **Reuses 90% of existing code** - Minimal new development
2. **Maintains compatibility** - Works with existing pipeline
3. **Incremental enhancement** - Can be developed in phases
4. **Production-ready** - Uses proven components
5. **Scalable** - Handles 44M points efficiently
6. **Maintainable** - Clear separation of concerns

## Next Steps

1. Create feature branch: `git checkout -b feature/kmeans-hierarchical`
2. Copy and rename kmeans directory
3. Implement core two-stage algorithm
4. Extend visualizations
5. Update configuration and registration
6. Test with sample data
7. Run full-scale test
8. Create pull request
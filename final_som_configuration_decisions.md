# Final SOM Configuration Decisions for Biodiversity Analysis

## 1. Distance Metric
**Decision: Bray-Curtis with Partial Comparison**
```python
distance_config = {
    "input_space": "bray_curtis",
    "missing_data_handling": "pairwise",  # Only compare non-NA pairs
    "min_valid_features": 2,  # Require at least 2 valid features
    "map_space": "euclidean"  # Standard for grid topology
}

def partial_bray_curtis(u, v):
    """Bray-Curtis for valid pairs only"""
    valid = ~(np.isnan(u) | np.isnan(v))
    if valid.sum() < 2:
        return np.nan
    return np.sum(np.abs(u[valid] - v[valid])) / np.sum(u[valid] + v[valid])
```

## 2. Preprocessing Pipeline
**Decision: Log Transform → Z-score Standardization**
```python
preprocessing_config = {
    "transformation": "log1p",  # np.log1p(x) handles zeros
    "standardization": "z_score_by_type",  # Separate for observed/predicted
    "missing_data": "keep_nan",  # Don't impute, handle in distance
    "spatial_sampling": {
        "method": "block_sampling",
        "block_size": "750km",  # Middle of 500-1000km range
        "for_data_at": "100km_resolution"
    }
}

def preprocess_biodiversity(data):
    # Step 1: Log transformation
    data_transformed = np.log1p(data)
    
    # Step 2: Separate standardization
    scaler_obs = StandardScaler()
    scaler_pred = StandardScaler()
    data_transformed[:, [0,1]] = scaler_obs.fit_transform(data_transformed[:, [0,1]])
    data_transformed[:, [2,3]] = scaler_pred.fit_transform(data_transformed[:, [2,3]])
    
    return data_transformed
```

## 3. Initialization Methods
**Decision: Multiple Methods Available**
```python
initialization_config = {
    "primary_method": "pca_transformed",
    "fallback_methods": ["stratified_sample", "random_best_of_n"],
    "available_methods": {
        "pca_transformed": {
            "transform_first": True,
            "n_components": 2,
            "handle_missing": "mean_impute_for_pca_only"
        },
        "stratified_sample": {
            "n_strata": "equal_to_grid_size",
            "method": "kmeans_clustering"
        },
        "random": {
            "n_trials": 1,
            "seed": 42
        },
        "random_best_of_n": {
            "n_trials": 5,
            "criterion": "lowest_initial_qe"
        }
    }
}
```

## 4. Training Mode
**Decision: Batch Training**
```python
training_config = {
    "mode": "batch",  # All data at once
    "parallel_processing": True,
    "n_cores": "auto",  # Use all available
    "memory_management": {
        "chunk_if_exceeds": "8GB",
        "chunk_size": 10000
    }
}
```

## 5. GeoSOM + VLRSOM Architecture
**Decision: Hybrid Architecture with Adaptive Parameters**
```python
architecture_config = {
    "type": "GeoSOM_VLRSOM",
    
    # GeoSOM Parameters
    "spatial_weight": 0.3,  # k=0.3 (30% spatial, 70% features)
    "geographic_distance": "haversine",
    "combine_distances": "weighted_sum",
    
    # VLRSOM Parameters
    "initial_learning_rate": 0.5,
    "min_learning_rate": 0.01,
    "max_learning_rate": 0.8,
    "lr_increase_factor": 1.1,  # When QE improves
    "lr_decrease_factor": 0.85,  # When QE worsens
    
    # Adaptive regions
    "high_qe_lr_range": [0.5, 0.8],  # Fast learning
    "low_qe_lr_range": [0.01, 0.1],  # Fine tuning
    
    # Neighborhood Parameters
    "neighborhood_function": "gaussian",
    "initial_radius": "grid_size / 2",  # e.g., 10 for 20×20
    "final_radius": 1.0,
    "radius_decay": "linear",
    
    # Grid Parameters
    "topology": "rectangular",  # or "hexagonal"
    "grid_size": "determined_by_data",  # Set based on n_samples
    
    # Convergence Criteria
    "convergence": {
        "geographic_coherence_threshold": 0.7,  # Moran's I
        "lr_stability_threshold": 0.02,  # LR changes < 2%
        "qe_improvement_threshold": 0.001,  # < 0.1% improvement
        "patience": 50,  # Epochs without improvement
        "max_epochs": 1000
    }
}
```

## 6. Validation Strategy
**Decision: Spatial Block Cross-Validation**
```python
validation_config = {
    "method": "spatial_block_cv",
    "n_folds": 5,
    "block_size": "750km",  # Consistent with preprocessing
    "stratification": "ensure_all_biodiversity_types",
    "metrics": [
        "quantization_error",
        "topographic_error",
        "geographic_coherence",
        "beta_diversity_preservation"
    ]
}
```

## 7. Visualization & Reporting Suite
**Decision: Comprehensive Multi-View Reporting**
```python
reporting_config = {
    "visualizations": {
        "u_matrix": {
            "colormap": "viridis",
            "add_contours": True,
            "show_cluster_boundaries": True
        },
        "component_planes": {
            "show_all_features": True,
            "separate_colormaps": {
                "observed": "viridis",
                "predicted": "plasma"
            }
        },
        "hit_map": {
            "show_density": True,
            "log_scale": True
        },
        "geographic_projection": {
            "map_clusters_to_coords": True,
            "basemap": "world",
            "show_bioregions": True
        }
    },
    
    "statistical_reports": {
        "cluster_profiles": {
            "statistics": ["mean", "median", "std", "coverage"],
            "per_feature": True
        },
        "quality_metrics": {
            "som_metrics": ["QE", "TE"],
            "biodiversity_metrics": ["beta_diversity", "spatial_coherence"],
            "cluster_metrics": ["silhouette", "calinski_harabasz"]
        },
        "environmental_correlates": {
            "variables": ["temperature", "precipitation", "elevation"],
            "method": "correlation_analysis"
        }
    },
    
    "outputs": {
        "main_figure": "4_panel_summary",  # U-matrix, hit map, geographic, profiles
        "supplementary": "all_component_planes",
        "data_tables": ["cluster_summary", "quality_metrics", "environmental_correlates"],
        "format": "publication_ready"
    }
}
```

## Complete Integrated Pipeline
```python
def run_biodiversity_som_analysis(data, coordinates):
    """Complete pipeline with all decisions"""
    
    # 1. Preprocess
    data_processed = preprocess_biodiversity(data)
    
    # 2. Initialize SOM
    som = GeoSOM_VLRSOM(architecture_config)
    som.initialize_weights(data_processed, method="pca_transformed")
    
    # 3. Train with batch mode
    som.train_batch(
        data_processed, 
        coordinates,
        distance_metric=partial_bray_curtis,
        spatial_weight=0.3
    )
    
    # 4. Validate with spatial blocks
    cv_results = spatial_block_cv(som, data_processed, coordinates)
    
    # 5. Generate comprehensive reports
    reports = generate_all_reports(som, data_processed, coordinates)
    
    return som, cv_results, reports
```

## Summary of All Decisions
1. ✅ **Distance**: Bray-Curtis with pairwise comparison
2. ✅ **Preprocessing**: Log(x+1) → Z-score (separate for obs/pred)
3. ✅ **Initialization**: PCA-transformed primary, multiple fallbacks
4. ✅ **Training**: Batch mode with parallel processing
5. ✅ **Architecture**: GeoSOM + VLRSOM hybrid
6. ✅ **Validation**: Spatial block CV (750km blocks)
7. ✅ **Reporting**: Full visualization and statistical suite

All parameters are optimized for global biodiversity data with 70% missing values and mixed observed/predicted features.
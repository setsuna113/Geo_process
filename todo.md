  Recommended Architecture

  biodiversity_analysis/
  ├── som/
  │   ├── config.py          # SOM-specific configuration
  │   ├── trainer.py         # Iterative training loop
  │   ├── validator.py       # QE/TE metrics
  │   ├── biodiversity_som.py # Main entry point
  │   └── visualization.py   # Grid visualization
  │
  ├── gwpca/
  │   ├── config.py          # GWPCA configuration
  │   ├── analyzer.py        # One-pass computation
  │   ├── bandwidth.py       # Bandwidth selection
  │   ├── biodiversity_gwpca.py
  │   └── local_diagnostics.py
  │
  ├── maxp/
  │   ├── config.py          # MaxP configuration
  │   ├── optimizer.py       # Region optimization
  │   ├── contiguity.py      # Spatial weights
  │   ├── biodiversity_maxp.py
  │   └── region_metrics.py
  │
  └── shared/
      ├── biodiversity_metrics.py  # Beta diversity, etc.
      ├── spatial_validation.py    # blockCV implementation
      ├── data_preprocessing.py    # Zero-inflation handling
      └── comparison_framework.py  # Cross-method validation

  Benefits of Separation

  1. Clarity: Each method's code is self-contained and easier to understand
  2. Flexibility: Can optimize each method without breaking others
  3. Maintenance: Changes to SOM convergence don't affect GWPCA
  4. Testing: Unit tests are cleaner when methods are separate
  5. Performance: Can use method-specific optimizations (GPU for SOM, spatial indexing for GWPCA)

  What SHOULD be Shared

  Create a lightweight shared module for truly common biodiversity functionality:

  # shared/biodiversity_metrics.py
  class BiodiversityMetrics:
      @staticmethod
      def beta_diversity_preservation(original_distances, cluster_distances):
          """Works for any clustering method"""

      @staticmethod
      def spatial_autocorrelation(values, coordinates):
          """Moran's I works for any spatial data"""

      @staticmethod
      def species_association_accuracy(original, clustered):
          """Universal metric"""

  # shared/spatial_validation.py  
  class SpatialCrossValidator:
      """blockCV works for any spatial model"""
      def create_blocks(self, coordinates, autocorr_range):
          # Same blocks can evaluate SOM, GWPCA, or MaxP

  Migration Strategy

  1. Keep factory pattern for backward compatibility:
  class AnalyzerFactory:
      def create(method):
          if method == 'som':
              return BiodiversitySOM()  # New implementation
  2. Deprecate shared base class gradually
  3. Move biodiversity features to shared module
BIODIVERSITY STATISTICAL ANALYSIS PIPELINE - REQUIREMENTS & STRUCTURE

  PREPROCESSING REQUIREMENTS

  Universal Data Cleaning Rules (All Methods)

  # 1. Invalid value handling
  mask = ~np.isnan(data) & (data > 0)

  # 2. Extreme value cleaning (fungi datasets)
  data[data < -1e20] = np.nan      # Corrupt values
  data[data < 0] = np.nan          # Negative richness invalid
  data[data == 0] = np.nan         # Zero treated as no-data

  # 3. Valid masking criteria
  - Remove NaN values
  - Remove infinite values
  - Remove negatives (richness must be positive)
  - Remove zeros (no-data indicator)

  Raster-Specific Requirements

  # Multi-dataset alignment
  - Find common dimensions across all rasters
  - Crop to smallest extent
  - Apply consistent spatial sampling (e.g., every 5th pixel)

  # Feature variants
  - Support 4 features: plants, animals, am_fungi, ecm_fungi
  - Support 3 features: plants, animals, total_fungi (pre-merged)

  Feature Engineering Requirements

  # Only ONE computed feature needed
  total_fungi = am_fungi + ecm_fungi

  # Note: Methods don't require other features
  # - Gradients computed during analysis
  # - Normalization done during visualization
  # - Coordinates already in data

  ---
  PROPOSED STRUCTURE

  biodiversity_analysis_v2/
  ├── stat_pipeline/
  │   ├── loaders/
  │   │   ├── parquet_loader.py
  │   │   └── raster_loader.py
  │   ├── merger.py
  │   └── orchestrator.py
  ├── methods/
  │   └── stat/
  │       ├── correlation.py
  │       ├── trust_maps.py
  │       ├── overlap.py
  │       ├── coverage.py
  │       ├── latitudinal.py
  │       └── global_maps.py
  └── config/
      └── analysis.yml

  ---
  LOADER IMPLEMENTATIONS WITH BAKED-IN PREPROCESSING

  ParquetLoader

  class ParquetLoader:
      """
      Efficient parquet loading with baked-in preprocessing.
      Optimized for columnar operations.
      """

      def load(self, path: str, variant: str = '4_features') -> pd.DataFrame:
          """
          Load and preprocess parquet data.
          
          Args:
              path: Path to parquet file
              variant: '4_features' or '3_features_merged'
              
          Returns:
              Cleaned DataFrame with requested features
          """
          # Efficient columnar loading
          if variant == '4_features':
              columns = ['x', 'y', 'latitude', 'longitude',
                        'plant_richness', 'animal_richness',
                        'am_fungi_richness', 'ecm_fungi_richness']
          else:
              columns = ['x', 'y', 'latitude', 'longitude',
                        'plant_richness', 'animal_richness']

          df = pd.read_parquet(path, columns=columns)

          # Vectorized preprocessing
          for col in df.columns:
              if 'richness' in col:
                  # Efficient boolean indexing
                  if 'fungi' in col:
                      df.loc[df[col] < -1e20, col] = np.nan
                  df.loc[df[col] <= 0, col] = np.nan

          # Feature engineering for 3-feature variant
          if variant == '3_features_merged':
              if 'am_fungi_richness' in df.columns:
                  # Efficient vectorized operation
                  df['total_fungi_richness'] = (
                      df['am_fungi_richness'].fillna(0) +
                      df['ecm_fungi_richness'].fillna(0)
                  )
                  df.loc[df['total_fungi_richness'] == 0, 'total_fungi_richness'] = np.nan
                  df = df.drop(['am_fungi_richness', 'ecm_fungi_richness'], axis=1)

          # Remove invalid rows
          return df.dropna()

  RasterLoader

  class RasterLoader:
      """
      Efficient raster loading with baked-in preprocessing.
      Optimized for spatial operations.
      """

      def load(self, paths: Dict[str, str], variant: str = '4_features',
               sample_factor: int = 1) -> Dict[str, np.ndarray]:
          """
          Load and preprocess raster data.
          
          Args:
              paths: {name: file_path} for each dataset
              variant: '4_features' or '3_features_merged'
              sample_factor: Spatial sampling (1 = full resolution)
              
          Returns:
              Dictionary of cleaned, aligned arrays
          """
          # Find common dimensions
          min_h, min_w = self._find_common_dimensions(paths)

          loaded = {}

          # Efficient windowed reading if needed
          for name, path in paths.items():
              with rasterio.open(path) as src:
                  # Read with optional sampling
                  if sample_factor > 1:
                      # Efficient decimated read
                      data = src.read(1,
                          out_shape=(min_h // sample_factor,
                                    min_w // sample_factor),
                          resampling=Resampling.average)
                  else:
                      # Full resolution with cropping
                      window = Window(0, 0, min_w, min_h)
                      data = src.read(1, window=window).astype(np.float32)

                  # Preprocessing based on dataset type
                  if 'fungi' in name:
                      data[data < -1e20] = np.nan
                  data[data <= 0] = np.nan

                  loaded[name] = data

          # Handle variant merging
          if variant == '3_features_merged' and 'am_fungi' in loaded:
              # Efficient in-place operations
              total = np.empty_like(loaded['am_fungi'])
              np.add(np.nan_to_num(loaded['am_fungi'], 0),
                    np.nan_to_num(loaded['ecm_fungi'], 0),
                    out=total)
              total[total == 0] = np.nan

              loaded['total_fungi'] = total
              del loaded['am_fungi']
              del loaded['ecm_fungi']

          return loaded

      def load_with_coords(self, path: str) -> Dict:
          """
          Load single raster with coordinate information.
          Needed for latitudinal and global map analysis.
          """
          with rasterio.open(path) as src:
              data = src.read(1).astype(np.float32)
              bounds = src.bounds

              # Preprocessing
              if 'fungi' in path:
                  data[data < -1e20] = np.nan
              data[data <= 0] = np.nan

              # Generate coordinates
              h, w = data.shape
              lons = np.linspace(bounds.west, bounds.east, w)
              lats = np.linspace(bounds.north, bounds.south, h)

              return {
                  'data': data,
                  'lons': lons,
                  'lats': lats,
                  'bounds': bounds
              }

  ---
  METHOD MODIFICATIONS REQUIRED

  1. CorrelationAnalyzer

  # CURRENT: Loads and prepares data internally
  def load_and_prepare_data(self, datasets: Dict[str, str], sample_factor: int = 1)

  # CHANGE TO: Accept pre-loaded DataFrame
  def analyze(self, data: pd.DataFrame) -> Dict:
      """
      Analyze correlations in pre-loaded data.

      Args:
          data: DataFrame from ParquetLoader with richness columns
      """
      # Remove all loading logic
      # Work directly with provided DataFrame

  2. TrustMapVisualizer

  # CURRENT: Loads and aligns internally
  def load_and_align_data(self, datasets: Dict[str, str], sample_factor: int = 10)

  # CHANGE TO: Accept pre-loaded arrays
  def analyze(self, data: Dict[str, np.ndarray]) -> None:
      """
      Create trust maps from pre-loaded arrays.

      Args:
          data: Dictionary of aligned arrays from RasterLoader
      """
      # Remove load_align_and_sample call
      # Work directly with provided arrays

  3. OverlapAnalyzer

  # CURRENT: Loads and aligns internally
  def load_and_align_datasets(self, dataset_paths: Dict[str, str], sample_factor: int = 5)

  # CHANGE TO: Accept pre-loaded arrays
  def analyze(self, data: Dict[str, np.ndarray]) -> None:
      """
      Analyze overlap in pre-loaded arrays.

      Args:
          data: Dictionary of aligned arrays from RasterLoader
      """
      # Remove loading logic
      # Use provided aligned arrays

  4. CoverageAnalyzer

  # CURRENT: Mixed - uses both parquet and raster
  def analyze_coverage(self, datasets: Dict[str, str], sample_factor: int = 1)

  # CHANGE TO: Accept both data types
  def analyze(self, raster_data: Dict[str, np.ndarray],
             parquet_data: pd.DataFrame = None) -> None:
      """
      Analyze coverage using both data sources.

      Args:
          raster_data: For spatial coverage maps
          parquet_data: For true land coverage calculation
      """
      # Remove direct file loading
      # Use provided data

  5. LatitudinalAnalyzer

  # CURRENT: Opens raster directly
  with rasterio.open(raster_path) as src:

  # CHANGE TO: Accept data with coordinates
  def analyze(self, data_with_coords: List[Dict]) -> None:
      """
      Analyze latitudinal patterns.
      
      Args:
          data_with_coords: List of dicts from RasterLoader.load_with_coords()
                           Each contains 'data', 'lats', 'lons', 'bounds'
      """
      # Remove rasterio operations
      # Use provided coordinate info

  6. GlobalMapAnalyzer

  # CURRENT: Opens raster directly
  def load_raster_data(self, raster_path: str)

  # CHANGE TO: Accept data with coordinates
  def analyze(self, data_with_coords: List[Dict]) -> None:
      """
      Create global maps from pre-loaded data.

      Args:
          data_with_coords: List of dicts from RasterLoader.load_with_coords()
      """
      # Remove loading logic
      # Use provided arrays with coordinates

  ---
  ORCHESTRATOR USAGE

  class StatPipelineOrchestrator:
      def __init__(self):
          self.parquet_loader = ParquetLoader()
          self.raster_loader = RasterLoader()

      def run_analysis(self, config):
          # Load data once with appropriate variant
          variant = config['variant']  # '4_features' or '3_features_merged'

          # For statistical methods
          parquet_data = self.parquet_loader.load(
              config['parquet_path'],
              variant=variant
          )

          # For spatial methods
          raster_data = self.raster_loader.load(
              config['raster_paths'],
              variant=variant,
              sample_factor=config.get('sample_factor', 5)
          )

          # Run methods with appropriate data
          CorrelationAnalyzer().analyze(parquet_data)
          TrustMapVisualizer().analyze(raster_data)
          OverlapAnalyzer().analyze(raster_data)

          # Some methods need both
          CoverageAnalyzer().analyze(raster_data, parquet_data)

          # Some need coordinate info
          coords_data = [
              self.raster_loader.load_with_coords(path)
              for name, path in config['raster_paths'].items()
          ]
          LatitudinalAnalyzer().analyze(coords_data)
          GlobalMapAnalyzer().analyze(coords_data)

  ---
  KEY BENEFITS

  1. Single Load: Data loaded once, used by all methods
  2. Optimized Preprocessing: Each loader uses format-specific optimizations
  3. Clean Interfaces: Methods receive data, not file paths
  4. Variant Support: Easy switching between 4-feature and 3-feature analyses
  5. Memory Efficient: No redundant loading or preprocessing
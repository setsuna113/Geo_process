# Google Earth Engine Climate Data Module

Ultra-standalone module for extracting WorldClim bioclimatic variables using Google Earth Engine. Designed to integrate seamlessly with the existing geospatial biodiversity analysis pipeline.

## Features

- **Ultra-Standalone**: Minimal dependencies, can run independently of main pipeline
- **Coordinate Alignment**: Exact coordinate matching with existing pipeline (0.016667° resolution)
- **GEE Integration**: Handles authentication, quota limits, and chunked extraction
- **Pipeline Compatibility**: Outputs parquet files matching export_stage.py schema
- **Resume Capability**: Checkpoint system for long-running extractions
- **Logging Integration**: Uses project's structured logging system

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_gee.txt
```

### 2. Setup GEE Authentication

```bash
# For user authentication (interactive)
earthengine authenticate

# For service account (recommended for production)
# Download service account key from Google Cloud Console
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### 3. Extract Climate Data

```bash
# Basic extraction using config bounds
python scripts/extract_climate_data.py --output data/climate/

# Extract specific region
python scripts/extract_climate_data.py --bounds -10,-10,10,10 --output data/climate_test/

# Test with small area
python scripts/extract_climate_data.py --test --output data/climate_test/
```

## Module Components

### Core Classes

- **`GEEClimateExtractor`**: Main extraction engine
- **`CoordinateGenerator`**: Generates coordinate grids matching pipeline
- **`ParquetConverter`**: Converts data to pipeline-compatible format
- **`GEEAuthenticator`**: Handles GEE authentication

### Climate Variables

- **BIO01**: Annual Mean Temperature (°C)
- **BIO04**: Temperature Seasonality (°C)  
- **BIO12**: Annual Precipitation (mm)

## Configuration

The module reads configuration from `config.yml`:

```yaml
resampling:
  target_resolution: 0.016667  # ~5km resolution

processing_bounds:
  global: [-180, -90, 180, 90]
  test_small: [-10, -10, 10, 10]
```

## Output Format

Creates parquet files with schema matching export_stage.py:

```
x: float64         # longitude
y: float64         # latitude  
bio01: float32     # temperature
bio04: float32     # seasonality
bio12: float32     # precipitation
```

## Usage Examples

### Python API

```python
from src.climate_gee import create_gee_extractor

# Create extractor
extractor = create_gee_extractor(
    config_path="config.yml",
    service_account_key="/path/to/key.json"
)

# Extract data
bounds = (-10, -10, 10, 10)  # min_x, min_y, max_x, max_y
climate_data = extractor.extract_climate_data(bounds)

# Convert to parquet
from src.climate_gee import ParquetConverter
converter = ParquetConverter()
stats = converter.convert_to_parquet(climate_data, "output.parquet")
```

### Command Line

```bash
# Full global extraction
python scripts/extract_climate_data.py --output data/climate/

# With service account
python scripts/extract_climate_data.py \
    --service-account /path/to/key.json \
    --output data/climate/

# Resume interrupted extraction  
python scripts/extract_climate_data.py --resume --output data/climate/

# Custom bounds and variables
python scripts/extract_climate_data.py \
    --bounds -50,-20,50,20 \
    --variables bio01 bio12 \
    --output data/climate_custom/
```

## GEE Quotas and Limits

- **5000 points per request**: Module automatically chunks larger extractions
- **Daily computation limits**: Script includes retry logic and delays
- **Export limits**: Uses GEE's export system with task monitoring

## Troubleshooting

### Authentication Issues

```bash
# Reset authentication
earthengine authenticate --force

# Check authentication status
earthengine authenticate --list
```

### Memory Issues

```bash
# Reduce chunk size
python scripts/extract_climate_data.py --chunk-size 1000 --output data/climate/
```

### Quota Exceeded

```bash
# Wait and retry, or use smaller regions
python scripts/extract_climate_data.py --bounds -5,-5,5,5 --output data/climate_test/
```

## Integration with Pipeline

The module outputs are directly compatible with the main pipeline:

1. **Coordinate System**: Uses identical coordinate generation logic
2. **File Format**: Parquet files match export_stage.py schema
3. **Resolution**: Respects `target_resolution` from config.yml
4. **Logging**: Integrates with project logging infrastructure

## Development

### Running Tests

```python
# Test coordinate generation
from src.climate_gee import CoordinateGenerator
coord_gen = CoordinateGenerator(0.016667)
test_coords = coord_gen.generate_coordinate_grid((-1, -1, 1, 1))
print(f"Generated {len(test_coords)} coordinates")

# Test extraction (small area)
extractor = create_gee_extractor()
test_data = extractor.test_extraction()
```

### Adding New Variables

Edit `WORLDCLIM_DATASETS` in `gee_extractor.py`:

```python
'bio05': {
    'asset': 'WORLDCLIM/V1/BIO',
    'band': 'bio05', 
    'description': 'Max Temperature of Warmest Month',
    'units': '°C * 10',
    'scale_factor': 0.1
}
```
# Google Earth Engine Climate Data Module Implementation Checklist

## Overview
Ultra-standalone GEE module for extracting WorldClim bioclimatic variables (BIO1, BIO4, BIO12) that integrates with the existing pipeline coordinate system and uses the project's logging infrastructure.

## Phase 1: Module Structure and Dependencies

### 1.1 Create Module Directory Structure
- [ ] Create `src/climate_gee/` directory
- [ ] Create `src/climate_gee/__init__.py`
- [ ] Create `src/climate_gee/gee_extractor.py` (main module)
- [ ] Create `src/climate_gee/coordinate_generator.py` 
- [ ] Create `src/climate_gee/parquet_converter.py`
- [ ] Create `scripts/extract_climate_data.py` (standalone runner)

### 1.2 Dependency Management
- [ ] Add `earthengine-api` to requirements.txt
- [ ] Add `pyarrow` for parquet conversion (already in project)
- [ ] Create minimal requirements file: `requirements_gee.txt`
  ```
  earthengine-api>=0.1.370
  pandas>=1.5.0
  pyarrow>=10.0.0
  pyyaml>=6.0
  numpy>=1.20.0
  ```

### 1.3 Configuration Integration
- [ ] Create config reader that loads `config.yml`
- [ ] Extract `target_resolution: 0.016667` from config
- [ ] Extract database bounds if needed
- [ ] Create fallback defaults if config missing

## Phase 2: Core GEE Integration

### 2.1 GEE Authentication Module
```python
# src/climate_gee/auth.py
- [ ] Implement GEE authentication wrapper
- [ ] Support both service account and user authentication
- [ ] Handle authentication errors gracefully
- [ ] Cache authentication tokens
```

### 2.2 Coordinate Generation (Match Existing Logic)
```python
# src/climate_gee/coordinate_generator.py
- [ ] Copy coordinate generation logic from coordinate_merger.py
- [ ] Generate grid points at 0.016667° resolution
- [ ] Support configurable bounds (default: global)
- [ ] Ensure exact alignment with existing pipeline
```

### 2.3 GEE Data Extraction
```python
# src/climate_gee/gee_extractor.py
- [ ] Load WorldClim datasets:
  - ee.Image("WORLDCLIM/V1/BIO").select("bio01")  # Annual Mean Temp
  - ee.Image("WORLDCLIM/V1/BIO").select("bio04")  # Temp Seasonality
  - ee.Image("WORLDCLIM/V1/BIO").select("bio12")  # Annual Precipitation
- [ ] Implement point extraction at grid coordinates
- [ ] Handle GEE quota limits (5000 points per request)
- [ ] Implement retry logic for failed requests
- [ ] Add progress tracking with logging
```

## Phase 3: Data Export and Conversion

### 3.1 GEE Export Strategy
- [ ] Implement chunked export (5000 points per chunk)
- [ ] Export to GEE-managed CSV format
- [ ] Download CSV files from Google Drive/Cloud Storage
- [ ] Handle export task monitoring
- [ ] Implement timeout and retry logic

### 3.2 CSV to Parquet Conversion
```python
# src/climate_gee/parquet_converter.py
- [ ] Read downloaded CSV files
- [ ] Validate coordinate alignment
- [ ] Convert to parquet with schema:
  - x: float64 (longitude)
  - y: float64 (latitude)
  - bio01: float32 (temperature)
  - bio04: float32 (seasonality)
  - bio12: float32 (precipitation)
- [ ] Use snappy compression (matching export_stage.py)
```

## Phase 4: Logging Integration

### 4.1 Setup Logging
```python
# In each module:
from src.infrastructure.logging import setup_simple_logging, get_logger

# At module start:
setup_simple_logging('INFO')
logger = get_logger(__name__)
```

### 4.2 Implement Structured Logging
- [ ] Log GEE authentication status
- [ ] Log coordinate generation (grid size, bounds)
- [ ] Log extraction progress (chunks completed)
- [ ] Log export task status
- [ ] Log conversion metrics (rows, file sizes)
- [ ] Use logger.log_performance() for timing

## Phase 5: Standalone Script

### 5.1 Create Runner Script
```python
# scripts/extract_climate_data.py
- [ ] Parse command line arguments:
  - --bounds: geographic bounds (default: config)
  - --output: output directory
  - --chunk-size: points per GEE request
  - --config: path to config.yml
- [ ] Initialize logging
- [ ] Run extraction pipeline
- [ ] Generate summary report
```

### 5.2 Error Handling
- [ ] Catch GEE authentication errors
- [ ] Handle network timeouts
- [ ] Manage quota exceeded errors
- [ ] Create checkpoint system for resume
- [ ] Log all errors with context

## Phase 6: Integration Points

### 6.1 Pipeline Integration
- [ ] Output parquet matches export_stage.py format
- [ ] Coordinate system matches exactly
- [ ] File naming convention compatible
- [ ] Metadata file generation

### 6.2 Configuration Compatibility
- [ ] Read from same config.yml
- [ ] Use same coordinate generation
- [ ] Compatible with resampling.target_resolution
- [ ] Support processing_bounds from config

## Phase 7: Testing and Validation

### 7.1 Unit Tests
- [ ] Test coordinate generation matches pipeline
- [ ] Test GEE authentication handling
- [ ] Test chunking logic
- [ ] Test CSV to parquet conversion

### 7.2 Integration Tests
- [ ] Test small region extraction
- [ ] Validate output coordinates
- [ ] Compare with existing pipeline data
- [ ] Test resume after interruption

### 7.3 Validation Checks
- [ ] Verify coordinate alignment (< 0.001° tolerance)
- [ ] Check data ranges (temperature, precipitation)
- [ ] Validate no missing values
- [ ] Ensure parquet schema matches

## Phase 8: Documentation

### 8.1 Usage Documentation
- [ ] Create README.md for climate_gee module
- [ ] Document GEE authentication setup
- [ ] Provide example commands
- [ ] List common issues and solutions

### 8.2 Code Documentation
- [ ] Add docstrings to all functions
- [ ] Document coordinate system
- [ ] Explain GEE quotas and limits
- [ ] Document resume capability

## Implementation Order

1. **Week 1**: Module structure, config integration, logging setup
2. **Week 1**: GEE authentication and basic extraction
3. **Week 2**: Coordinate generation and alignment
4. **Week 2**: Chunked extraction and export
5. **Week 3**: CSV download and parquet conversion
6. **Week 3**: Testing and validation
7. **Week 4**: Documentation and integration

## Key Design Decisions

1. **Ultra-Standalone**: Minimal dependencies, can run independently
2. **Config-Driven**: Reads resolution and bounds from config.yml
3. **Logging Integration**: Uses project's structured logging
4. **Coordinate Alignment**: Exact match with existing pipeline
5. **Resume Capability**: Checkpoint system for long runs
6. **GEE Efficiency**: Chunked extraction to avoid quotas

## Example Usage

```bash
# Basic extraction
python scripts/extract_climate_data.py --output data/climate/

# With custom bounds
python scripts/extract_climate_data.py \
    --bounds -10,-10,10,10 \
    --output data/climate_test/

# Resume interrupted extraction
python scripts/extract_climate_data.py \
    --resume \
    --output data/climate/
```

## Success Criteria

- [ ] Extracts BIO1, BIO4, BIO12 for all grid points
- [ ] Output coordinates match pipeline exactly
- [ ] Parquet files compatible with export_stage.py
- [ ] Logging provides clear progress tracking
- [ ] Can resume after interruption
- [ ] Handles GEE quotas gracefully
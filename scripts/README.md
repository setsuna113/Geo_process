# Scripts Directory Organization

This directory contains utility scripts organized by purpose.

## Directory Structure

```
scripts/
├── debug/          # Debugging and analysis scripts
│   └── som/        # SOM-specific debug tools
├── monitoring/     # Real-time monitoring scripts
├── testing/        # Test scripts and data generators
└── tools/          # General utility scripts
```

## Debug Scripts (`debug/`)

### SOM Debug Tools (`debug/som/`)
- `check_som_internals.py` - Inspect SOM internal state and weights
- `optimize_som_performance.py` - Performance optimization utilities
- `debug_som_issue.py` - Debug specific SOM issues
- `debug_vectorized_som.py` - Debug vectorized implementation
- `check_vectorization_memory.py` - Memory usage analysis
- `debug_geo_distance.py` - Geographic distance calculation debugging
- `estimate_optimized_time.py` - Performance estimation tools
- `quick_som_check.py` - Quick SOM health check

## Monitoring Scripts (`monitoring/`)
- `monitor_som.py` - Basic SOM training monitor
- `monitor_som_enhanced.py` - Enhanced monitoring with detailed metrics
- `monitor_som_live.py` - Live monitoring with progress tracking
- `monitor_fixed_som.py` - Monitor for debugging fixed issues

## Testing Scripts (`testing/`)
- `generate_test_data.py` - Generate synthetic test datasets
- `test_fixed_som.py` - Test SOM fixes
- `memory_profile_test.py` - Memory profiling utilities

## General Scripts
- `run_analysis.py` - Main analysis runner
- `process_manager.py` - Daemon process management
- `fix_hardcoded_paths.py` - Fix hardcoded paths in scripts

## Usage Notes

1. All scripts now use relative paths based on project root
2. To run a script from anywhere:
   ```bash
   python /path/to/geo/scripts/debug/som/check_som_internals.py
   ```

3. For monitoring scripts, ensure the experiment is running first
4. Debug scripts can be run independently for analysis

## Path Resolution

Scripts use `Path(__file__).parent.parent.parent` to find the project root, ensuring they work regardless of where they're called from.
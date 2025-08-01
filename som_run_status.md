# SOM Production Run Status

## Current Run (PID: 3817140)
- **Started**: 10:11:17 AM
- **Current Time**: ~10:35 AM (24+ minutes elapsed)
- **Status**: Running CV fold 1/5
- **CPU Usage**: 102% (actively computing)
- **Memory Usage**: 3.7GB
- **Grid Size**: 15×15 (225 neurons) ✓
- **Dataset**: 11.3M samples, 94.8% missing data

## Configuration Issues Fixed
1. ✅ Grid size correctly using 15×15 instead of 50×50
2. ✅ Fixed current_qe undefined bug
3. ✅ Fixed missing datetime import
4. ✅ Fixed parameter passing (cv_folds, observed_columns, etc.)
5. ⚠️ Current run still using 5 CV folds (fix applied for future runs)

## Expected Timeline
- Each CV fold: ~30-40 minutes (processing 9M training samples)
- Total CV time: ~150-200 minutes (2.5-3.3 hours)
- Final model training: ~30-40 minutes
- **Total estimated time**: 3-4 hours (well within 5-hour deadline)

## Why It's Taking Time
1. Large dataset: 11.3M samples with very sparse data (94.8% missing)
2. Partial Bray-Curtis distance calculation for each sample pair
3. 5 CV folds × up to 1000 epochs each
4. Spatial block cross-validation requires geographic computations

## Monitoring
- Progress file: `/home/yl998/dev/geo/outputs/analysis_results/som/som_progress_som_20250801_101125.json`
- Live monitor: `python monitor_som_live.py`
- Log file: `som_fixed_run.log`

## Next Steps
1. Let current run complete (3-4 hours remaining)
2. Future runs will use cv_folds=3 for faster execution
3. Results will be saved to configured output directory

The analysis is proceeding correctly and will complete within the deadline.
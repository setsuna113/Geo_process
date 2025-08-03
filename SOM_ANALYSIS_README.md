# SOM Analysis Scripts

## Essential Scripts for SOM Analysis

After cleanup, we have only TWO essential scripts for running and monitoring SOM analysis:

### 1. `run_som_analysis.sh` - Main Analysis Runner
This is the ONLY script needed to run SOM analysis.

**Usage:**
```bash
# Run analysis (default: optimized config)
./run_som_analysis.sh

# Run with monitoring in tmux
./run_som_analysis.sh --both

# Only monitor existing run
./run_som_analysis.sh --monitor

# Run in detached tmux session
./run_som_analysis.sh --both --detached
```

**Features:**
- Automatically uses optimized configuration (50x50 grid, 200 workers)
- Sets up environment variables (SOM_MAX_WORKERS=200)
- Activates conda environment
- Logs to timestamped files
- Color-coded output for different message types

### 2. `som_progress_monitor.py` - Live Progress Monitor
Python-based progress monitor with clean display.

**Usage:**
```bash
# Run the monitor
python som_progress_monitor.py

# Or make it executable
./som_progress_monitor.py
```

**Features:**
- Live progress updates (CV folds, epochs, QE)
- System resource monitoring (load, processes)
- Auto-detects latest log file
- Clean, formatted display
- Updates every 2 seconds

## Current Configuration

The analysis uses `som_config_optimized.yml` with:
- Grid size: 50x50 (2,500 neurons)
- Workers: 200 (for parallel processing)
- Chunk size: 50,000 samples
- CV folds: 5
- Preprocessing: log1p + minmax normalization

## Typical Workflow

1. Start analysis in tmux with monitoring:
   ```bash
   ./run_som_analysis.sh --both
   ```

2. Or run separately:
   ```bash
   # Terminal 1
   ./run_som_analysis.sh
   
   # Terminal 2
   ./som_progress_monitor.py
   ```

3. Check results in:
   - `outputs/analysis_results/som/test_som/`
   - `logs/som_optimized_*.log`

## Performance Optimizations

- Chunk size reduced from 100k to 50k samples
- Worker count increased from 128 to 200
- Creates ~182 parallel chunks for better CPU utilization
- Expected load average: 150-200 on 256-core system
# Geo_process

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible spatial analysis pipeline for biodiversity research supporting multiple grid systems (cubic and hexagonal). This pipeline processes species occurrence data and environmental variables for ecological modeling.

## Features

- **Multiple grid systems**: Cubic and hexagonal (H3) grids
- **Efficient processing**: PostgreSQL/PostGIS backend
- **Modular design**: Inspired by GeoCore architecture
- **Reproducible**: Full workflow from raw data to analysis-ready datasets

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/biodiversity-spatial-pipeline
cd biodiversity-spatial-pipeline

# Install dependencies
conda env create -f environment.yml
conda activate your-env-name

# Setup database
./scripts/setup_database.sh
```

## Quick Start
```bash
from geo_process import Pipeline

# Initialize pipeline
pipeline = Pipeline(
    grid_type='cubic',
    resolution=5000,  # 5km cells
    database_config='config.yml'
)

# Run analysis
results = pipeline.run(
    species_data='data/species/',
    climate_data='data/worldclim/'
)
```

🎮 Usage:

  Start analysis:
  ./run_analysis.sh

  Monitor running analysis:
  tmux attach -t richness_analysis

  Detach and let run in background:
  Ctrl+B then d
  or tmux detach-client -s richness_analysis from another terminal window

  Check if still running:
  tmux list-sessions
  tmux capture-pane -p -t richness_analysis

  Emergency stop:
  tmux kill-session -t richness_analysis
d
  📁 Output Files:

  - Results: outputs/spatial_analysis/Richness_SOM_*
  - Logs: logs/richness_analysis_*.log
  - Checkpoints: checkpoint_*.json

## Acknowledgments
This project's architecture was inspired by the GeoCore framework. See ATTRIBUTIONS.md for full credits.

## License
MIT License



Yes, tmux is perfect for this. Here's a copyable command that will run Claude in the background starting from Phase 4:

  tmux new-session -d -s claude_phase4 'claude -p "Continue implementation from Phase 4: Migration & Integration in 
  implementation_plan.md. Work through Phase 4 step by step, updating the markdown file to check off completed items [x], 
  testing each migration, and cleaning up temporary files. Continue through Phases 5-6 until all phases are complete. Work 
  silently and autonomously." --dangerously-skip-permissions'

  To monitor progress while you sleep, you can also set up a second pane to watch the implementation plan file:

  tmux new-session -d -s claude_phase4 \; \
    send-keys 'claude -p "Continue implementation from Phase 4: Migration & Integration in implementation_plan.md. Work 
  through Phase 4 step by step, updating the markdown file to check off completed items [x], testing each migration, and 
  cleaning up temporary files. Continue through Phases 5-6 until all phases are complete. Work silently and autonomously." 
  --dangerously-skip-permissions' Enter \; \
    split-window -h \; \
    send-keys 'watch -n 30 "echo \"=== PROGRESS ===\"; grep -E \"\\[x\\]|\\[ \\]\" implementation_plan.md | head -20"' Enter

  When you wake up, attach with:
  tmux attach -t claude_phase4

  The second command gives you a split view where the right pane shows progress updates every 30 seconds.

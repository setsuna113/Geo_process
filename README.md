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
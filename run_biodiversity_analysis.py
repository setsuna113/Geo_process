#!/usr/bin/env python
"""
Top-Level Biodiversity Analysis Runner

Provides unified access to all biodiversity statistical analyses.

Usage:
    python run_biodiversity_analysis.py                    # Run all statistical analyses
    python run_biodiversity_analysis.py --analysis coverage # Run specific analysis
    python run_biodiversity_analysis.py --clean           # Clean all outputs
    python run_biodiversity_analysis.py --list            # List available analyses
    python run_biodiversity_analysis.py --help            # Show detailed help

Available Analysis Types:
    Statistical Methods (methods/stat/):
        - coverage: Data coverage and completeness analysis
        - trust_maps: Trust and reliability mapping
        - latitudinal: Biodiversity patterns across latitude
        - global_maps: Global biodiversity visualization
        - correlation: Multi-dataset correlation studies
        - artifacts: Data artifact and anomaly detection
        - overlap: Dataset overlap and intersection analysis
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point - delegates to statistical analysis runner"""
    
    # Path to the unified biodiversity analysis runner
    stat_runner = Path(__file__).parent / "scripts/run_biodiversity_analysis.py"
    
    if not stat_runner.exists():
        print(f"❌ Error: Statistical analysis runner not found at {stat_runner}")
        sys.exit(1)
    
    # Pass all arguments to the statistical runner
    cmd = [sys.executable, str(stat_runner)] + sys.argv[1:]
    
    try:
        # Run the statistical analysis runner with same arguments
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
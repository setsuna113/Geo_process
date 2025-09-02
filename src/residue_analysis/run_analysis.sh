#!/bin/bash
# Run DML residual analysis pipeline

# Default values
DATA_PATH="/scratch/yl998/econml_results_5m/homogeneous/residuals.arrow"
OUTPUT_DIR="./results"
CONFIG_FILE=""

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--data)
      DATA_PATH="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -d, --data PATH     Path to residuals arrow file"
      echo "  -o, --output DIR    Output directory for results"
      echo "  -c, --config FILE   JSON configuration file"
      echo "  -h, --help          Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the analysis
if [ -n "$CONFIG_FILE" ]; then
  echo "Running analysis with config file: $CONFIG_FILE"
  Rscript src/residue_analysis/main_analysis.R "$CONFIG_FILE"
else
  echo "Running analysis:"
  echo "  Data: $DATA_PATH"
  echo "  Output: $OUTPUT_DIR"
  Rscript src/residue_analysis/main_analysis.R "$DATA_PATH" "$OUTPUT_DIR"
fi
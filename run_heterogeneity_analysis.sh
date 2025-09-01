#!/bin/bash
# ============================================================================
# Run Heterogeneity Analysis on Raw Biodiversity Data
# ============================================================================
# This script analyzes whether biological relationships vary across biomes
# Requires RAW biodiversity data, not residuals
# ============================================================================

# Default values
DATA_PATH=""
OUTPUT_DIR="./results/heterogeneity_analysis"

# Parse arguments
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
    -h|--help)
      echo "Usage: $0 -d DATA_PATH [-o OUTPUT_DIR]"
      echo ""
      echo "Run heterogeneity analysis to test if biological relationships vary across biomes"
      echo ""
      echo "Arguments:"
      echo "  -d, --data PATH     Path to raw biodiversity data (arrow format)"
      echo "  -o, --output DIR    Output directory (default: ./results/heterogeneity_analysis)"
      echo ""
      echo "Example:"
      echo "  $0 -d /data/biodiversity_raw.arrow -o ./results/heterogeneity"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h for help"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$DATA_PATH" ]; then
  echo "Error: Data path is required"
  echo "Use -h for help"
  exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
  echo "Error: Data file not found: $DATA_PATH"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run analysis
echo "============================================"
echo "Running Heterogeneity Analysis"
echo "============================================"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

Rscript src/heterogeneity_analysis/heterogeneity_analysis.R "$DATA_PATH" "$OUTPUT_DIR"

echo ""
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
#!/bin/bash
# SHAP analysis pipeline: Julia GLM → Python SHAP → R shapviz

set -e  # Exit on error

# Simple and direct - no fancy abstractions
JULIA_OUTPUT_DIR="${1:-results/julia_results}"
SHAP_OUTPUT_DIR="${2:-results/shap}"
SHAPVIZ_OUTPUT_DIR="${3:-results/shapviz}"

echo "SHAP Pipeline: $JULIA_OUTPUT_DIR → $SHAP_OUTPUT_DIR → $SHAPVIZ_OUTPUT_DIR"

# Check Julia output exists
[ ! -f "$JULIA_OUTPUT_DIR/julia_results_latest.json" ] && {
    echo "Error: No Julia results at $JULIA_OUTPUT_DIR"
    exit 1
}

# Run Python SHAP
python src/machine_learning/interpretation/shap_glm_analysis.py \
    "$JULIA_OUTPUT_DIR" \
    --output_dir "$SHAP_OUTPUT_DIR"

# Run R shapviz (if R is available)
command -v Rscript >/dev/null 2>&1 && \
    Rscript src/machine_learning/visualization/shapviz_plots.R \
        "$SHAP_OUTPUT_DIR" \
        "$SHAPVIZ_OUTPUT_DIR"

echo "Done. Results in $SHAP_OUTPUT_DIR"
#!/bin/bash
# Complete SHAP analysis pipeline: Julia GLM → Python SHAP → R shapviz

set -e  # Exit on error

# Configuration
JULIA_OUTPUT_DIR="${1:-results/julia_results}"
SHAP_OUTPUT_DIR="${2:-results/shap}"
SHAPVIZ_OUTPUT_DIR="${3:-results/shapviz}"

echo "==========================================="
echo "SHAP Analysis Pipeline"
echo "==========================================="
echo "Julia output: $JULIA_OUTPUT_DIR"
echo "SHAP output: $SHAP_OUTPUT_DIR"
echo "Shapviz output: $SHAPVIZ_OUTPUT_DIR"
echo ""

# Step 1: Check if Julia models exist
if [ ! -d "$JULIA_OUTPUT_DIR/shap_data" ]; then
    echo "❌ Error: Julia SHAP data not found at $JULIA_OUTPUT_DIR/shap_data"
    echo "Please run Julia pipeline with export_for_shap() first"
    exit 1
fi

echo "✓ Found Julia SHAP exports"

# Step 2: Run Python SHAP analysis
echo ""
echo "Step 2: Computing SHAP values with Python LinearExplainer..."
echo "-------------------------------------------"
python src/machine_learning/interpretation/shap_glm_analysis.py \
    "$JULIA_OUTPUT_DIR" \
    --output_dir "$SHAP_OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ SHAP computation complete"
else
    echo "❌ SHAP computation failed"
    exit 1
fi

# Step 3: Create R shapviz visualizations
echo ""
echo "Step 3: Creating shapviz visualizations..."
echo "-------------------------------------------"
Rscript src/machine_learning/visualization/shapviz_plots.R \
    "$SHAP_OUTPUT_DIR" \
    "$SHAPVIZ_OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Shapviz visualizations complete"
else
    echo "❌ Shapviz visualization failed"
    exit 1
fi

# Step 4: Summary
echo ""
echo "==========================================="
echo "Pipeline Complete!"
echo "==========================================="
echo "Results available at:"
echo "  - SHAP values: $SHAP_OUTPUT_DIR"
echo "  - Visualizations: $SHAPVIZ_OUTPUT_DIR"
echo ""
echo "View results with:"
echo "  ls -la $SHAPVIZ_OUTPUT_DIR/"
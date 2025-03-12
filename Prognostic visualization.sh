#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Step 1: Calculating mean of survival SHAP values..."
if python scripts/surv_shap_calculate_mean.py; then
    echo "Successfully calculated mean of survival SHAP values."
else
    echo "Error: Failed to calculate mean of survival SHAP values."
    exit 1
fi

echo "Step 2: Visualizing prognostic predictions..."
if python scripts/prognosis_visualization.py; then
    echo "Successfully visualized prognostic predictions."
else
    echo "Error: Failed to visualize prognostic predictions."
    exit 1
fi

echo "All steps completed successfully."

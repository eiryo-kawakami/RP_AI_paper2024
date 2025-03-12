#!/bin/bash

# Step 1: Save predictions for each figure
echo "Step 1: Saving predictions for each figure..."
python scripts/preds_nobatch.py

if [ $? -ne 0 ]; then
    echo "Error in Step 1: Saving predictions failed."
    exit 1
fi

# Step 2: Extract feature values
echo "Step 2: Extracting feature values..."
python scripts/extract_values.py

if [ $? -ne 0 ]; then
    echo "Error in Step 2: Extracting feature values failed."
    exit 1
fi

# Step 3: Calculate and visualize SHAP values
echo "Step 3: Calculating SHAP values..."
python scripts/calculate_shap_value.py

if [ $? -ne 0 ]; then
    echo "Error in Step 3: Calculating SHAP values failed."
    exit 1
fi

echo "Visualizing SHAP values..."
python scripts/diagnosis_visualization.py

if [ $? -ne 0 ]; then
    echo "Error in Step 3: Visualizing SHAP values failed."
    exit 1
fi

echo "All steps completed successfully!"
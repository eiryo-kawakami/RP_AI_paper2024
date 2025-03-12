#!/bin/bash

# Step 1: Run inference for diagnostic predictions
echo "Step 1: Running inference for diagnostic predictions..."
python scripts/inference.py

if [ $? -ne 0 ]; then
    echo "Error in Step 1: Inference failed."
    exit 1
fi

# Step 2: Run ensemble for diagnostic predictions
echo "Step 2: Running ensemble for diagnostic predictions..."
python scripts/ensemble.py

if [ $? -ne 0 ]; then
    echo "Error in Step 2: Ensemble failed."
    exit 1
fi

echo "Diagnostic predictions completed successfully!"

#!/bin/bash

# Run the test script
python test.py

# Check the exit status of the test script
if [ $? -ne 0 ]; then
    exit 1
else
    echo "Test passed (micro-commit)...."
    git add .
    git commit -m "🧗 Commit Recovery"
    python train.py
fi
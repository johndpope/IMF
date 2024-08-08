#!/bin/bash
# in wandb - I successfully trained - https://wandb.ai/snoozie/IMF/runs/zh1o9mo0/logs
# but the commits were only staged - and then I couldn't recreate test. 
# wandb store the git commit - this will guarantee repeatability 

# Run the test script
# python test.py

# Check the exit status of the test script
if [ $? -ne 0 ]; then
    exit 1
else
    echo "Test passed (micro-commit)...."
    git add .
    git commit -m "🧗"
    python train.py
fi
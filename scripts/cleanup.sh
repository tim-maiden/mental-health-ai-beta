#!/bin/bash

echo "--- Cleaning previous artifacts ---"
rm -rf data models venv wandb outputs __pycache__
find . -name "__pycache__" -type d -exec rm -rf {} +
rm *.csv *.png *.jsonl 2>/dev/null || true
echo "--- Cleanup complete ---"


#!/bin/bash

# Exit script on any error
set -e

echo "Starting Control System..."

# Activate virtual environment (if used)
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Installing required dependencies..."
pip install -r requirements.txt

echo "Running CONSYS..."
python -m system.consys

echo "Execution finished successfully!"
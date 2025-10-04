#!/bin/bash
# Lockless Biometric Authentication System - Linux/Mac Launcher
# This script automatically activates the virtual environment and runs commands

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup first:"
    echo "python setup.py"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if arguments provided
if [ $# -eq 0 ]; then
    echo "Lockless Biometric Authentication System"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --test-camera"
    echo "  ./run.sh --enroll --user john --password secret123"
    echo "  ./run.sh --authenticate --user john --password secret123"
    echo "  ./run.sh --list-users"
    echo "  ./run.sh --diagnostics"
    echo ""
    exit 0
fi

# Run the main application with all arguments
export PYTHONPATH=.
python src/main.py "$@"
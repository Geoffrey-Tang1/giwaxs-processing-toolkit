#!/usr/bin/env bash
# run_giwaxs_platform.command
#
# Double-click launcher for macOS/Linux.
# On first run: creates a local virtual environment (.venv) inside this
# folder and installs the required packages -- no admin rights needed.
# On later runs: just activates that environment and launches the platform.

set -e

# Always operate relative to this script's own location, so double-clicking
# from Finder (or running from any directory) works correctly.
cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "First run detected -- setting up a local Python environment..."
    echo "(This only happens once; it will be much faster next time.)"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

echo ""
echo "Starting GIWAXS Processing Platform..."
echo ""
python3 giwaxs_platform.py "$@"

echo ""
read -p "Press Enter to close this window..."

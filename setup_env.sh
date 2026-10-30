#!/usr/bin/env bash
# Create a virtual environment and install the libraries (Linux / macOS / Git Bash).
# Run:  bash setup_env.sh
set -e

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name obesity-env --display-name "Python (obesity-env)"

echo ""
echo "Done. Activate the environment with:  source .venv/bin/activate"

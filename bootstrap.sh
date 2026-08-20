#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "PyStereo bootstrap"
echo "=================="

if [ ! -d .venv ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi

echo "Activating virtualenv..."
source .venv/bin/activate

echo "Upgrading pip..."
python3 -m pip install -U pip -q

echo "Installing dependencies..."
python3 -m pip install -r requirements.txt -q

echo ""
echo "Done. To run PyStereo:"
echo ""
echo "  source .venv/bin/activate"
echo "  python app.py                     # Web UI"
echo "  python -m pystereo_core           # Desktop GUI"
echo "  python -m pystereo_core --cli     # CLI batch mode"
echo ""

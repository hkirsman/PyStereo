#!/usr/bin/env bash
# Build both standalone bundles on macOS:
#   dist/PyStereo/PyStereo.app   - Qt GUI + CLI batch tool
#   dist/PyStereoWeb/PyStereoWeb.app - Flask web UI (open http://127.0.0.1:8766)
#
# Creates .venv and installs deps if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

PYTHON=".venv/bin/python3"

if [[ ! -x "$ROOT/$PYTHON" ]]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi

echo "Installing project dependencies into .venv..."
"$ROOT/$PYTHON" -m pip install -U pip
"$ROOT/$PYTHON" -m pip install -r requirements.txt

echo "Installing PyInstaller into .venv (if needed)..."
"$ROOT/$PYTHON" -m pip install -q -U pyinstaller

echo "Building PyStereo (batch GUI + CLI)..."
rm -rf dist/PyStereo dist/PyStereo.app
"$ROOT/$PYTHON" -m PyInstaller --noconfirm packaging/pystereo_batch.spec
rm -rf dist/PyStereo
mkdir -p dist/PyStereo
mv dist/PyStereo.app dist/PyStereo/

echo "Building PyStereoWeb (Flask server)..."
rm -rf dist/PyStereoWeb dist/PyStereoWeb.app
"$ROOT/$PYTHON" -m PyInstaller --noconfirm packaging/pystereo_web.spec
rm -rf dist/PyStereoWeb
mkdir -p dist/PyStereoWeb
mv dist/PyStereoWeb.app dist/PyStereoWeb/

VERSION=$("$ROOT/$PYTHON" -c "from pystereo_core._version import __version__; print(__version__)")

echo "Packaging archives..."
(
  cd "$ROOT/dist"
  rm -f "PyStereo-${VERSION}-mac.zip" "PyStereoWeb-${VERSION}-mac.zip"
  zip -r -y -q "PyStereo-${VERSION}-mac.zip" PyStereo
  zip -r -y -q "PyStereoWeb-${VERSION}-mac.zip" PyStereoWeb
)

echo ""
echo "========================================================================"
echo "Build finished OK - version ${VERSION}."
echo ""
echo "Batch tool (GUI/CLI):"
echo "  $ROOT/dist/PyStereo/PyStereo.app"
echo "  Folder: $ROOT/dist/PyStereo/"
echo "  CLI: $ROOT/dist/PyStereo/PyStereo.app/Contents/MacOS/PyStereo --cli ..."
echo ""
echo "Web UI (Flask server - open http://127.0.0.1:8766 after starting):"
echo "  $ROOT/dist/PyStereoWeb/PyStereoWeb.app"
echo "  Folder: $ROOT/dist/PyStereoWeb/"
echo ""
echo "Archives (upload these to the GitHub release):"
echo "  $ROOT/dist/PyStereo-${VERSION}-mac.zip"
echo "  $ROOT/dist/PyStereoWeb-${VERSION}-mac.zip"
echo ""
echo "NOTE: bundles built locally are not notarised. To open them the first"
echo "  time: right-click -> Open -> Open, or run:"
echo "  xattr -dr com.apple.quarantine dist/PyStereo/ dist/PyStereoWeb/"
echo "========================================================================"

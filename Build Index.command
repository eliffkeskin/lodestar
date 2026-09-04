#!/bin/bash
# ============================================================
# Lodestar — build/refresh the doc index (macOS)
# ============================================================
# Double-click this after adding or changing docs in the docs/ folder.

cd "$(dirname "$0")"
if [ -d "venv" ]; then source venv/bin/activate; fi

echo "Building the documentation index from docs/ ..."
echo ""
python rag.py --index
echo ""
echo "Done. Restart the app to use the updated index."
echo "Press any key to close..."
read -n 1

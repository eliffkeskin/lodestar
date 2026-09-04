#!/bin/bash
# ============================================================
# Lodestar — one-time setup (macOS)
# ============================================================
# Double-click this ONCE before the first run.
# It creates a virtual environment and installs dependencies.

cd "$(dirname "$0")"

echo "Setting up Lodestar..."
echo "This creates a local Python environment and installs dependencies."
echo ""

# Create venv if missing.
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and add your ANTHROPIC_API_KEY"
echo "  2. Put your redacted docs (.pdf/.md) in the docs/ folder"
echo "  3. Build the index:  double-click 'Build Index.command'"
echo "  4. Start the app:    double-click 'Lodestar.command'"
echo ""
echo "Press any key to close..."
read -n 1

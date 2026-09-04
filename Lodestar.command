#!/bin/bash
# ============================================================
# Lodestar — double-click launcher (macOS)
# ============================================================
# Double-click this file in Finder to start the app.
# It starts the local server and opens a compact app window (Chrome app mode),
# without browser tabs or an address bar. Close the Terminal window to stop.

cd "$(dirname "$0")"

# Activate the virtual environment if it exists.
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Load the API key from a local .env file if present.
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "  NOTE: ANTHROPIC_API_KEY is not set."
  echo "  Screenshot reading and doc-based answers will be disabled."
  echo ""
fi

PORT=8502
URL="http://localhost:${PORT}"

echo "Starting Lodestar..."
echo "A compact app window will open shortly. Close this Terminal window to stop."
echo ""

# Start Streamlit as a server only (don't let it open a normal browser tab).
streamlit run app.py \
  --server.port ${PORT} \
  --server.headless true \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!

# Wait until the server is actually responding before opening the window.
echo "Waiting for the server to be ready..."
for i in {1..30}; do
  if curl -s "${URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Open in Chrome "app mode": no tabs, no address bar, compact window.
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --app="${URL}" --window-size=560,820 --window-position=200,60 \
            >/dev/null 2>&1 &
else
  # Fallback: if Chrome isn't installed, just open the default browser.
  echo "Google Chrome not found — opening in your default browser instead."
  open "${URL}"
fi

# Keep the script alive so closing the Terminal stops the server.
echo ""
echo "App is running. Close this Terminal window (or press Ctrl+C) to stop."
wait $STREAMLIT_PID

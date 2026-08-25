#!/usr/bin/env bash
set -euo pipefail
cd /app

echo "=== ENTRYPOINT: printing env ==="
env | grep -E 'ZD_HEADLESS|USE_VIRTUAL_DISPLAY|NO_INITIAL_FETCH|PORT' || true

# Install Google Chrome stable (runtime) if not present.
if ! command -v google-chrome >/dev/null 2>&1; then
  echo "Downloading google-chrome-stable..."
  wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb || { echo "Failed to download chrome"; }
  apt-get update && apt-get install -y --no-install-recommends /tmp/chrome.deb || true
  rm -f /tmp/chrome.deb
fi

# Print chrome version
if command -v google-chrome >/dev/null 2>&1; then
  CHROME_VER="$(google-chrome --product-version 2>/dev/null || echo unknown)"
  echo "Google Chrome version: $CHROME_VER"
else
  echo "google-chrome not found"
fi

# Install matching chromedriver for installed Chrome major version
if command -v google-chrome >/dev/null 2>&1 && ! command -v chromedriver >/dev/null 2>&1; then
  MAJOR="$(echo $CHROME_VER | cut -d. -f1)"
  if [ -n "$MAJOR" ]; then
    echo "Resolving chromedriver for Chrome major version: $MAJOR"
    LATEST=$(curl -fsSL "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${MAJOR}" || true)
    if [ -n "$LATEST" ]; then
      echo "Downloading chromedriver v$LATEST"
      curl -fsSL -o /tmp/chromedriver_linux64.zip "https://chromedriver.storage.googleapis.com/${LATEST}/chromedriver_linux64.zip"
      unzip -o /tmp/chromedriver_linux64.zip -d /usr/local/bin
      chmod +x /usr/local/bin/chromedriver
      rm -f /tmp/chromedriver_linux64.zip
      echo "chromedriver installed: $(/usr/local/bin/chromedriver --version 2>/dev/null || true)"
    else
      echo "Could not find chromedriver LATEST_RELEASE for $MAJOR"
    fi
  else
    echo "Could not determine chrome major version"
  fi
fi

# Print a bit of debug info
echo "=== Versions ==="
google-chrome --version || true
chromedriver --version || true
python3 --version
pip show cloudscraper || true

# Run the FastAPI app via uvicorn
echo "Starting uvicorn..."
exec uvicorn server:app --host 0.0.0.0 --port "${PORT}" --loop asyncio --workers 1

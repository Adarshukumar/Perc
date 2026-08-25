# Dockerfile (use alongside perchance_server_with_pyvirtualdisplay.py and entrypoint.sh)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    PORT=7860 \
    ZD_HEADLESS=false \
    USE_VIRTUAL_DISPLAY=true \
    NO_INITIAL_FETCH=0

# Install system deps (Xvfb, fonts, curl, unzip, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-distutils \
    ca-certificates wget curl gnupg unzip \
    xvfb x11-xkb-utils xauth xdg-utils \
    libnss3 libxss1 libasound2 libatk1.0-0 libatk-bridge2.0-0 \
    libgtk-3-0 libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    fonts-liberation fonts-dejavu-core fonts-noto-cjk \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY server.py /app/server.py
COPY requirements.txt /app/requirements.txt
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Install python deps
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

EXPOSE ${PORT}

# # Health check (simple)
# HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
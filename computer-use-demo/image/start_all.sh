#!/bin/bash
set -e

# Source configuration
source ./config.sh

# Ensure Ollama is running
pgrep ollama >/dev/null || (echo "Starting Ollama..." && DISPLAY=:${DISPLAY_NUM} ollama serve &)

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null; then
        break
    fi
    sleep 1
done

# Try to pull the Mistral model if not already present
echo "Checking Mistral model..."
if ! curl -s http://localhost:11434/api/show --data '{"name":"mistral"}' | grep -q "mistral"; then
    echo "Pulling Mistral model... This may take a while..."
    curl -s http://localhost:11434/api/pull --data '{"name":"mistral"}' || echo "Warning: Failed to pull Mistral model"
fi

# Start X11 services
echo "Starting X11 services..."
./xvfb_startup.sh
./tint2_startup.sh
./mutter_startup.sh
./x11vnc_startup.sh

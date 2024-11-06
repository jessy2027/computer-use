#!/bin/bash

# Default display number
export DISPLAY_NUM=${DISPLAY_NUM:-1}
export DISPLAY=:${DISPLAY_NUM}

# Default screen dimensions
export WIDTH=${WIDTH:-1024}
export HEIGHT=${HEIGHT:-768}

# Default API provider
export API_PROVIDER=${API_PROVIDER:-ollama}

# Ollama settings
export OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
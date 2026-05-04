#!/bin/sh
# Starts the Ollama server and pulls Mistral-7B on first run if not already cached.
# Used as the entrypoint for the ollama service in docker-compose.yml.

set -e

# Start Ollama server in the background
ollama serve &
SERVE_PID=$!

# Wait until the Ollama HTTP API is accepting requests
echo "[ollama_start] Waiting for Ollama API..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "[ollama_start] Ollama API ready."

# Pull Mistral only if it is not already in the local model cache
if ! ollama list 2>/dev/null | grep -q "^mistral"; then
    echo "[ollama_start] Pulling mistral model (this may take several minutes on first run)..."
    ollama pull mistral
    echo "[ollama_start] Mistral pull complete."
else
    echo "[ollama_start] Mistral already cached — skipping pull."
fi

# Hand control back to the serve process
wait $SERVE_PID

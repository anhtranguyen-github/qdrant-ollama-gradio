#!/bin/sh

echo "Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for the server to be ready
echo "Waiting for Ollama server to be ready..."
sleep 5

echo "Pulling models..."
ollama pull nomic-embed-text:latest
ollama pull  qwen3:8b

# Keep the container running
wait $SERVER_PID



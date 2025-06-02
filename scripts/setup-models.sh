#!/bin/bash

# Wait for Ollama service to be ready
echo "Waiting for Ollama service to be ready..."
until curl -s http://ollama:11434/api/tags > /dev/null; do
  echo "Ollama not ready yet, waiting..."
  sleep 5
done

echo "Ollama is ready! Starting model setup..."

# Function to pull model with retry logic
pull_model() {
  local model=$1
  local max_retries=3
  local retry_count=0
  
  while [ $retry_count -lt $max_retries ]; do
    echo "Pulling model: $model (attempt $((retry_count + 1))/$max_retries)"
    
    if curl -X POST http://ollama:11434/api/pull \
         -H "Content-Type: application/json" \
         -d "{\"name\": \"$model\"}" \
         --max-time 1800; then  # 30 minute timeout
      echo "Successfully pulled $model"
      return 0
    else
      echo "Failed to pull $model, attempt $((retry_count + 1))/$max_retries"
      retry_count=$((retry_count + 1))
      sleep 10
    fi
  done
  
  echo "Failed to pull $model after $max_retries attempts"
  return 1
}

# Pull required models
echo "=== Pulling Chat Model ==="
pull_model "qwen2.5:7b"  # Using qwen2.5 instead of qwen3 as it's more available

echo "=== Pulling Embedding Model ==="
pull_model "nomic-embed-text:latest"

# Verify models are available
echo "=== Verifying installed models ==="
if curl -s http://ollama:11434/api/tags | grep -q "qwen2.5:7b"; then
  echo "✓ Chat model (qwen2.5:7b) is available"
else
  echo "✗ Chat model is not available"
  exit 1
fi

if curl -s http://ollama:11434/api/tags | grep -q "nomic-embed-text"; then
  echo "✓ Embedding model (nomic-embed-text) is available"
else
  echo "✗ Embedding model is not available"
  exit 1
fi

echo "=== Model setup completed successfully! ==="
echo "Available models:"
curl -s http://ollama:11434/api/tags | jq '.models[].name' 2>/dev/null || echo "Models list not available in JSON format"
#!/bin/bash
# Start the Calculate Agent server
# Run this inside WSL2 Ubuntu terminal: bash start_server.sh

cd /mnt/d/agent_learning
source venv_linux/bin/activate

# Load .env
export $(grep -v '^#' .env | xargs)

# Start Ollama (if not already running)
ollama serve &>/tmp/ollama.log &
sleep 2

# Start backend
echo "Starting Calculate Agent server (model: $OLLAMA_MODEL)..."
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000

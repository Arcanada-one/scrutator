#!/usr/bin/env bash
# Scrutator deploy script
# Target: arcana-db (Tailscale mesh only)
# Usage: ./scripts/deploy.sh [--first-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

HEALTH_URL="http://localhost:8310/health"

# Check .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example and fill in real values."
    exit 1
fi

# First deploy: kill bare-metal process if running on port 8310
if [ "${1:-}" = "--first-run" ]; then
    echo "First run: checking for bare-metal process on port 8310..."
    PID=$(lsof -ti :8310 || true)
    if [ -n "$PID" ]; then
        echo "Killing bare-metal process PID=$PID on port 8310"
        kill "$PID" || true
        sleep 2
    fi
fi

# Build and start
echo "Building and starting Scrutator..."
docker compose up -d --build

# Wait for startup
echo "Waiting 10s for startup..."
sleep 10

# Health check
echo "Health check: $HEALTH_URL"
if curl -fsS "$HEALTH_URL"; then
    echo ""
    echo "Deploy successful."
else
    echo ""
    echo "ERROR: Health check failed!"
    docker compose logs --tail=50
    exit 1
fi

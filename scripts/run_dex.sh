#!/bin/bash

# Run DEX Simulator using UV from monorepo root

echo "Starting DEX Simulator..."

# Change to project root
cd "$(dirname "$0")/.."

echo "Running from: $(pwd)"

# Run dex-simulator using UV
uv run --directory dex-simulator uvicorn main:app --host 0.0.0.0 --port 8001


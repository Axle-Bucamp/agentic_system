#!/bin/bash

# Test API endpoints with tool calls and mocked AI responses

echo "========================================"
echo "API Tool Call Testing"
echo "========================================"
echo ""

# Set environment variables
export MCP_API_KEY="sk_EvUybnfnyK3MImCECBhB0Jhhks4FsTd9H9AF3d5F32o"
export MCP_API_URL="https://forecasting.guidry-cloud.com"
export USE_MOCK_SERVICES="false"

API_KEY_PREVIEW="${MCP_API_KEY:0:10}...${MCP_API_KEY: -4}"

echo "API Key: $API_KEY_PREVIEW"
echo "API URL: $MCP_API_URL"
echo "Mock Services: $USE_MOCK_SERVICES"
echo ""

echo "Running API tests..."
python tests/test_api_tools.py

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
echo ""
read -p "Press Enter to exit..."


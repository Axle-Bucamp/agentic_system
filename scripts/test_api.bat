@echo off
REM Test API endpoints with tool calls and mocked AI responses

echo ========================================
echo API Tool Call Testing
echo ========================================
echo.

REM Set environment variables
set MCP_API_KEY=sk_EvUybnfnyK3MImCECBhB0Jhhks4FsTd9H9AF3d5F32o
set MCP_API_URL=https://forecasting.guidry-cloud.com
set USE_MOCK_SERVICES=false

echo API Key: %MCP_API_KEY:~0,10%...%MCP_API_KEY:~-4%
echo API URL: %MCP_API_URL%
echo Mock Services: %USE_MOCK_SERVICES%
echo.

echo Running API tests...
python tests/test_api_tools.py

echo.
echo ========================================
echo Test Complete
echo ========================================
pause


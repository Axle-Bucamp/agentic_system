# Test API endpoints with tool calls and mocked AI responses

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Tool Call Testing" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variables
$env:MCP_API_KEY = "sk_EvUybnfnyK3MImCECBhB0Jhhks4FsTd9H9AF3d5F32o"
$env:MCP_API_URL = "https://forecasting.guidry-cloud.com"
$env:USE_MOCK_SERVICES = "false"

$apiKeyPreview = $env:MCP_API_KEY.Substring(0, 10) + "..." + $env:MCP_API_KEY.Substring($env:MCP_API_KEY.Length - 4)

Write-Host "API Key: $apiKeyPreview" -ForegroundColor Yellow
Write-Host "API URL: $env:MCP_API_URL" -ForegroundColor Yellow
Write-Host "Mock Services: $env:USE_MOCK_SERVICES" -ForegroundColor Yellow
Write-Host ""

Write-Host "Running API tests..." -ForegroundColor Green
python tests/test_api_tools.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"


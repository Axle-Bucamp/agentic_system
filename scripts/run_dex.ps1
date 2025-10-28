# Run DEX Simulator using UV from monorepo root
Write-Host "Starting DEX Simulator..." -ForegroundColor Cyan

# Change to project root
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "Running from: $(Get-Location)" -ForegroundColor Yellow

# Run dex-simulator using UV
uv run --directory dex-simulator uvicorn main:app --host 0.0.0.0 --port 8001


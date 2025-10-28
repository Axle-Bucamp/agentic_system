@echo off
REM Run DEX Simulator using UV from monorepo root

echo Starting DEX Simulator...

REM Change to project root
cd /d "%~dp0\.."

echo Running from: %CD%

REM Run dex-simulator using UV
uv run --directory dex-simulator uvicorn main:app --host 0.0.0.0 --port 8001


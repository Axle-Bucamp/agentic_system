@echo off
REM Build script for Windows

setlocal enabledelayedexpansion

echo 🏗️ Building Agentic Trading System
echo ===================================

REM Check if UV is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] UV is not installed. Please install UV first.
    exit /b 1
)

echo [INFO] UV version:
uv --version

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker first.
    exit /b 1
)

echo [INFO] Docker is running

REM Build main API image
echo.
echo [INFO] Building main API Docker image...
docker build -t agentic-trading-api:latest -f Dockerfile .
if errorlevel 1 (
    echo [ERROR] Failed to build main API image!
    exit /b 1
)
echo [SUCCESS] Main API image built successfully

REM Build agent image
echo.
echo [INFO] Building agent Docker image...
docker build -t agentic-trading-agent:latest -f Dockerfile.agent .
if errorlevel 1 (
    echo [ERROR] Failed to build agent image!
    exit /b 1
)
echo [SUCCESS] Agent image built successfully

REM Build DEX simulator image
echo.
echo [INFO] Building DEX simulator image...
docker build -t agentic-trading-dex:latest -f services/dex-simulator/Dockerfile services/dex-simulator/
if errorlevel 1 (
    echo [ERROR] Failed to build DEX simulator image!
    exit /b 1
)
echo [SUCCESS] DEX simulator image built successfully

REM Show built images
echo.
echo [INFO] Built Docker images:
docker images | findstr agentic-trading

echo.
echo [SUCCESS] All Docker images built successfully! 🎉
echo.
echo [INFO] To run the system:
echo   docker-compose up -d
echo.
echo [INFO] To run in development mode:
echo   docker-compose -f docker-compose.dev.yml up -d

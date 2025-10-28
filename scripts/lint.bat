@echo off
REM Linting and formatting script for Windows

setlocal enabledelayedexpansion

echo 🧹 Running Linting and Formatting
echo ==================================

REM Check if UV is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] UV is not installed. Please install UV first.
    exit /b 1
)

echo [INFO] UV version:
uv --version

REM Run Ruff linting
echo.
echo [INFO] Running Ruff linting...
uv run ruff check .
if errorlevel 1 (
    echo [ERROR] Ruff linting failed!
    exit /b 1
)
echo [SUCCESS] Ruff linting passed

REM Run Ruff formatting
echo.
echo [INFO] Running Ruff formatting...
uv run ruff format --check .
if errorlevel 1 (
    echo [WARNING] Code formatting issues found. Running auto-format...
    uv run ruff format .
    echo [INFO] Code formatted automatically
) else (
    echo [SUCCESS] Code formatting is correct
)

REM Run MyPy type checking
echo.
echo [INFO] Running MyPy type checking...
uv run mypy .
if errorlevel 1 (
    echo [ERROR] MyPy type checking failed!
    exit /b 1
)
echo [SUCCESS] MyPy type checking passed

REM Run Black formatting check
echo.
echo [INFO] Running Black formatting check...
uv run black --check .
if errorlevel 1 (
    echo [WARNING] Black formatting issues found. Running auto-format...
    uv run black .
    echo [INFO] Code formatted with Black
) else (
    echo [SUCCESS] Black formatting is correct
)

REM Run isort import sorting
echo.
echo [INFO] Running isort import sorting...
uv run isort --check-only .
if errorlevel 1 (
    echo [WARNING] Import sorting issues found. Running auto-sort...
    uv run isort .
    echo [INFO] Imports sorted automatically
) else (
    echo [SUCCESS] Import sorting is correct
)

echo.
echo [SUCCESS] All linting and formatting checks passed! 🎉

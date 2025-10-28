@echo off
REM Test runner script for the agentic trading system (Windows)

setlocal enabledelayedexpansion

echo 🧪 Running Agentic Trading System Tests
echo ========================================

REM Colors for output (Windows doesn't support ANSI colors in batch, but we can use echo)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM Check if UV is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% UV is not installed. Please install UV first.
    echo %INFO% Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh
    exit /b 1
)

echo %INFO% UV version: 
uv --version

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo %ERROR% pyproject.toml not found. Please run this script from the project root.
    exit /b 1
)

REM Parse command line arguments
set "TEST_TYPE=all"
set "VERBOSE=false"
set "COVERAGE=false"
set "PARALLEL=false"

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--unit" (
    set "TEST_TYPE=unit"
    shift
    goto :parse_args
)
if "%~1"=="--integration" (
    set "TEST_TYPE=integration"
    shift
    goto :parse_args
)
if "%~1"=="--functional" (
    set "TEST_TYPE=functional"
    shift
    goto :parse_args
)
if "%~1"=="--e2e" (
    set "TEST_TYPE=e2e"
    shift
    goto :parse_args
)
if "%~1"=="--verbose" (
    set "VERBOSE=true"
    shift
    goto :parse_args
)
if "%~1"=="-v" (
    set "VERBOSE=true"
    shift
    goto :parse_args
)
if "%~1"=="--coverage" (
    set "COVERAGE=true"
    shift
    goto :parse_args
)
if "%~1"=="-c" (
    set "COVERAGE=true"
    shift
    goto :parse_args
)
if "%~1"=="--parallel" (
    set "PARALLEL=true"
    shift
    goto :parse_args
)
if "%~1"=="-p" (
    set "PARALLEL=true"
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_help
)
if "%~1"=="-h" (
    goto :show_help
)
echo %ERROR% Unknown option: %~1
echo Use --help for usage information
exit /b 1

:show_help
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   --unit          Run only unit tests
echo   --integration   Run only integration tests
echo   --functional    Run only functional tests
echo   --e2e           Run only end-to-end tests
echo   --verbose, -v   Verbose output
echo   --coverage, -c  Generate coverage report
echo   --parallel, -p  Run tests in parallel
echo   --help, -h      Show this help message
echo.
echo Examples:
echo   %0                    # Run all tests
echo   %0 --unit --coverage  # Run unit tests with coverage
echo   %0 --integration -v   # Run integration tests with verbose output
echo   %0 --parallel         # Run all tests in parallel
exit /b 0

:args_done

REM Build test command
set "TEST_CMD=uv run pytest"

REM Add test path based on type
if "%TEST_TYPE%"=="unit" (
    set "TEST_CMD=%TEST_CMD% tests/unit/"
)
if "%TEST_TYPE%"=="integration" (
    set "TEST_CMD=%TEST_CMD% tests/integration/"
)
if "%TEST_TYPE%"=="functional" (
    set "TEST_CMD=%TEST_CMD% tests/functional/"
)
if "%TEST_TYPE%"=="e2e" (
    set "TEST_CMD=%TEST_CMD% tests/e2e/"
)
if "%TEST_TYPE%"=="all" (
    set "TEST_CMD=%TEST_CMD% tests/"
)

REM Add options
if "%VERBOSE%"=="true" (
    set "TEST_CMD=%TEST_CMD% -v"
)

if "%COVERAGE%"=="true" (
    set "TEST_CMD=%TEST_CMD% --cov=api --cov=agents --cov=core --cov-report=term-missing --cov-report=html"
)

if "%PARALLEL%"=="true" (
    set "TEST_CMD=%TEST_CMD% -n auto"
)

REM Add markers for test type
if "%TEST_TYPE%"=="unit" (
    set "TEST_CMD=%TEST_CMD% -m unit"
)
if "%TEST_TYPE%"=="integration" (
    set "TEST_CMD=%TEST_CMD% -m integration"
)
if "%TEST_TYPE%"=="functional" (
    set "TEST_CMD=%TEST_CMD% -m functional"
)
if "%TEST_TYPE%"=="e2e" (
    set "TEST_CMD=%TEST_CMD% -m e2e"
)

echo %INFO% Running %TEST_TYPE% tests...
echo %INFO% Command: %TEST_CMD%
echo.

REM Run tests
%TEST_CMD%
if errorlevel 1 (
    echo %ERROR% Some tests failed! ❌
    exit /b 1
) else (
    echo %SUCCESS% All tests passed! 🎉
    
    if "%COVERAGE%"=="true" (
        echo %INFO% Coverage report generated in htmlcov/index.html
    )
    
    exit /b 0
)

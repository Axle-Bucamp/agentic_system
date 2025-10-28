@echo off
REM Development environment setup and management script for Windows

setlocal enabledelayedexpansion

REM Configuration
set "PROJECT_NAME=agentic-trading-system"
set "PYTHON_VERSION=3.11"

REM Functions
:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

:check_uv
uv --version >nul 2>&1
if errorlevel 1 (
    call :log_error "UV is not installed. Please install UV first."
    call :log_info "Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit /b 1
)
call :log_success "UV is installed"
uv --version
goto :eof

:check_python
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed."
    exit /b 1
)
call :log_success "Python is installed"
python --version
goto :eof

:setup_environment
call :log_info "Setting up development environment..."

REM Create .env file if it doesn't exist
if not exist .env (
    call :log_info "Creating .env file..."
    (
        echo # Environment
        echo ENVIRONMENT=development
        echo DEBUG=true
        echo.
        echo # API Configuration
        echo API_HOST=0.0.0.0
        echo API_PORT=8000
        echo.
        echo # Redis Configuration
        echo REDIS_HOST=localhost
        echo REDIS_PORT=6379
        echo REDIS_DB=0
        echo.
        echo # PostgreSQL Configuration
        echo POSTGRES_HOST=localhost
        echo POSTGRES_PORT=5432
        echo POSTGRES_DB=trading_system
        echo POSTGRES_USER=trading_user
        echo POSTGRES_PASSWORD=trading_pass
        echo.
        echo # External APIs
        echo MCP_API_URL=https://forecasting.guidry-cloud.com
        echo MCP_API_KEY=your_api_key_here
        echo.
        echo # Exchange API Keys
        echo MEXC_API_KEY=your_mexc_api_key
        echo MEXC_SECRET_KEY=your_mexc_secret_key
        echo.
        echo # DEX Configuration
        echo PRIVATE_KEY=your_private_key_here
        echo WALLET_ADDRESS=your_wallet_address
        echo.
        echo # Trading Configuration
        echo INITIAL_CAPITAL=10000.0
        echo MAX_POSITION_SIZE=0.20
        echo MAX_DAILY_LOSS=0.05
        echo MAX_DRAWDOWN=0.15
        echo TRADING_FEE=0.001
        echo MIN_CONFIDENCE=0.7
    ) > .env
    call :log_success ".env file created"
) else (
    call :log_info ".env file already exists"
)
goto :eof

:install_dependencies
call :log_info "Installing dependencies with UV..."
uv sync --dev
call :log_success "Dependencies installed"
goto :eof

:setup_pre_commit
call :log_info "Setting up pre-commit hooks..."
uv add --dev pre-commit
uv run pre-commit install
call :log_success "Pre-commit hooks installed"
goto :eof

:start_services
call :log_info "Starting development services..."
docker-compose up -d redis postgres
call :log_info "Waiting for services to be ready..."
timeout /t 10 /nobreak >nul
call :log_success "Services started"
goto :eof

:run_tests
call :log_info "Running tests..."
uv run pytest tests/ -v --cov=api --cov=agents --cov=core --cov-report=term-missing
call :log_success "Tests completed"
goto :eof

:run_linting
call :log_info "Running linting..."
uv run ruff check .
uv run ruff format --check .
uv run mypy .
call :log_success "Linting completed"
goto :eof

:start_api
call :log_info "Starting API server..."
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
goto :eof

:start_agents
call :log_info "Starting trading agents..."
uv run python -m agents.runner
goto :eof

:cleanup
call :log_info "Cleaning up development environment..."
docker-compose down
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
rmdir /s /q .pytest_cache 2>nul
rmdir /s /q htmlcov 2>nul
del .coverage 2>nul
call :log_success "Cleanup completed"
goto :eof

:show_help
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   setup     - Set up development environment
echo   install   - Install dependencies
echo   test      - Run tests
echo   lint      - Run linting
echo   api       - Start API server
echo   agents    - Start trading agents
echo   services  - Start development services (Redis, PostgreSQL)
echo   clean     - Clean up development environment
echo   help      - Show this help message
echo.
echo Examples:
echo   %0 setup     # Complete setup
echo   %0 test      # Run tests only
echo   %0 api       # Start API server
goto :eof

REM Main script
if "%1"=="setup" (
    call :log_info "Setting up development environment..."
    call :check_uv
    call :check_python
    call :setup_environment
    call :install_dependencies
    call :setup_pre_commit
    call :log_success "Development environment setup complete!"
) else if "%1"=="install" (
    call :check_uv
    call :install_dependencies
) else if "%1"=="test" (
    call :check_uv
    call :run_tests
) else if "%1"=="lint" (
    call :check_uv
    call :run_linting
) else if "%1"=="api" (
    call :check_uv
    call :start_api
) else if "%1"=="agents" (
    call :check_uv
    call :start_agents
) else if "%1"=="services" (
    call :start_services
) else if "%1"=="clean" (
    call :cleanup
) else if "%1"=="help" (
    call :show_help
) else if "%1"=="--help" (
    call :show_help
) else if "%1"=="-h" (
    call :show_help
) else if "%1"=="" (
    call :show_help
) else (
    call :log_error "Unknown command: %1"
    call :show_help
    exit /b 1
)

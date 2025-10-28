@echo off
REM Comprehensive test script for Agentic Trading System with mock services

echo Starting comprehensive tests with mock services...

REM Set environment variables for testing
set USE_MOCK_SERVICES=true
set MCP_API_KEY=mock_api_key
set OPENAI_API_KEY=mock_openai_key
set ENVIRONMENT=test

REM Install test dependencies
echo Installing test dependencies...
uv pip install pytest pytest-asyncio pytest-cov pytest-mock respx fakeredis

REM Run unit tests
echo Running unit tests...
uv run pytest tests/unit/ -v --cov=api --cov=agents --cov=core --cov-report=term-missing

REM Run integration tests
echo Running integration tests...
uv run pytest tests/integration/ -v

REM Run functional tests
echo Running functional tests...
uv run pytest tests/functional/ -v

REM Run all tests with coverage report
echo Running all tests with coverage...
uv run pytest tests/ -v --cov=api --cov=agents --cov=core --cov-report=html --cov-report=term-missing

echo Tests completed!
echo Coverage report available at htmlcov/index.html

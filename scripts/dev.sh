#!/bin/bash
# Development environment setup and management script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agentic-trading-system"
PYTHON_VERSION="3.11"
UV_VERSION="latest"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_uv() {
    if ! command -v uv &> /dev/null; then
        log_error "UV is not installed. Please install UV first."
        log_info "Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    log_success "UV is installed: $(uv --version)"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed."
        exit 1
    fi
    
    PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_VER" != "$PYTHON_VERSION" ]]; then
        log_warning "Python version $PYTHON_VER detected, but $PYTHON_VERSION is recommended."
    fi
    
    log_success "Python is installed: $(python3 --version)"
}

setup_environment() {
    log_info "Setting up development environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env file from template..."
        cp .env.example .env 2>/dev/null || {
            log_info "Creating basic .env file..."
            cat > .env << EOF
# Environment
ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trading_system
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_pass

# External APIs
MCP_API_URL=https://forecasting.guidry-cloud.com
MCP_API_KEY=your_api_key_here

# Exchange API Keys
MEXC_API_KEY=your_mexc_api_key
MEXC_SECRET_KEY=your_mexc_secret_key

# DEX Configuration
PRIVATE_KEY=your_private_key_here
WALLET_ADDRESS=your_wallet_address

# Trading Configuration
INITIAL_CAPITAL=10000.0
MAX_POSITION_SIZE=0.20
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.15
TRADING_FEE=0.001
MIN_CONFIDENCE=0.7
EOF
        }
        log_success ".env file created"
    else
        log_info ".env file already exists"
    fi
}

install_dependencies() {
    log_info "Installing dependencies with UV..."
    
    # Sync dependencies
    uv sync --dev
    
    log_success "Dependencies installed"
}

setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    # Install pre-commit if not already installed
    if ! command -v pre-commit &> /dev/null; then
        log_info "Installing pre-commit..."
        uv add --dev pre-commit
    fi
    
    # Install pre-commit hooks
    uv run pre-commit install
    
    log_success "Pre-commit hooks installed"
}

start_services() {
    log_info "Starting development services..."
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_warning "Docker is not running. Starting services with Docker Compose..."
        log_info "Please start Docker and run: docker-compose up -d"
        return
    fi
    
    # Start services with Docker Compose
    docker-compose up -d redis postgres
    
    log_info "Waiting for services to be ready..."
    sleep 10
    
    log_success "Services started"
}

run_tests() {
    log_info "Running tests..."
    
    # Run tests with UV
    uv run pytest tests/ -v --cov=api --cov=agents --cov=core --cov-report=term-missing
    
    log_success "Tests completed"
}

run_linting() {
    log_info "Running linting..."
    
    # Run ruff
    uv run ruff check .
    uv run ruff format --check .
    
    # Run mypy
    uv run mypy .
    
    log_success "Linting completed"
}

start_api() {
    log_info "Starting API server..."
    
    # Start FastAPI with UV
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    
    log_success "API server started on http://localhost:8000"
}

start_agents() {
    log_info "Starting trading agents..."
    
    # Start agents with UV
    uv run python -m agents.runner
    
    log_success "Trading agents started"
}

cleanup() {
    log_info "Cleaning up development environment..."
    
    # Stop Docker services
    docker-compose down
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Clean test artifacts
    rm -rf .pytest_cache/ htmlcov/ .coverage 2>/dev/null || true
    
    log_success "Cleanup completed"
}

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Set up development environment"
    echo "  install   - Install dependencies"
    echo "  test      - Run tests"
    echo "  lint      - Run linting"
    echo "  api       - Start API server"
    echo "  agents    - Start trading agents"
    echo "  services  - Start development services (Redis, PostgreSQL)"
    echo "  clean     - Clean up development environment"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Complete setup"
    echo "  $0 test      # Run tests only"
    echo "  $0 api       # Start API server"
}

# Main script
main() {
    case "${1:-help}" in
        setup)
            log_info "Setting up development environment..."
            check_uv
            check_python
            setup_environment
            install_dependencies
            setup_pre_commit
            log_success "Development environment setup complete!"
            ;;
        install)
            check_uv
            install_dependencies
            ;;
        test)
            check_uv
            run_tests
            ;;
        lint)
            check_uv
            run_linting
            ;;
        api)
            check_uv
            start_api
            ;;
        agents)
            check_uv
            start_agents
            ;;
        services)
            start_services
            ;;
        clean)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

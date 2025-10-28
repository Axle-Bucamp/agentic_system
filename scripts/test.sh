#!/bin/bash

# Test runner script for the agentic trading system

set -e

echo "🧪 Running Agentic Trading System Tests"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    print_error "UV is not installed. Please install UV first."
    print_status "Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

print_status "UV version: $(uv --version)"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Parse command line arguments
TEST_TYPE="all"
VERBOSE=false
COVERAGE=false
PARALLEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --functional)
            TEST_TYPE="functional"
            shift
            ;;
        --e2e)
            TEST_TYPE="e2e"
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --parallel|-p)
            PARALLEL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --functional    Run only functional tests"
            echo "  --e2e           Run only end-to-end tests"
            echo "  --verbose, -v   Verbose output"
            echo "  --coverage, -c  Generate coverage report"
            echo "  --parallel, -p  Run tests in parallel"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests"
            echo "  $0 --unit --coverage  # Run unit tests with coverage"
            echo "  $0 --integration -v   # Run integration tests with verbose output"
            echo "  $0 --parallel         # Run all tests in parallel"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build test command
TEST_CMD="uv run pytest"

# Add test path based on type
case $TEST_TYPE in
    "unit")
        TEST_CMD="$TEST_CMD tests/unit/"
        ;;
    "integration")
        TEST_CMD="$TEST_CMD tests/integration/"
        ;;
    "functional")
        TEST_CMD="$TEST_CMD tests/functional/"
        ;;
    "e2e")
        TEST_CMD="$TEST_CMD tests/e2e/"
        ;;
    "all")
        TEST_CMD="$TEST_CMD tests/"
        ;;
esac

# Add options
if [ "$VERBOSE" = true ]; then
    TEST_CMD="$TEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    TEST_CMD="$TEST_CMD --cov=api --cov=agents --cov=core --cov-report=term-missing --cov-report=html"
fi

if [ "$PARALLEL" = true ]; then
    TEST_CMD="$TEST_CMD -n auto"
fi

# Add markers for test type
case $TEST_TYPE in
    "unit")
        TEST_CMD="$TEST_CMD -m unit"
        ;;
    "integration")
        TEST_CMD="$TEST_CMD -m integration"
        ;;
    "functional")
        TEST_CMD="$TEST_CMD -m functional"
        ;;
    "e2e")
        TEST_CMD="$TEST_CMD -m e2e"
        ;;
esac

print_status "Running $TEST_TYPE tests..."
print_status "Command: $TEST_CMD"
echo ""

# Run tests
if eval $TEST_CMD; then
    print_success "All tests passed! 🎉"
    
    if [ "$COVERAGE" = true ]; then
        print_status "Coverage report generated in htmlcov/index.html"
    fi
    
    exit 0
else
    print_error "Some tests failed! ❌"
    exit 1
fi

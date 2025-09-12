#!/bin/bash
# VoyageurCompass Test Runner Script
# Runs the test suite with proper configuration after repository reorganization

set -e  # Exit on any error

echo "🧪 VoyageurCompass Test Suite Runner"
echo "====================================="
echo ""

# Set up environment
export DJANGO_SETTINGS_MODULE=VoyageurCompass.test_settings
export PYTHONPATH=.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run tests with consistent configuration
run_tests() {
    local test_path="$1"
    local test_name="$2"
    
    echo -e "${BLUE}Running $test_name tests...${NC}"
    python -m pytest \
        -c config/pytest.ini \
        --tb=short \
        -v \
        --maxfail=5 \
        "$test_path"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $test_name tests passed${NC}"
    else
        echo -e "${RED}❌ $test_name tests failed${NC}"
        return 1
    fi
    echo ""
}

# Function to run Django checks
run_django_checks() {
    echo -e "${BLUE}Running Django system checks...${NC}"
    
    echo "Basic system check:"
    python manage.py check
    
    echo ""
    echo "Migration check:"
    python manage.py makemigrations --check --dry-run
    
    echo -e "${GREEN}✅ Django checks passed${NC}"
    echo ""
}

# Function to run code quality checks
run_quality_checks() {
    echo -e "${BLUE}Running code quality checks...${NC}"
    
    echo "Linting with flake8:"
    if command -v flake8 &> /dev/null; then
        flake8 --max-line-length=120 --exclude=migrations,venv,node_modules,Temp Analytics Core Data || echo -e "${YELLOW}⚠️  Linting issues found${NC}"
    else
        echo -e "${YELLOW}⚠️  flake8 not installed, skipping linting${NC}"
    fi
    
    echo ""
    echo "Security scan with bandit:"
    if command -v bandit &> /dev/null; then
        bandit -r Analytics Core Data -ll || echo -e "${YELLOW}⚠️  Security issues found${NC}"
    else
        echo -e "${YELLOW}⚠️  bandit not installed, skipping security scan${NC}"
    fi
    
    echo ""
}

# Main execution
main() {
    local test_type="${1:-all}"
    local failed_tests=()
    
    case "$test_type" in
        "quick")
            echo "Running quick test suite..."
            run_tests "Analytics/tests/test_logging.py" "Logging" || failed_tests+=("Logging")
            run_tests "Core/tests/test_models.py::UserSecurityProfileTestCase::test_model_creation" "Core Models (sample)" || failed_tests+=("Core Models")
            ;;
        "core")
            echo "Running Core application tests..."
            run_tests "Core/tests/test_database_configuration.py" "Database Configuration" || failed_tests+=("Database Configuration")
            run_tests "Core/tests/test_auth_backends.py" "Authentication" || failed_tests+=("Authentication") 
            run_tests "Core/tests/test_models.py" "Core Models" || failed_tests+=("Core Models")
            ;;
        "data")
            echo "Running Data application tests..."
            run_tests "Data/tests/test_models.py::TestStockModel::test_create_stock" "Data Models (sample)" || failed_tests+=("Data Models")
            run_tests "Data/tests/test_views.py::StockViewSetTestCase::test_market_status" "Data Views (sample)" || failed_tests+=("Data Views")
            ;;
        "analytics")
            echo "Running Analytics application tests..."
            run_tests "Analytics/tests/test_logging.py" "Analytics Logging" || failed_tests+=("Analytics Logging")
            run_tests "Analytics/tests/test_ta_engine.py::TechnicalAnalysisEngineTestCase::test_engine_initialization" "TA Engine (sample)" || failed_tests+=("TA Engine")
            ;;
        "integration")
            echo "Running Integration tests..."
            run_tests "Core/tests/test_integration_workflows.py" "Integration Workflows" || failed_tests+=("Integration Workflows")
            ;;
        "all"|*)
            echo "Running full test suite..."
            
            # Django checks first
            run_django_checks || failed_tests+=("Django Checks")
            
            # Core tests
            run_tests "Core/tests/test_database_configuration.py" "Database Configuration" || failed_tests+=("Database Configuration")
            run_tests "Core/tests/test_auth_backends.py" "Authentication" || failed_tests+=("Authentication")
            run_tests "Core/tests/test_models.py" "Core Models" || failed_tests+=("Core Models")
            
            # Data tests (sample to avoid long execution)
            run_tests "Data/tests/test_models.py::TestStockModel::test_create_stock" "Data Models (sample)" || failed_tests+=("Data Models")
            
            # Analytics tests
            run_tests "Analytics/tests/test_logging.py" "Analytics Logging" || failed_tests+=("Analytics Logging")
            
            # Code quality
            run_quality_checks
            ;;
    esac
    
    # Summary
    echo "======================================="
    if [ ${#failed_tests[@]} -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed!${NC}"
        echo ""
        echo "Test suite completed successfully."
        exit 0
    else
        echo -e "${RED}❌ Some tests failed:${NC}"
        for test in "${failed_tests[@]}"; do
            echo -e "${RED}  - $test${NC}"
        done
        echo ""
        echo "Please review the failing tests above."
        exit 1
    fi
}

# Help message
show_help() {
    echo "Usage: $0 [test_type]"
    echo ""
    echo "Test types:"
    echo "  quick       - Run quick smoke tests"
    echo "  core        - Run Core application tests"
    echo "  data        - Run Data application tests" 
    echo "  analytics   - Run Analytics application tests"
    echo "  integration - Run integration tests"
    echo "  all         - Run full test suite (default)"
    echo ""
    echo "Examples:"
    echo "  $0          # Run all tests"
    echo "  $0 quick    # Run quick tests"
    echo "  $0 core     # Run core tests only"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
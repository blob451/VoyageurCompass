# VoyageurCompass Comprehensive Testing Guide

## Table of Contents
1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Unit Testing](#unit-testing)
6. [Integration Testing](#integration-testing)
7. [API Testing](#api-testing)
8. [Frontend Testing](#frontend-testing)
9. [Performance Testing](#performance-testing)
10. [Coverage Requirements](#coverage-requirements)
11. [Mocking Strategies](#mocking-strategies)
12. [Continuous Integration](#continuous-integration)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

## Overview

This document provides comprehensive information about testing in the VoyageurCompass project, covering unit tests, integration tests, API tests, frontend tests, and performance testing across all modules.

## Test Structure

### Backend Tests (Django)
```
VoyageurCompass/
├── Data/tests/
│   ├── test_models.py        # Model unit tests
│   ├── test_views.py         # API endpoint tests
│   └── test_services.py      # Service layer tests
├── Analytics/tests/
│   ├── test_views.py         # Analytics API tests
│   ├── test_engine.py        # Analytics engine tests
│   ├── test_multilingual_generation.py  # LLM translation tests
│   ├── test_financial_accuracy.py       # Financial validation tests
│   ├── test_error_handling.py           # Error handling tests
│   ├── test_load_performance.py         # Load testing
│   ├── test_security.py                 # Security validation tests
│   └── test_phase_3_multi_model_integration.py  # Multi-model LLM tests
├── Core/tests/
│   ├── test_views.py         # Core API tests
│   ├── test_auth.py          # Authentication tests
│   ├── test_integration_workflows.py    # Cross-module workflow tests
│   ├── test_api_integration.py          # API chain integration tests
│   └── test_database_integration.py     # Database integration tests
└── tests/
    └── test_integration.py   # Legacy integration tests
```

### Frontend Tests (React)
```
Design/frontend/src/
├── components/
│   ├── Layout/
│   │   ├── Layout.test.jsx
│   │   └── Navbar.test.jsx
│   └── ...
├── pages/
│   ├── DashboardPage.test.jsx
│   └── LoginPage.test.jsx
├── features/
│   ├── auth/
│   │   └── authSlice.test.js
│   └── api/
│       └── apiSlice.test.js
└── test/
    └── setup.js              # Test setup configuration
```

## Running Tests

### Quick Commands

#### Backend Tests
```bash
# Run all backend tests
pytest

# Run with coverage
pytest --cov=Data --cov=Analytics --cov=Core --cov-report=html

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m api          # API tests only
pytest -m models       # Model tests only
pytest -m performance  # Performance tests only
pytest -m "not slow"   # Exclude slow tests

# Run specific app tests
pytest Data/tests
pytest Analytics/tests
pytest Core/tests

# Parallel execution for faster testing
pytest -n auto --dist worksteal
```

#### Frontend Tests
```bash
cd Design/frontend

# Run all frontend tests
npm run test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch

# Run with UI
npm run test:ui
```

#### Comprehensive Test Execution
```bash
# Run comprehensive test runner
python Core/tools/run_tests.py --all --coverage --lint

# Backend only
python Core/tools/run_tests.py --backend --coverage

# Frontend only
python Core/tools/run_tests.py --frontend --coverage

# Fast tests only
python Core/tools/run_tests.py --test-type fast
```

## Test Categories

### 1. Unit Tests
Test individual components/functions in isolation.

**Backend Examples:**
- Model methods and properties
- Utility functions
- Service class methods

**Frontend Examples:**
- Redux reducers
- Utility functions
- Individual component rendering

### 2. Integration Tests
Test interactions between multiple components and modules.

#### Cross-Module Workflow Tests
**Location**: `Core/tests/test_integration_workflows.py`
- Complete user authentication → portfolio creation → stock analysis workflows
- Real service integration (Yahoo Finance, Ollama LLM)
- Performance benchmarking with time limits
- Concurrent user operation testing

#### API Integration Tests
**Location**: `Core/tests/test_api_integration.py`
- Authentication chain validation
- Data-Analytics API chain testing
- Portfolio management workflow testing
- Error propagation and handling validation

#### Database Integration Tests
**Location**: `Core/tests/test_database_integration.py`
- Cross-module transaction rollback testing
- Foreign key cascade operation validation
- Concurrent database operation testing
- Performance benchmarking with specific targets

### 3. LLM-Specific Tests
Comprehensive testing of the Local Large Language Model system.

#### Multi-Model Integration
- Tests all three LLM models (phi3:3.8b, llama3.1:8b, qwen2:latest)
- Model selection logic validation
- Fallback mechanism testing

#### Multilingual Generation
- Translation quality validation for French and Spanish
- Financial terminology preservation
- Cultural context adaptation

#### Financial Accuracy
- Score-to-recommendation alignment (100% accuracy requirement)
- Boundary condition testing at critical thresholds
- Technical indicator interpretation accuracy

#### Error Handling & Fallback
- Ollama service failure scenarios
- Circuit breaker functionality
- Graceful degradation testing

#### Load Performance
- Concurrent request handling (50+ users)
- Resource utilisation monitoring
- Performance degradation analysis

#### Security Validation
- Input sanitisation testing
- Access control verification
- Rate limiting protection

### 4. API Tests
Test REST API endpoints end-to-end.

**Coverage:**
- Authentication endpoints
- CRUD operations
- Error handling
- Permissions and authorisation
- Data validation

### 5. Component Tests
Test React components with user interactions.

**Coverage:**
- Component rendering
- User interactions (clicks, form submissions)
- Props handling
- State management
- Event handlers

## Unit Testing

### Backend Unit Testing
```python
@pytest.mark.django_db
class TestStockModel:
    def test_create_stock(self):
        stock = Stock.objects.create(symbol='AAPL', short_name='Apple')
        assert stock.symbol == 'AAPL'
    
    def test_get_latest_price(self):
        # Test implementation
        pass
```

### Test Markers
```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test  
@pytest.mark.api          # API endpoint test
@pytest.mark.models       # Model test
@pytest.mark.slow         # Slow-running test
@pytest.mark.performance  # Performance test
```

## Integration Testing

### Performance Standards

| Test Category | Performance Target | Measurement |
|---------------|-------------------|-------------|
| Complete Integration Suite | < 5 minutes | Total execution time |
| Individual Workflow Tests | < 3 minutes | Per test method |
| API Chain Tests | < 2 seconds | Per API operation |
| Database Operations | < 1 second | Per complex query |
| Bulk Operations | < 10 seconds | Per 1000 records |

### Integration Test Example
```python
@pytest.mark.integration
class UserJourneyIntegrationTest(TransactionTestCase):
    """Complete user journey integration tests."""
    
    def test_complete_user_analysis_workflow(self):
        """Test complete workflow: Auth → Portfolio → Analysis → LLM."""
        start_time = time.time()
        
        # Step 1: User Authentication
        # Step 2: Portfolio Creation  
        # Step 3: Stock Data Synchronisation
        # Step 4: Add Stock to Portfolio
        # Step 5: Technical Analysis Generation
        # Step 6: LLM Explanation Generation
        # Step 7: Portfolio Performance Analysis
        
        total_time = time.time() - start_time
        self.assertLess(total_time, 180.0, "Complete workflow should finish within 3 minutes")
```

## API Testing

### API Testing Patterns
```python
class StockAPITestCase(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='test', password='test')
        
    def test_authenticated_request(self):
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/stocks/')
        self.assertEqual(response.status_code, 200)
```

## Frontend Testing

### Test Configuration
Frontend tests use Vitest with React Testing Library:
- **Environment**: jsdom
- **Coverage**: Minimum 70% overall, 80% for components
- **Mocking**: Vitest mocks for API calls and external dependencies
- **Setup**: Global test setup in `src/test/setup.js`

### Component Test Patterns
```javascript
describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent prop="value" />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })
  
  it('handles user interaction', async () => {
    const user = userEvent.setup()
    render(<MyComponent />)
    
    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Updated Text')).toBeInTheDocument()
  })
})
```

### Redux Testing
```javascript
describe('authSlice', () => {
  it('should handle login success', () => {
    const initialState = { user: null, isAuthenticated: false }
    const action = loginSuccess({ user: { username: 'test' } })
    const state = authReducer(initialState, action)
    
    expect(state.isAuthenticated).toBe(true)
    expect(state.user.username).toBe('test')
  })
})
```

## Performance Testing

### Load Testing
```python
@pytest.mark.performance
def test_concurrent_database_operations(self):
    """Test concurrent database operations performance."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(create_stock_data, i) for i in range(1, 11)]
        results = [future.result() for future in as_completed(futures)]
    
    # Verify performance targets
    avg_execution_time = statistics.mean(execution_times)
    self.assertLess(avg_execution_time, 2.0, "Average execution time should be < 2s")
```

### Memory Testing
```python
def test_memory_usage():
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Perform operations
    
    gc.collect()
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## Coverage Requirements

### Backend Coverage Thresholds
- **Overall**: 70% minimum
- **Critical Components** (models, views): 80% minimum
- **Business Logic** (services, utilities): 85% minimum

### Frontend Coverage Thresholds
- **Overall**: 70% minimum
- **Components**: 80% minimum  
- **Redux Logic**: 75% minimum
- **Utilities**: 85% minimum

### Coverage Reports
Coverage reports are generated in:
- **Backend**: `htmlcov/index.html`
- **Frontend**: `coverage/index.html`

## Mocking Strategies

### Backend Mocking
```python
from unittest.mock import patch, MagicMock

@patch('Data.services.yahoo_finance.yahoo_finance_service.get_stock_data')
def test_stock_sync(self, mock_get_data):
    mock_get_data.return_value = {'symbol': 'AAPL', 'price': 150.00}
    # Test implementation
```

### Frontend Mocking
```javascript
import { vi } from 'vitest'

// Mock API calls
global.fetch = vi.fn()
fetch.mockResolvedValue({
  ok: true,
  json: async () => ({ data: 'test' })
})

// Mock components
vi.mock('recharts', () => ({
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>
}))
```

## Continuous Integration

### GitHub Actions CI/CD Pipeline

#### Quality Gates
1. **Test Coverage**: Minimum 80% unit test coverage, 65% integration coverage
2. **Performance Benchmarks**: All tests must complete within target times
3. **Security Scanning**: Automated bandit and safety checks
4. **Code Quality**: Linting and formatting validation

#### Automated Test Execution
```yaml
name: Tests
on: [push, pull_request]
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov --cov-report=xml -n auto --dist worksteal
      
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
      - name: Install dependencies
        run: cd Design/frontend && npm ci
      - name: Run tests
        run: cd Design/frontend && npm run test:coverage

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: pytest -m integration -v --tb=short -n auto --maxfail=5
```

### Test Data Management

#### Fixtures and Factories
```python
# conftest.py
import pytest
from django.contrib.auth.models import User

@pytest.fixture
def test_user():
    return User.objects.create_user(
        username='testuser',
        password='testpass123',
        email='test@example.com'
    )

@pytest.fixture
def test_stock():
    return Stock.objects.create(
        symbol='AAPL',
        short_name='Apple Inc.',
        sector='Technology'
    )
```

## Best Practices

### General Guidelines
1. **Write tests first** (TDD approach when possible)
2. **Test behaviour, not implementation**
3. **Use descriptive test names**
4. **Keep tests independent and isolated**
5. **Mock external dependencies**
6. **Test edge cases and error conditions**

### Backend Best Practices
1. Use Django's TestCase for database tests
2. Use APITestCase for API endpoint tests
3. Mock external API calls (Yahoo Finance, Ollama)
4. Test permissions and authentication
5. Use factories for test data creation

### Frontend Best Practices
1. Test component behaviour, not internal state
2. Use screen queries over container queries
3. Test user interactions with userEvent
4. Mock API calls and external dependencies
5. Test error boundaries and loading states

### Integration Testing Best Practices

#### 1. Isolation and Independence
- Each test should be independent and not rely on other tests
- Use database transactions or rollback mechanisms for data isolation
- Clean up test data after execution

#### 2. Real Service Integration
- Test with actual external services when possible (Yahoo Finance, Redis, PostgreSQL)
- Use mocks only when external services are unavailable
- Implement graceful degradation for service unavailability

#### 3. Performance Consciousness
- Set realistic but challenging performance targets
- Monitor execution times and identify slow tests
- Optimise database queries and API calls

#### 4. Error Handling Validation
- Test error propagation across module boundaries
- Validate proper HTTP status codes and error messages
- Ensure graceful handling of external service failures

## Troubleshooting

### Common Backend Issues
- **Database errors**: Ensure proper test database setup
- **Import errors**: Check Django settings and app configuration
- **Async issues**: Use proper async test decorators

### Common Frontend Issues
- **Component not found**: Check test setup and imports
- **Async operations**: Use waitFor and findBy queries
- **Mock issues**: Ensure mocks are properly configured

### Performance Issues
- **Slow tests**: Use pytest markers to skip slow tests during development
- **Memory leaks**: Check for proper cleanup in tearDown methods
- **Database locks**: Ensure tests don't interfere with each other

### Integration Testing Issues

#### Test Environment Setup
```bash
# Database connection issues
export DATABASE_URL="postgresql://user:password@localhost:5432/test_db"
python manage.py migrate --verbosity=0

# Redis connection issues  
export REDIS_URL="redis://localhost:6379/0"

# Django settings
export DJANGO_SETTINGS_MODULE="VoyageurCompass.test_settings"
```

#### Performance Issues
- **Slow Database Queries**: Add indexes, optimise ORM queries, use select_related/prefetch_related
- **External Service Timeouts**: Implement proper timeout handling and mock services for CI/CD
- **Memory Usage**: Use database transactions, clean up test data, optimise fixture usage

#### Parallel Execution Issues
- **Database Conflicts**: Use unique test data prefixes, separate test databases per worker
- **Resource Contention**: Limit concurrent workers based on system resources
- **Test Dependencies**: Ensure tests are truly independent and don't share mutable state

### Debugging Tests

#### Backend Debugging
```bash
# Run with pdb debugger
pytest --pdb

# Run specific test with verbose output
pytest -v Data/tests/test_models.py::TestStock::test_create_stock

# Show print statements
pytest -s
```

#### Frontend Debugging
```bash
# Run in debug mode
npm run test -- --reporter=verbose

# Run single test file
npm run test Layout.test.jsx
```

## Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --reuse-db
    -n auto
    --dist worksteal
    --maxfail=5
markers =
    unit: marks tests as unit tests
    integration: marks tests as integration tests
    performance: marks tests as performance benchmarks
    slow: marks tests as slow (deselect with '-m "not slow"')
    api: marks tests as API-focused tests
    models: marks tests as model-focused tests
    database: marks tests as database-focused tests
```

## Maintenance and Updates

### Regular Review Process
1. **Monthly Performance Review**: Analyse test execution times and identify optimisations
2. **Quarterly Integration Audit**: Review cross-module integration points and update tests
3. **Release Testing**: Full integration test suite execution before production deployments

### Continuous Improvement
- Monitor test failure patterns and improve test reliability
- Optimise slow tests through better data management or parallelisation
- Expand integration test coverage for new features and modules

## Conclusion

This comprehensive testing framework provides validation for all aspects of the VoyageurCompass system, from individual components to complete user workflows. The framework ensures code quality, reliability, and performance throughout the development process.

For questions or issues with tests, refer to the project documentation or contact the development team.